"""Двумерные уравнения мелкой воды на карте залива (docs/theory.md, раздел 2.5).

    H_t + (H·u)_x + (H·v)_y = 0
    (H·u)_t + (H·u² + g·H²/2)_x + (H·u·v)_y = −g·H·z_x + f·H·v + τx/ρ − H·∂x p_a/ρ − C_f·|u|·u
    (H·v)_t + (H·u·v)_x + (H·v² + g·H²/2)_y = −g·H·z_y − f·H·u + τy/ρ − H·∂y p_a/ρ − C_f·|u|·v

Схема та же, что в saint_venant: конечные объёмы, поток HLL, MUSCL κ = 1/3,
гидростатическая реконструкция; потоки по x и по y считаются одной функцией.
Суша — стенка. На западной границе открытая граница: приходит длинная волна,
отражённая уходит свободно. Ветер, Кориолис, перепад давления, донное трение.

Проверки из TASKS.md запускаются командой:  python -m backend.models.shallow_2d
"""
import time
from pathlib import Path

import numpy as np

from backend.contract import Params, Result, WaveModel
from backend.models.saint_venant import G, H_DRY, _hll, _minmod, _setup, _velocity

CFL = 0.8             # число Куранта для суммы по двум направлениям (SSP-RK2 устойчив до 1)
F_CORIOLIS = 1.26e-4  # параметр Кориолиса на широте 60°, 1/с
RHO, RHO_AIR, C_D = 1000.0, 1.25, 2e-3   # плотности воды и воздуха, кг/м³; коэффициент C_d
H_WIND = 0.1          # ветер действует только при глубине больше этой, м
MAX_FRAMES = 60       # больше кадров карты браузер не потянет
NY_CHANNEL = 4        # число рядов по y в режимах flat и slope
DATA = Path(__file__).resolve().parents[2] / "data" / "gulf_bathymetry.npz"
SPB = (30.26, 59.93)  # Петербург, пост «Горный институт» (долгота, широта)
NEVA_BAY_LON = 29.6   # Невская губа — восточнее Котлина (Кронштадт ≈ 29,77° в. д.)

EXTRA = [
    {"name": "wind_speed", "label": "Скорость ветра", "unit": "м/с", "default": 0.0, "min": 0.0, "max": 40.0},
    {"name": "wind_dir", "label": "Откуда дует ветер (270 — западный)", "unit": "°",
     "default": 270.0, "min": 0.0, "max": 360.0},
    {"name": "friction", "label": "Донное трение C_f", "unit": "", "default": 0.0025, "min": 0.0, "max": 0.01},
    {"name": "dp", "label": "Перепад давления запад − восток", "unit": "гПа",
     "default": 0.0, "min": -50.0, "max": 50.0},
    {"name": "coriolis", "label": "Кориолис (1 — вкл, 0 — выкл)", "unit": "", "default": 1, "min": 0, "max": 1},
]


# --- Постановка --------------------------------------------------------------

def load_gulf():
    """Карта глубин Влада: x, y (м, центры ячеек), h[iy, ix] (м, суша — 0), lon, lat."""
    if not DATA.exists():
        raise NotImplementedError("Нет data/gulf_bathymetry.npz (ветка vlad/analysis)")
    with np.load(DATA, allow_pickle=False) as d:
        return tuple(np.asarray(d[k], float) for k in ("x", "y", "h", "lon", "lat"))


def gauge(lon, lat, h, point=SPB):
    """Индексы (iy, ix) мокрой ячейки, ближайшей к точке (долгота, широта)."""
    dist = ((lon[None, :] - point[0]) * np.cos(np.radians(60))) ** 2 + (lat[:, None] - point[1]) ** 2
    return np.unravel_index(np.argmin(np.where(h > 0, dist, np.inf)), h.shape)


def _setup_2d(p: Params):
    """Вернуть x, y, h, η₀, u₀ и режим границ: 'gulf', 'periodic' или 'wall'."""
    bottom = getattr(p, "bottom", "gulf")
    if bottom == "gulf":
        x, y, h, _, _ = load_gulf()
        return x, y, h, np.zeros_like(h), np.zeros_like(h), "gulf"
    # Проверочный канал: одномерная постановка saint_venant, вытянутая по y.
    x, h1, eta1, u1, periodic = _setup(p)
    y = (np.arange(NY_CHANNEL) + 0.5) * (x[1] - x[0])
    rows = lambda a: np.repeat(a[None, :], NY_CHANNEL, axis=0)
    return x, y, rows(h1), rows(eta1), rows(u1), "periodic" if periodic else "wall"


# --- Расчётное ядро ----------------------------------------------------------

def _mesh(act, bc):
    """Список граней для расчёта потоков; строится один раз на расчёт.

    Ячейки нумеруются как в act.ravel(); за ними идут фиктивная ячейка суши (DUMMY)
    и по две ячейки-призрака на каждый ряд открытой западной границы. Берутся только
    грани, у которых хотя бы с одной стороны вода, — по x и по y вместе, так что
    суша, а это половина карты, в расчёт не попадает. Для каждой грани хранятся
    ячейки LL, L, R, RR вдоль нормали: по ним идёт реконструкция с обеих сторон.
    """
    ny, nx = act.shape
    n = ny * nx
    dummy = n
    idx = np.arange(n).reshape(ny, nx)
    open_rows = act[:, 0] if bc == "gulf" else np.zeros(ny, dtype=bool)
    g1 = np.where(open_rows, n + 1 + np.arange(ny), dummy)        # призрак у границы
    g2 = np.where(open_rows, n + 1 + ny + np.arange(ny), dummy)   # призрак дальше
    act_ext = np.concatenate((act.ravel(), [False], open_rows, open_rows))

    # Грани по x: k = 0..nx между ячейками k−1 и k каждого ряда.
    ext = np.full((ny, nx + 4), dummy)
    ext[:, 2:-2] = idx
    if bc == "periodic":
        ext[:, :2], ext[:, -2:] = idx[:, -2:], idx[:, :2]
    else:
        ext[:, 0], ext[:, 1] = g2, g1
    k = np.arange(nx if bc == "periodic" else nx + 1)
    fx = np.stack([ext[:, k + s] for s in range(4)], axis=-1).reshape(-1, 4)
    # Грани по y: стенки на южном и северном краю.
    ext = np.full((ny + 4, nx), dummy)
    ext[2:-2] = idx
    k = np.arange(ny + 1)
    fy = np.stack([ext[k + s, :] for s in range(4)], axis=-1).reshape(-1, 4)

    faces = np.concatenate((fx, fy))
    is_x = np.r_[np.ones(len(fx), bool), np.zeros(len(fy), bool)]
    used = act_ext[faces[:, 1]] | act_ext[faces[:, 2]]
    faces, is_x = faces[used], is_x[used]

    # Для каждой ячейки воды — номера её граней: запад, восток, юг, север.
    cells = idx[act]
    side = np.full((n + 1 + 2 * ny, 4), -1)
    num = np.arange(len(faces))
    side[faces[is_x, 2], 0] = num[is_x]
    side[faces[is_x, 1], 1] = num[is_x]
    side[faces[~is_x, 2], 2] = num[~is_x]
    side[faces[~is_x, 1], 3] = num[~is_x]
    # Индексы в плоский массив S.ravel() (H, w, u, v): для граней по y скорости
    # переставлены, чтобы на любой грани шли (H, w, u_поперёк, u_вдоль).
    n_ext = n + 1 + 2 * ny
    var = np.where(is_x[None, :], np.arange(4)[:, None], np.array([0, 1, 3, 2])[:, None])
    take = [var * n_ext + faces[None, :, k] for k in range(4)]
    return dict(faces=faces, take=take, cols=[faces[:, k] for k in range(4)], is_x=is_x, wall=~(act_ext[faces[:, 1]] & act_ext[faces[:, 2]]),
                act_ext=act_ext, cells=cells, side=side[cells], g1=g1[open_rows],
                g2=g2[open_rows], open_rows=open_rows, n_ext=n + 1 + 2 * ny)


def _fluxes(mesh, S, dx, dy):
    """Производные (H, H·u, H·v) в ячейках воды от потоков через все грани.

    S — массив (H, w = H + z, u, v) на расширенном наборе ячеек. Схема та же, что
    в saint_venant: MUSCL κ = 1/3 с ограничителем Корена (без реконструкции у
    берега и кромки воды), гидростатическая реконструкция, поток HLL; расход
    вдоль грани переносится по потоку. Грань с сушей — стенка.
    """
    flat = S.ravel()
    LL, L, R, RR = (flat.take(k) for k in mesh["take"])   # (H, w, u_n, u_t) × грань
    wet = (S[0] > H_DRY) & mesh["act_ext"]
    wLL, wL, wR, wRR = (wet[c] for c in mesh["cols"])
    d1, d2, d3 = L - LL, R - L, RR - R               # разности вдоль нормали к грани
    # Те же приращения Корена, что в saint_venant._koren, но только к нужной грани.
    inc_l = 0.5 * _minmod(2 * d1, (d1 + 2 * d2) / 3, 2 * d2)      # восточная грань ячейки L
    inc_r = -0.5 * _minmod(2 * d3, (2 * d2 + d3) / 3, 2 * d2)     # западная грань ячейки R
    hl, wl, ul, tl = L + inc_l * (wLL & wL & wR)
    hr, wr, ur, tr = R + inc_r * (wL & wR & wRR)
    zl, zr = wl - hl, wr - hr

    zs = np.maximum(zl, zr)
    hls = np.maximum(0.0, wl - zs)
    hrs = np.maximum(0.0, wr - zs)
    f_h, f_q = _hll(hls, ul, hrs, ur)
    f_t = f_h * np.where(f_h > 0, tl, tr)
    wall = mesh["wall"]
    f_h = np.where(wall, 0.0, f_h)
    f_t = np.where(wall, 0.0, f_t)
    # Поток нормального импульса, каким его видит ячейка слева и справа: у стенки
    # остаётся давление g·H²/2 со стороны воды, это же даёт гидростатическая
    # поправка, поэтому покой у берега сохраняется точно.
    f_for_l = np.where(wall, 0.5 * G * hl ** 2, f_q + 0.5 * G * (hl ** 2 - hls ** 2))
    f_for_r = np.where(wall, 0.5 * G * hr ** 2, f_q + 0.5 * G * (hr ** 2 - hrs ** 2))

    w_, e_, s_, n_ = mesh["side"].T
    dH = -(f_h[e_] - f_h[w_]) / dx - (f_h[n_] - f_h[s_]) / dy
    src_x = -0.5 * G * (hl[e_] + hr[w_]) * (zl[e_] - zr[w_]) / dx
    src_y = -0.5 * G * (hl[n_] + hr[s_]) * (zl[n_] - zr[s_]) / dy
    dqx = src_x - (f_for_l[e_] - f_for_r[w_]) / dx - (f_t[n_] - f_t[s_]) / dy
    dqy = src_y - (f_t[e_] - f_t[w_]) / dx - (f_for_l[n_] - f_for_r[s_]) / dy
    return dH, dqx, dqy


def solve_2d(x, y, h, eta0, u0, bc, t_frames, a=0.0, wavelength=1.0, extra=None, return_uv=False):
    """Решить двумерные уравнения и вернуть (кадры η, наибольшее за всё время η).

    return_uv=True — третьим и четвёртым элементом ещё кадры скоростей u и v (м/с),
    для визуализации течений (showcase).

    bc: 'gulf' — суша (h ≤ 0) и края карты — стенки, запад открыт, туда приходит
    горб η = a·sin²(π·t/T), T = wavelength/√(g·h_запад); 'periodic' — периодика по x,
    стенки по y; 'wall' — стенки со всех сторон. На суше η = 0.
    """
    e = {p["name"]: p["default"] for p in EXTRA}
    e.update(extra or {})
    dx, dy = x[1] - x[0], y[1] - y[0]
    act = h > 0 if bc == "gulf" else np.ones(h.shape, dtype=bool)
    z = -h
    H = np.where(act, np.maximum(h + eta0, 0.0), 0.0)
    qx, qy = H * u0, np.zeros_like(H)

    f = F_CORIOLIS if e["coriolis"] else 0.0
    ang = np.radians(e["wind_dir"])          # откуда дует; ветер направлен в обратную сторону
    wind = RHO_AIR * C_D * e["wind_speed"] ** 2 / RHO
    tau_x, tau_y = -wind * np.sin(ang), -wind * np.cos(ang)
    grad_p = e["dp"] * 100.0 / (x[-1] - x[0]) / RHO   # −∂x p_a / ρ
    cf = e["friction"]
    period = wavelength / np.sqrt(G * h[act[:, 0], 0].mean()) if bc == "gulf" else 1.0

    def ghost(H, u, v, t):
        """Западная открытая граница: инвариант R+ из приходящей волны, R− изнутри."""
        H0, h0 = H[:, 0], np.maximum(h[:, 0], 0.0)
        eta_in = a * np.sin(np.pi * min(t, period) / period) ** 2
        r_out = u[:, 0] - 2 * np.sqrt(G * H0)
        r_in = 4 * np.sqrt(G * (h0 + eta_in)) - 2 * np.sqrt(G * h0)
        c = np.maximum(0.25 * (r_in - r_out), 0.0)
        return c * c / G, 0.5 * (r_in + r_out), v[:, 0], z[:, 0]

    mesh = _mesh(act, bc)
    cells = mesh["cells"]
    S = np.zeros((4, mesh["n_ext"]))           # H, w, u, v на расширенном наборе ячеек
    z_ext = np.zeros(mesh["n_ext"])
    z_ext[:z.size] = z.ravel()
    rows = mesh["open_rows"]
    z_ext[mesh["g1"]] = z_ext[mesh["g2"]] = z[rows, 0]

    def rhs(H, qx, qy, u, v, t):
        S[0, :H.size], S[2, :H.size], S[3, :H.size] = H.ravel(), u.ravel(), v.ravel()
        if bc == "gulf":
            gh, gu, gv, _ = ghost(H, u, v, t)
            for g in (mesh["g1"], mesh["g2"]):
                S[0, g], S[2, g], S[3, g] = gh[rows], gu[rows], gv[rows]
        S[1] = S[0] + z_ext
        dH, dqx, dqy = _fluxes(mesh, S, dx, dy)
        Hc = H.ravel()[cells]
        windy = Hc > H_WIND
        dqx = dqx + f * qy.ravel()[cells] + Hc * grad_p + np.where(windy, tau_x, 0.0)
        dqy = dqy - f * qx.ravel()[cells] + np.where(windy, tau_y, 0.0)
        out = np.zeros((3, H.size))
        out[0, cells], out[1, cells], out[2, cells] = dH, dqx, dqy
        return out.reshape((3,) + H.shape)

    def clean(H, qx, qy, dt):
        """Глубина не отрицательна; донное трение — неявно, чтобы не было неустойчивости."""
        H = np.maximum(H, 0.0)
        u, v = _velocity(H, qx), _velocity(H, qy)
        damp = 1.0 / (1.0 + dt * cf * np.sqrt(u * u + v * v) / np.maximum(H, H_DRY))
        u, v = u * damp, v * damp
        return H, H * u, H * v, u, v

    eta = lambda H: np.where(act, H + z, 0.0)
    frames = np.empty((len(t_frames),) + h.shape)
    frames[0] = eta(H)
    eta_max = frames[0].copy()
    t = float(t_frames[0])
    u, v = _velocity(H, qx), _velocity(H, qy)
    if return_uv:
        frames_u, frames_v = np.zeros_like(frames), np.zeros_like(frames)
        frames_u[0], frames_v[0] = u, v
    for j in range(1, len(t_frames)):
        while t < t_frames[j] - 1e-9:
            c = np.sqrt(G * H)
            rate = np.max((np.abs(u) + c) / dx + (np.abs(v) + c) / dy)
            dt = min(CFL / max(rate, 1e-12), t_frames[j] - t)
            k = rhs(H, qx, qy, u, v, t)
            s = clean(H + dt * k[0], qx + dt * k[1], qy + dt * k[2], dt)
            k = rhs(*s, t + dt)
            H, qx, qy, u, v = clean(0.5 * (H + s[0] + dt * k[0]), 0.5 * (qx + s[1] + dt * k[1]),
                                    0.5 * (qy + s[2] + dt * k[2]), 0.5 * dt)
            t += dt
            if not (np.all(np.isfinite(H)) and np.all(np.isfinite(qx)) and np.all(np.isfinite(qy))):
                raise RuntimeError(f"Двумерный расчёт разошёлся при t = {t:.0f} с")
            eta_max = np.maximum(eta_max, eta(H))
        frames[j] = eta(H)
        if return_uv:
            frames_u[j], frames_v[j] = u, v
    if return_uv:
        return frames, eta_max, frames_u, frames_v
    return frames, eta_max


class Shallow2DModel(WaveModel):
    id = "shallow_2d"
    name = "Мелкая вода 2D (залив)"
    description = (
        "Уравнения Сен-Венана на карте восточной части Финского залива и Невской губы "
        "(GEBCO, без дамбы). С запада приходит длинная волна a·sin²(πt/T), T = λ/√(g·h). "
        "Ветер, Кориолис, перепад давления, донное трение. Берег — стенка, дисперсии нет. "
        "Для залива берите λ ~ 100–500 км и t_end — часы."
    )
    dim = 2
    extra_params = EXTRA

    def solve(self, p: Params) -> Result:
        x, y, h, eta0, u0, bc = _setup_2d(p)
        t = np.linspace(0.0, p.t_end, min(p.n_frames, MAX_FRAMES))
        frames, _ = solve_2d(x, y, h, eta0, u0, bc, t, p.a, p.wavelength, getattr(p, "extra", None))
        # η с точностью 0,1 мм: ответ сервера для карты 150 × 50 × 60 кадров втрое короче.
        fields = dict(x=x.tolist(), t=t.tolist(), eta=np.round(frames, 4).tolist(), h=h.tolist())
        if "y" in getattr(Result, "__dataclass_fields__", {}):
            fields["y"] = y.tolist()
        res = Result(**fields)
        res.y = y.tolist()      # пока в контракте нет поля y
        return res


# --- Проверки «готово, когда» из TASKS.md -----------------------------------

HOUR = 3600.0


def _gulf_run(a, wavelength, t_end, **extra):
    x, y, h, lon, lat = load_gulf()
    t = np.linspace(0.0, t_end, MAX_FRAMES)
    start = time.perf_counter()
    frames, eta_max = solve_2d(x, y, h, np.zeros_like(h), np.zeros_like(h), "gulf", t, a, wavelength, extra)
    return h, lon, lat, frames, eta_max, time.perf_counter() - start


def _check_channel() -> float:
    """Ровное дно, без ветра, Кориолиса и трения: ряд двумерного расчёта против saint_venant.

    Схема та же, но шаг по времени разный (в 2D он считается по сумме двух
    направлений), поэтому совпадение не до машинной точности, а до ошибки схемы.
    """
    from backend.models.saint_venant import SaintVenantModel
    p = Params(a=0.5, wavelength=100.0, depth=5.0, L=1000.0, t_end=10.0, nx=1000, n_frames=2)
    p.bottom = "flat"
    x, y, h, eta0, u0, bc = _setup_2d(p)
    frames, _ = solve_2d(x, y, h, eta0, u0, bc, np.array([0.0, p.t_end]),
                         extra={"coriolis": 0, "friction": 0.0})
    ref = np.array(SaintVenantModel().solve(p).eta[-1])
    return float(np.abs(frames[-1] - ref[None, :]).max() / p.a)


def _check_rest() -> float:
    """Покой на карте залива: a = 0, без ветра, 6 часов."""
    return float(np.abs(_gulf_run(0.0, 300e3, 6 * HOUR)[3]).max())


def _check_wind_setup() -> tuple[float, float]:
    """Постоянный ветер над закрытым каналом 20 км × 5 м: перекос уровня против теории.

    В равновесии ∂η/∂x = τ/(ρ·g·H) (раздел 2.5). Перекос усредняем по 6–12 ч, чтобы
    убрать сейши, которые возбуждает включение ветра.
    """
    p = Params(a=0.0, wavelength=100.0, depth=5.0, L=20e3, t_end=12 * HOUR, nx=200, n_frames=2)
    p.bottom, p.depth_shore = "slope", 5.0          # ровное дно, стенки на концах
    x, y, h, eta0, u0, bc = _setup_2d(p)
    t = np.linspace(0.0, p.t_end, 121)
    frames, _ = solve_2d(x, y, h, eta0, u0, bc, t, extra={"wind_speed": 25.0, "coriolis": 0})
    rise = (frames[:, 1, -1] - frames[:, 1, 0])[t >= 6 * HOUR].mean()
    tau = RHO_AIR * C_D * 25.0 ** 2
    theory = tau / (RHO * G * p.depth) * (x[-1] - x[0])
    return float(rise), float(theory)


def _zone(h, lon, eta_max, threshold):
    """Ячейки, где наибольшее η/h за расчёт больше порога, и доля их в Невской губе."""
    zone = (h > 0) & (eta_max > threshold * h)
    east = np.broadcast_to(lon[None, :], h.shape)[zone] > NEVA_BAY_LON
    return int(zone.sum()), float(east.mean()) if zone.any() else 0.0


def _check_wave():
    """Длинная волна 0,3 м без ветра: рост у Петербурга и зона a/h > 0,2 и > 0,3."""
    a = 0.3
    h, lon, lat, _, eta_max, _ = _gulf_run(a, 300e3, 12 * HOUR)
    g = gauge(lon, lat, h)
    green = (h[h[:, 0] > 0, 0].mean() / h[g]) ** 0.25
    return float(eta_max[g] / a), float(green), _zone(h, lon, eta_max, 0.2), _zone(h, lon, eta_max, 0.3)


def _check_surge():
    """Волна 0,5 м и западный ветер 25 м/с: подъём у Петербурга, зона a/h > 0,2, время."""
    h, lon, lat, _, eta_max, elapsed = _gulf_run(0.5, 300e3, 12 * HOUR, wind_speed=25.0, wind_dir=270.0)
    return float(eta_max[gauge(lon, lat, h)]), _zone(h, lon, eta_max, 0.2), elapsed


if __name__ == "__main__":
    ch = _check_channel()
    print(f"Канал с ровным дном: отличие от saint_venant {ch:.1e}·a (нужно меньше 1%)")
    rise, theory = _check_wind_setup()
    print(f"Ветер 25 м/с над каналом 20 км × 5 м: перекос уровня {rise:.3f} м, "
          f"по формуле τ·L/(ρ·g·H) {theory:.3f} м (допуск 10%)")
    rest = _check_rest()
    print(f"Покой на карте залива за 6 ч: наибольшее |η| = {rest:.1e} м (нужно меньше 1e-6 м)")
    gain, green, (n2, s2), (n3, s3) = _check_wave()
    print(f"Волна 0,3 м без ветра: у Петербурга выросла в {gain:.1f} раза "
          f"(закон Грина без учёта сужения даёт {green:.1f}; ожидаем 1,5–8)")
    print(f"  зона a/h > 0,2: {n2} ячеек, в Невской губе (восточнее {NEVA_BAY_LON}° в. д.) {s2:.0%}; "
          f"a/h > 0,3: {n3} ячеек, {s3:.0%} (нужно больше 80%)")
    surge, (nw, sw), elapsed = _check_surge()
    print(f"Волна 0,5 м и западный ветер 25 м/с: у Петербурга +{surge:.2f} м "
          f"(наводнения 1824 — 4,21 м, 1924 — 3,80 м, 1955 — 2,93 м)")
    print(f"  зона a/h > 0,2: {nw} ячеек, в Невской губе {sw:.0%} — ветер поднимает уровень "
          f"и на других мелководьях залива")
    print(f"Время расчёта 150 × 50 на 12 ч: {elapsed:.1f} с (нужно не больше 10 с)")
    assert ch < 0.01, "канал"
    assert abs(rise / theory - 1) < 0.1, "ветровой нагон"
    assert rest < 1e-6, "покой"
    assert 1.5 < gain < 8, "усиление волны"
    assert n2 > 0 and s2 > 0.8 and s3 > 0.8, "зона нелинейности"
    assert 1.5 < surge < 5.5, "нагон"
    assert elapsed < 10, "время"
    print("Все проверки пройдены")
