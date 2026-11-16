"""Нелинейные уравнения мелкой воды (Сен-Венан), 1D (docs/theory.md, раздел 2.2).

    H_t + (H·u)_x = 0,    (H·u)_t + (H·u² + g·H²/2)_x = −g·H·z_x

H = h + η — полная глубина, z = −h — отметка дна (на суше z > 0).

Схема — метод конечных объёмов для (H, H·u): поток HLL, реконструкция MUSCL
с κ = 1/3 и ограничителем Корена (третий порядок на гладком решении, без
осцилляций на боре), гидростатическая реконструкция
Audusse et al. (2004), чтобы покоящаяся вода над любым дном, в том числе с сухим
берегом, оставалась в покое точно. По времени — Рунге — Кутта 2-го порядка (SSP),
шаг из условия Куранта. Ячейка с H < H_DRY считается сухой.

Проверки из TASKS.md запускаются командой:  python -m backend.models.saint_venant
"""
import numpy as np

from backend.common import G, setup, u_simple_wave
from backend.contract import Params, Result, WaveModel

CFL = 0.5      # число Куранта
H_DRY = 1e-6   # порог сухой ячейки, м
H_EPS = 1e-3   # ниже этой глубины скорость считается со сглаживанием, м


# --- Постановка --------------------------------------------------------------

def _setup(p: Params):
    """Вернуть x, h(x), η₀(x), u₀(x) и признак периодических границ (backend/common.py).

    Начальная скорость — простая волна, бегущая вправо: инвариант Римана u − 2√(gH)
    тот же, что в покое. Функцию использует и shallow_2d.
    """
    s = setup(p)
    return s.x, s.h, s.eta0, u_simple_wave(s.h, s.eta0), s.periodic


# --- Расчётное ядро: от контракта не зависит --------------------------------

def _minmod(a, b, c):
    """Наименьшее по модулю из трёх, если знаки совпадают, иначе 0."""
    return (np.maximum(np.minimum(np.minimum(a, b), c), 0.0)
            + np.minimum(np.maximum(np.maximum(a, b), c), 0.0))


def _faces(v):
    """Приращения к восточной и западной граням ячеек 1..N−2 по значениям v в ячейках.

    MUSCL κ = 1/3 с ограничителем Корена. Схема второго порядка (κ = 0) гасит волну
    заметно сильнее: около 1% амплитуды за 10 длин волны при 100 ячейках на длину,
    и пакет, бегущий к берегу, не выдерживает закон Грина.
    """
    dm, dp = np.diff(v)[..., :-1], np.diff(v)[..., 1:]  # разности с западным и восточным соседом
    return _koren(dm, dp)


def _koren(dm, dp):
    """Приращения к восточной и западной граням по разностям с западным и восточным соседом."""
    return (0.5 * _minmod(2 * dm, (dm + 2 * dp) / 3, 2 * dp),
            -0.5 * _minmod(2 * dp, (2 * dm + dp) / 3, 2 * dm))


def _velocity(H, q):
    """u = q/H без деления на почти нулевую глубину (Kurganov, Petrova 2007)."""
    H2 = H * H
    return np.where(H > H_DRY, 2.0 * H * q / (H2 + np.maximum(H2, H_EPS ** 2)), 0.0)


def _hll(hl, ul, hr, ur):
    """Поток HLL для (H, H·u) через грань с состояниями слева и справа."""
    cl, cr = np.sqrt(G * hl), np.sqrt(G * hr)
    sl = np.minimum(np.minimum(ul - cl, ur - cr), 0.0)
    sr = np.maximum(np.maximum(ul + cl, ur + cr), 0.0)
    ql, qr = hl * ul, hr * ur
    fl, fr = ql * ul + 0.5 * G * hl ** 2, qr * ur + 0.5 * G * hr ** 2
    den = sr - sl
    ok = den > 0                      # den = 0 только между двумя сухими ячейками
    den = np.where(ok, den, 1.0)
    f_h = np.where(ok, (sr * ql - sl * qr + sl * sr * (hr - hl)) / den, 0.0)
    f_q = np.where(ok, (sr * fl - sl * fr + sl * sr * (qr - ql)) / den, 0.0)
    return f_h, f_q


def solve_sv(z, H0, q0, dx, t_frames, periodic):
    """Решить уравнения Сен-Венана и вернуть η = H + z в моменты t_frames.

    z — отметка дна, H0 и q0 = H·u — начальные глубина и расход в ячейках с шагом dx.
    periodic=True — периодические границы, False — стенки (u = 0) на краях.
    На суше η равна отметке грунта. Возвращает массив (число кадров, число ячеек).
    """
    z = np.asarray(z, dtype=float)
    H = np.maximum(np.asarray(H0, dtype=float), 0.0)
    q = np.asarray(q0, dtype=float).copy()

    def ext(a, odd=False):
        """Две фиктивные ячейки с каждой стороны: периодика или зеркало у стенки."""
        if periodic:
            return np.pad(a, 2, mode="wrap")
        e = np.pad(a, 2, mode="symmetric")
        if odd:
            e[:2] *= -1.0
            e[-2:] *= -1.0
        return e

    ze = ext(z)

    def rhs(H, q):
        He, ue = ext(H), ext(_velocity(H, q), odd=True)
        we = He + ze                                   # уровень воды
        # Значения на восточной (E) и западной (W) гранях ячеек 1..n+2 расширенного
        # массива. У кромки воды реконструкции нет, чтобы не создать воду на суше.
        wet = He > H_DRY
        edge = ~(wet[:-2] & wet[1:-1] & wet[2:])
        (HE, HW), (wE, wW), (uE, uW) = (
            tuple(np.where(edge, 0.0, d) + v[1:-1] for d in _faces(v)) for v in (He, we, ue))
        zE, zW = wE - HE, wW - HW

        # Грань k разделяет ячейки k−1 и k исходной сетки (k = 0..n).
        hl, hr = HE[:-1], HW[1:]
        zs = np.maximum(zE[:-1], zW[1:])
        hls = np.maximum(0.0, wE[:-1] - zs)
        hrs = np.maximum(0.0, wW[1:] - zs)
        f_h, f_q = _hll(hls, uE[:-1], hrs, uW[1:])

        # Гидростатические поправки потока импульса и источник внутри ячейки.
        f_q_west = f_q[:-1] + 0.5 * G * (hr[:-1] ** 2 - hrs[:-1] ** 2)
        f_q_east = f_q[1:] + 0.5 * G * (hl[1:] ** 2 - hls[1:] ** 2)
        i = slice(1, -1)
        source = -0.5 * G * (HE[i] + HW[i]) * (zE[i] - zW[i])
        return -(f_h[1:] - f_h[:-1]) / dx, (source - (f_q_east - f_q_west)) / dx

    def clean(H, q):
        """Глубина не отрицательна, в сухих ячейках вода стоит."""
        H = np.maximum(H, 0.0)
        return H, H * _velocity(H, q)

    frames = np.empty((len(t_frames), len(H)))
    frames[0] = H + z
    t = float(t_frames[0])
    for j in range(1, len(t_frames)):
        # Шаги подгоняются так, чтобы попасть точно на момент кадра.
        while t < t_frames[j] - 1e-12:
            speed = np.max(np.abs(_velocity(H, q)) + np.sqrt(G * H))
            dt = min(CFL * dx / max(speed, 1e-12), t_frames[j] - t)
            dH, dq = rhs(H, q)
            H1, q1 = clean(H + dt * dH, q + dt * dq)
            dH, dq = rhs(H1, q1)
            H, q = clean(0.5 * (H + H1 + dt * dH), 0.5 * (q + q1 + dt * dq))
            t += dt
            if not (np.all(np.isfinite(H)) and np.all(np.isfinite(q))):
                raise RuntimeError(f"Расчёт Сен-Венана разошёлся при t = {t:.2f} с")
        frames[j] = H + z
    return frames


class SaintVenantModel(WaveModel):
    id = "saint_venant"
    name = "Сен-Венан 1D"
    description = (
        "Нелинейная мелкая вода: H_t + (H·u)_x = 0, (H·u)_t + (H·u² + g·H²/2)_x = g·H·h_x. "
        "Гребень бежит быстрее подошвы, фронт укручается до вертикального и превращается "
        "в бор, синусоида становится пилой. Дисперсии нет. Есть уклон дна и накат на сухой берег."
    )
    dim = 1
    extra_params: list = []

    def solve(self, p: Params) -> Result:
        x, h, eta0, u0, periodic = _setup(p)
        t = np.linspace(0.0, p.t_end, p.n_frames)
        H0 = np.maximum(h + eta0, 0.0)
        eta = solve_sv(-h, H0, H0 * u0, x[1] - x[0], t, periodic)
        return Result(x=x.tolist(), t=t.tolist(), eta=eta.tolist(), h=h.tolist())


# --- Проверки «готово, когда» из TASKS.md -----------------------------------

def _solve(p):
    r = SaintVenantModel().solve(p)
    return np.array(r.x), np.array(r.t), np.array(r.eta), np.array(r.h)


def _check_speed() -> tuple[float, float]:
    """a = 0,01 м: скорость первой гармоники и её амплитуда после пробега области."""
    p = Params(a=0.01, wavelength=100.0, depth=5.0, L=1000.0, nx=1000, n_frames=2)
    c0 = np.sqrt(G * p.depth)
    p.t_end = p.L / c0                                # волна обходит область ровно раз
    _, t, eta, _ = _solve(p)
    m = round(p.L / p.wavelength)                     # номер гармоники λ = 100 м
    f0, f1 = np.fft.rfft(eta[0])[m], np.fft.rfft(eta[-1])[m]
    # После полного оборота фаза должна вернуться; остаток фазы — ошибка пути.
    lag = np.angle(f1 / f0) / (2 * np.pi * m / p.L)   # отставание, м
    return float(lag / p.L), float(abs(f1) / abs(f0) - 1.0)


def _check_mass() -> float:
    """Ровное дно, значения по умолчанию: относительное изменение массы за 60 с."""
    p = Params(a=0.5, wavelength=100.0, depth=5.0, L=1000.0, t_end=60.0, nx=1000, n_frames=2)
    _, _, eta, h = _solve(p)
    mass = (eta + h).sum(axis=1)
    return float(abs(mass[-1] - mass[0]) / mass[0])


def _check_breaking() -> tuple[float, float, float]:
    """Значения по умолчанию: время вертикального фронта против t* и форма пилы.

    До обрушения наибольший наклон простой волны растёт как s₀/(1 − t/t_b), то есть
    1/max|η_x| убывает по времени линейно. Прямую строим по участку, где фронт ещё
    хорошо разрешён сеткой (наклон меньше трёх начальных), и продолжаем до нуля:
    это момент, когда фронт становится вертикальным. От шага сетки не зависит.
    """
    p = Params(a=0.5, wavelength=100.0, depth=5.0, L=1000.0, t_end=30.0, nx=2000, n_frames=301)
    _, t, eta, _ = _solve(p)
    inv = 1.0 / (np.abs(np.diff(eta, axis=1)).max(axis=1) / (p.L / p.nx))
    fit = inv > inv[0] / 3
    slope, icept = np.polyfit(t[fit], inv[fit], 1)
    t_break = float(-icept / slope)
    t_star = p.wavelength / (3 * np.pi * (p.a / p.depth) * np.sqrt(G * p.depth))
    # Пила: волна бежит вправо, фронт — спад η, тыл — пологий подъём.
    d = np.diff(eta[-1])
    return t_break, float(t_star), float(-d.min() / d.max())


def _check_rest() -> float:
    """a = 0 над склоном с сухим берегом: наибольшее |η| за 60 с в мокрых ячейках."""
    p = Params(a=0.0, wavelength=100.0, depth=5.0, L=1000.0, t_end=60.0, nx=500, n_frames=7)
    p.bottom, p.depth_shore = "slope", -1.0
    _, _, eta, h = _solve(p)
    return float(np.abs(eta[:, h > 0]).max())


def _check_green() -> tuple[float, float]:
    """Склон до обрушения: амплитуда пакета растёт как h^(−1/4), как у модели linear.

    Амплитуда 1 мм: при a = 1 см волна за 600 с пути заметно укручивается у берега
    (там t* ≈ 170 с), и это уже не участок «до обрушения». Сетка нужна мельче, чем
    у linear (4000 ячеек против 1600): схема с потоком HLL слегка гасит волну, у
    берега λ ≈ 55 м — около 70 ячеек на длину волны.
    """
    p = Params(a=0.001, wavelength=100.0, depth=5.0, L=4000.0, t_end=680.0, nx=4000, n_frames=1361)
    p.bottom, p.depth_shore = "slope", 1.0
    x, _, eta, h = _solve(p)
    amp = np.abs(eta).max(axis=0)
    zone = (x > 400.0) & (x < 3500.0)
    ref = np.argmax(zone)
    green = (h[ref] / h[zone]) ** 0.25
    return float(green[-1]), float(np.max(np.abs(amp[zone] / amp[ref] / green - 1.0)))


def _check_runup() -> tuple[float, float, float, float]:
    """Пакет на сухом берегу: урез воды до, в максимуме наката и в конце; наименьшая H."""
    p = Params(a=0.3, wavelength=100.0, depth=5.0, L=1000.0, t_end=300.0, nx=1000, n_frames=301)
    p.bottom, p.depth_shore = "slope", -2.0
    x, _, eta, h = _solve(p)
    H = eta + h
    shore = np.array([x[np.flatnonzero(row > 1e-2)].max() for row in H])
    return float(shore[0]), float(shore.max()), float(shore[-1]), float(H.min())


if __name__ == "__main__":
    dc, da = _check_speed()
    print(f"a = 0,01 м: путь за оборот отличается от √(gh)·t на {dc:+.2%}, амплитуда на {da:+.2%} (допуск 1%)")
    dm = _check_mass()
    print(f"Масса на ровном дне за 60 с: относительное изменение {dm:.1e} (нужно меньше 1e-8)")
    tb, ts, saw = _check_breaking()
    print(f"Вертикальный фронт при t = {tb:.1f} с, по теории t* = {ts:.1f} с "
          f"(отличие {tb / ts - 1:+.0%}, допуск 20%); при t = 30 с фронт круче тыла в {saw:.0f} раз")
    rest = _check_rest()
    print(f"Покой над склоном с сухим берегом: наибольшее |η| = {rest:.1e} м (нужно меньше 1 мм)")
    growth, dev = _check_green()
    print(f"Склон, закон Грина: рост по теории в {growth:.2f} раза, "
          f"наибольшее отклонение расчёта {dev:.1%} (допуск 10%)")
    s0, smax, s_end, hmin = _check_runup()
    print(f"Накат: урез x = {s0:.0f} м → {smax:.0f} м → {s_end:.0f} м, наименьшая глубина {hmin:.1e} м")
    assert abs(dc) < 0.01 and abs(da) < 0.01, "скорость или форма малой волны"
    assert dm < 1e-8, "масса"
    assert abs(tb / ts - 1) < 0.2 and saw > 5, "обрушение"
    assert rest < 1e-3, "покой над склоном"
    assert dev < 0.10, "закон Грина"
    assert smax > s0 + 5 and s_end < smax - 5 and hmin >= 0, "накат"
    print("Все проверки пройдены")
