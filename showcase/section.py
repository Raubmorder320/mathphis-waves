"""Разрез волн через любую точку залива: путь волны до точки и три модели на нём.

Путь — луч наименьшего времени (принцип Ферма): кратчайший по времени пробега
√(g·h) путь по воде от западной открытой границы карты до точки, дальше — прямо,
пока не упрётся в берег. Вдоль пути глубина h(s) берётся из карты Влада, и на этом
дне считаются те же ядра, что у моделей проекта: linear, saint_venant, boussinesq.

    python -m showcase.section              # проверка на нескольких точках
"""
import heapq
import time

import numpy as np

from backend.common import G, N_PACKET, u_linear, u_simple_wave
from backend.models.boussinesq import GAMMA, H_WALL, solve_boussinesq
from backend.models.kdv import solve_kdv
from backend.models.linear import solve_linear
from backend.models.saint_venant import solve_sv
from backend.models.shallow_2d import load_gulf

NX = 1000            # узлов на разрезе
OUT_NODES = 400      # узлов в ответе
OUT_FRAMES = 160     # кадров в ответе
H_MIN = 1.0          # м — меньшую глубину внутри пути не допускаем (сглаживание у берега)
MODELS = ("linear", "saint_venant", "boussinesq", "kdv")


class Gulf:
    """Карта глубин и перевод координат карты (м) ↔ долгота/широта."""

    def __init__(self):
        self.x, self.y, self.h, self.lon, self.lat = load_gulf()
        self.wet = self.h > 0

    def to_lonlat(self, px, py):
        return np.interp(px, self.x, self.lon), np.interp(py, self.y, self.lat)

    def to_xy(self, lon, lat):
        return np.interp(lon, self.lon, self.x), np.interp(lat, self.lat, self.y)

    def depth(self, px, py):
        """Билинейная интерполяция глубины в точках (px, py); суша — 0."""
        fx = np.clip(np.interp(px, self.x, np.arange(len(self.x))), 0, len(self.x) - 1.001)
        fy = np.clip(np.interp(py, self.y, np.arange(len(self.y))), 0, len(self.y) - 1.001)
        i, j = fy.astype(int), fx.astype(int)
        a, b = fy - i, fx - j
        h = self.h
        return ((1 - a) * (1 - b) * h[i, j] + (1 - a) * b * h[i, j + 1]
                + a * (1 - b) * h[i + 1, j] + a * b * h[i + 1, j + 1])

    def cell(self, lon, lat):
        px, py = self.to_xy(lon, lat)
        return int(np.argmin(np.abs(self.y - py))), int(np.argmin(np.abs(self.x - px)))


def fastest_path(gulf: Gulf, target):
    """Дейкстра по ячейкам воды (8 соседей), вес — время пробега ds/√(g·h)."""
    ny, nx = gulf.h.shape
    dx, dy = gulf.x[1] - gulf.x[0], gulf.y[1] - gulf.y[0]
    slow = 1.0 / np.sqrt(G * np.maximum(gulf.h, 1e-3))
    best = np.full((ny, nx), np.inf)
    prev = -np.ones((ny, nx, 2), dtype=int)
    heap = []
    for iy in np.flatnonzero(gulf.wet[:, 0]):
        best[iy, 0] = 0.0
        heap.append((0.0, int(iy), 0))
    heapq.heapify(heap)
    steps = [(di, dj, np.hypot(di * dy, dj * dx)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    while heap:
        t, i, j = heapq.heappop(heap)
        if (i, j) == target:
            break
        if t > best[i, j]:
            continue
        for di, dj, ds in steps:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and gulf.wet[a, b]:
                nt = t + ds * 0.5 * (slow[i, j] + slow[a, b])
                if nt < best[a, b]:
                    best[a, b] = nt
                    prev[a, b] = (i, j)
                    heapq.heappush(heap, (nt, a, b))
    if not np.isfinite(best[target]):
        raise ValueError("До этой точки нет пути по воде от западной границы")
    path = [target]
    while path[-1][1] != 0 or best[path[-1]] > 0:
        p = tuple(prev[path[-1]])
        if p[0] < 0:
            break
        path.append(p)
    path.reverse()
    return np.array([(gulf.x[j], gulf.y[i]) for i, j in path], dtype=float)


def _smooth(poly, rounds=3):
    """Сглаживание ломаной по Чайкину; концы на месте."""
    for _ in range(rounds):
        q = 0.75 * poly[:-1] + 0.25 * poly[1:]
        r = 0.25 * poly[:-1] + 0.75 * poly[1:]
        poly = np.vstack([poly[:1], np.column_stack([q, r]).reshape(-1, 2), poly[-1:]])
    return poly


def _resample(poly, n):
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    si = np.linspace(0.0, s[-1], n)
    return si, np.column_stack([np.interp(si, s, poly[:, 0]), np.interp(si, s, poly[:, 1])])


def build_path(gulf: Gulf, lon, lat):
    """Путь волны: от западной границы до точки и дальше по прямой до берега."""
    target = gulf.cell(lon, lat)
    px, py = gulf.to_xy(lon, lat)
    if not gulf.wet[target]:
        # Карта грубая (ячейка 1,3 × 2,7 км): у берега берём ближайшую воду в пределах 3 ячеек.
        iy, ix = np.nonzero(gulf.wet)
        d = np.hypot(iy - target[0], ix - target[1])
        if d.min() > 3:
            raise ValueError("Точка на суше — щёлкните по воде")
        target = (int(iy[d.argmin()]), int(ix[d.argmin()]))
        px, py = gulf.x[target[1]], gulf.y[target[0]]
    poly = fastest_path(gulf, target)
    poly[-1] = (px, py)                       # точно в точку щелчка
    # Продолжение: по направлению последнего участка, шагами по 200 м, до берега.
    tail = poly[-1] - poly[max(0, len(poly) - 4)]
    if np.hypot(*tail) > 0:
        d = tail / np.hypot(*tail)
        ext, p = [], poly[-1].copy()
        for _ in range(2000):
            p = p + 200.0 * d
            inside = gulf.x[0] <= p[0] <= gulf.x[-1] and gulf.y[0] <= p[1] <= gulf.y[-1]
            if not inside or gulf.depth(p[:1], p[1:])[0] <= 0.2:
                break
            ext.append(p.copy())
        poly = np.vstack([poly, ext]) if ext else poly
    smooth = _smooth(poly)
    s, pts = _resample(smooth, NX)
    h = gulf.depth(pts[:, 0], pts[:, 1])
    h[:-1] = np.maximum(h[:-1], H_MIN)
    h[-1] = 0.0                               # берег в конце пути
    i_click = int(np.argmin(np.hypot(pts[:, 0] - px, pts[:, 1] - py)))
    return s, pts, h, i_click


def _run_sv(h, eta0, dx, t):
    H0 = np.maximum(h + eta0, 0.0)
    return solve_sv(-h, H0, H0 * u_simple_wave(h, eta0), dx, t, False)


def _run_boussinesq(h, eta0, dx, t, a, wavelength):
    """Буссинеск: берег — стенка там, где глубина меньше амплитуды (как в BoussinesqModel).

    Наката модель не описывает: если волна на мелком пути осушает дно, стенка
    переносится глубже (2a, 4a). Не помогло — Буссинеск на этом пути не считается.
    """
    eta = np.full((len(t), len(h)), np.nan)
    for depth in (max(H_WALL, a), 2 * a, 4 * a):
        shallow = np.flatnonzero(h < depth)
        wet = int(shallow[0]) if len(shallow) else len(h)
        if wet < 10:
            break
        try:
            eta[:, :wet] = solve_boussinesq(h[:wet], eta0[:wet], u_simple_wave(h[:wet], eta0[:wet]),
                                            dx, t, False, window=wavelength, gamma=GAMMA)
            return eta, wet
        except RuntimeError:
            continue
    return eta, 1


def _run_kdv(s, eta0, t, h_eff):
    """КдФ: дно не учитывается — ровное дно h_eff (волна приходит в точку в то же время).

    Область периодическая, поэтому берём её вдвое длиннее пути: волна не успевает
    обернуться. Результат интерполируется на узлы пути.
    """
    # Солитоны КдФ узкие (~140 м при Ur ~ 5000): шаг 94 м, иначе ответ зависит от сетки.
    n, L = 4096, 2.0 * s[-1]
    x = np.arange(n) * L / n
    dense = np.linspace(t[0], t[-1], 4 * (len(t) - 1) + 1)     # пик в точке: солитон быстрее кадров
    frames = solve_kdv(np.interp(x, s, eta0, right=0.0), h_eff, L, dense)
    out = np.array([np.interp(s, x, row) for row in frames])
    return out[::4], out.max(axis=0)


def solve_section(lon, lat, a=0.5, wavelength=10e3, pool=None):
    """Посчитать три модели вдоль пути волны через точку (lon, lat).

    pool — concurrent.futures.Executor: Сен-Венан и Буссинеск считаются в нём
    параллельно (сервер держит два процесса), иначе по очереди.
    """
    gulf = Gulf()
    s, pts, h, i_click = build_path(gulf, lon, lat)
    dx = s[1] - s[0]
    if N_PACKET * wavelength > 0.6 * s[-1]:
        raise ValueError("Путь короче пакета волны: возьмите длину волны меньше")

    # Пакет из N_PACKET периодов у западного края, как в backend/common.py для gulf.
    k = 2 * np.pi / wavelength
    width = N_PACKET * wavelength
    eta0 = np.where(s < width, a * np.sin(np.pi * s / width) ** 2 * np.sin(k * s), 0.0)

    # Время: пока пакет целиком пройдёт точку щелчка, но до прихода отражения от берега.
    c = np.sqrt(G * np.maximum(h, H_MIN))
    travel = np.r_[0.0, np.cumsum(dx / (0.5 * (c[1:] + c[:-1])))]
    t_end = min(travel[i_click] + width / c[i_click] * 1.2 + width / c[0],
                2 * travel[-1] - travel[i_click] - 0.5 * width / c[-1])
    t_end = max(t_end, travel[i_click] + 0.25 * width / c[i_click])
    t = np.linspace(0.0, t_end, OUT_FRAMES)

    start = time.perf_counter()
    # КдФ на ровном дне: глубина, при которой время пробега до точки то же, что над реальным дном.
    h_eff = float((s[i_click] / max(travel[i_click], 1e-6)) ** 2 / G) if i_click > 0 else float(h[0])
    if pool is not None:
        f_sv = pool.submit(_run_sv, h, eta0, dx, t)
        f_b = pool.submit(_run_boussinesq, h, eta0, dx, t, a, wavelength)
        f_k = pool.submit(_run_kdv, s, eta0, t, h_eff)
    eta = {"linear": solve_linear(h, eta0, u_linear(h, eta0), dx, t, False)}
    if pool is not None:
        eta["saint_venant"], (eta["boussinesq"], wet) = f_sv.result(), f_b.result()
        eta["kdv"], kdv_peak = f_k.result()
    else:
        eta["saint_venant"] = _run_sv(h, eta0, dx, t)
        eta["boussinesq"], wet = _run_boussinesq(h, eta0, dx, t, a, wavelength)
        eta["kdv"], kdv_peak = _run_kdv(s, eta0, t, h_eff)
    elapsed = time.perf_counter() - start

    # Крутизна фронта: наибольший |η_x| за время в каждом узле, относительно линейной.
    # Сравниваем только там, где волна уже пришла (линейный наклон не меньше 5% от
    # начального a·k), иначе отношение двух почти нулей даёт шум.
    steep = {m: np.abs(np.diff(eta[m], axis=1)).max(axis=0) / dx for m in ("linear", "saint_venant")}
    lin = steep["linear"]
    ratio = np.where(lin > 0.05 * a * k, steep["saint_venant"] / np.maximum(lin, 1e-12), 1.0)
    ratio = np.r_[ratio, 1.0]
    over = np.flatnonzero(ratio > 2.0)
    i_steep = int(over[0]) if len(over) else None

    at = {m: float(np.nanmax(e[:, i_click])) if np.isfinite(e[:, i_click]).any() else 0.0
          for m, e in eta.items()}
    at["kdv"] = float(kdv_peak[i_click])
    lon_p, lat_p = gulf.to_lonlat(pts[:, 0], pts[:, 1])
    pick = np.linspace(0, len(s) - 1, OUT_NODES).round().astype(int)
    pick = np.unique(np.r_[pick, i_click])
    rnd = lambda v: np.round(np.nan_to_num(v, nan=0.0), 4).tolist()
    return dict(
        lon=np.round(lon_p[pick], 5).tolist(), lat=np.round(lat_p[pick], 5).tolist(),
        s=np.round(s[pick], 1).tolist(), h=np.round(h[pick], 2).tolist(),
        t=np.round(t, 1).tolist(), click=int(np.searchsorted(pick, i_click)),
        eta={m: [rnd(row[pick]) for row in e] for m, e in eta.items()},
        wet_boussinesq=float(s[wet - 1]),
        summary=dict(
            a=a, wavelength=wavelength, depth=float(h[i_click]), distance=float(s[i_click]),
            to_shore=float(s[-1] - s[i_click]), arrival=float(travel[i_click]),
            max_eta=at, a_over_h={m: v / max(float(h[i_click]), 1e-6) for m, v in at.items()},
            steep_at_click=float(ratio[i_click]), kdv_depth=h_eff,
            steep2_distance=None if i_steep is None else float(s[i_steep]),
            elapsed=elapsed,
        ),
    )


if __name__ == "__main__":
    points = {
        "Невская губа": (30.1, 59.95), "Кронштадт": (29.68, 60.0), "середина": (28.5, 60.0),
        "у входа": (27.3, 60.0), "Выборгский залив": (28.75, 60.45), "суша": (30.4, 60.0),
    }
    for name, (lo, la) in points.items():
        try:
            r = solve_section(lo, la)
        except ValueError as e:
            print(f"{name}: {e}")
            continue
        sm = r["summary"]
        e = {m: np.array(v) for m, v in r["eta"].items()}
        bad = any(not np.all(np.isfinite(v)) for v in e.values())
        print(f"{name}: путь {sm['distance'] / 1e3:.0f} км, глубина {sm['depth']:.1f} м, "
              f"η_max лин/СВ/Бус = {sm['max_eta']['linear']:.2f}/{sm['max_eta']['saint_venant']:.2f}/"
              f"{sm['max_eta']['boussinesq']:.2f} м, фронт СВ ×{sm['steep_at_click']:.1f}, "
              f"×2 с {(sm['steep2_distance'] or 0) / 1e3:.0f} км, {sm['elapsed']:.1f} с, NaN: {bad}")
