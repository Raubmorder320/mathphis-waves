"""Заранее посчитанные двумерные сценарии для 3D-эмулятора.

    python -m showcase.build            # все сценарии → showcase/data/

Каждый сценарий — расчёт shallow_2d на карте залива (data/gulf_bathymetry.npz):
кадры η, u, v в формате Int16 с масштабом и описанием в showcase/data/manifest.json.
Здесь же подбор под наводнение 1975 года (+2,81 м у Горного института).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from backend.models.shallow_2d import SPB, gauge, load_gulf, solve_2d

OUT = Path(__file__).resolve().parent / "data"
HOUR = 3600.0
T_END = 15 * HOUR
N_FRAMES = 121          # кадры η, каждые 7,5 мин
UV_EVERY = 3            # течения — каждый третий кадр
WAVELENGTH = 300e3      # длинная нагонная волна, T ≈ 5 ч на входе
FLOOD_1975 = 2.81       # м, пост «Горный институт», 30.09.1975 (analysis/gulf.md)
WIND_1975 = 27.5        # м/с, середина 25–30 м/с из описания наводнения

PLACES = [
    ("Санкт-Петербург", *SPB),
    ("Кронштадт", 29.77, 59.99),
    ("Петергоф", 29.91, 59.88),
    ("Сестрорецк", 29.96, 60.09),
    ("Ломоносов", 29.77, 59.92),
    ("Выборгский залив", 28.6, 60.55),
    ("Лужская губа", 28.35, 59.75),
    ("о. Гогланд", 27.0, 60.05),
]
FLOODS = [("1824", 4.21), ("1924", 3.80), ("1955", 2.93), ("1975", FLOOD_1975)]


def run(a, wind, t_end=T_END, frames=N_FRAMES, uv=False, wavelength=WAVELENGTH, wind_dir=270.0):
    x, y, h, lon, lat = load_gulf()
    t = np.linspace(0.0, t_end, frames)
    extra = {"wind_speed": wind, "wind_dir": wind_dir}
    out = solve_2d(x, y, h, np.zeros_like(h), np.zeros_like(h), "gulf", t, a, wavelength, extra,
                   return_uv=uv)
    return (t, *out)


def spb_max(a, wind):
    """Наибольший уровень у Горного института за 12 ч (для подбора — короче и реже)."""
    x, y, h, lon, lat = load_gulf()
    _, _, eta_max = run(a, wind, 12 * HOUR, 25)
    return float(eta_max[gauge(lon, lat, h)])


def bisect(f, lo, hi, target, tol=0.01, steps=12):
    """Корень f(v) = target на [lo, hi]; f растёт по v."""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def pack(a):
    """Int16 с общим масштабом: значение = код · scale."""
    scale = float(np.abs(a).max()) / 32000 or 1.0
    return np.round(a / scale).astype("<i2"), scale


def main(only=None):
    """only — список id: пересчитать только их, остальное взять из прежнего манифеста."""
    OUT.mkdir(exist_ok=True)
    x, y, h, lon, lat = load_gulf()
    g = gauge(lon, lat, h)
    old = json.loads((OUT / "manifest.json").read_text(encoding="utf-8")) if only else None
    if old:
        fit = old["fit_1975"]
        wind_for_05, a_1975 = fit["wind_for_wave_05"], fit["wave_for_wind"]
    else:
        wind_for_05, a_1975 = fit_1975()
    scenarios = [
        # Видимая волна: 40 км (32 ячейки карты на длину), огибает острова, заходит в губы.
        ("front", "Волна 1 м длиной 40 км — фронт идёт по заливу", 1.0, 0.0, 40e3, 5 * HOUR),
        ("wave", "Длинная волна 0,3 м, без ветра", 0.3, 0.0, WAVELENGTH, T_END),
        ("wind15", "Волна 0,5 м + западный ветер 15 м/с", 0.5, 15.0, WAVELENGTH, T_END),
        ("wind20", "Волна 0,5 м + западный ветер 20 м/с", 0.5, 20.0, WAVELENGTH, T_END),
        ("wind25", "Волна 0,5 м + западный ветер 25 м/с", 0.5, 25.0, WAVELENGTH, T_END),
        ("wind30", "Волна 0,5 м + западный ветер 30 м/с", 0.5, 30.0, WAVELENGTH, T_END),
        ("flood1975", f"Наводнение 1975: волна {a_1975:.2f} м + ветер {WIND_1975:g} м/с", a_1975, WIND_1975,
         WAVELENGTH, T_END),
    ]
    meta = []
    for sid, title, a, wind, wavelength, t_end in scenarios:
        if only and sid not in only:
            meta.append(next(m for m in old["scenarios"] if m["id"] == sid))
            continue
        start = time.perf_counter()
        t, eta, eta_max, u, v = run(a, wind, t_end, uv=True, wavelength=wavelength)
        eta_c, eta_s = pack(eta)
        uv_c, uv_s = pack(np.stack((u[::UV_EVERY], v[::UV_EVERY])))
        eta_c.tofile(OUT / f"{sid}_eta.bin")
        uv_c.tofile(OUT / f"{sid}_uv.bin")
        level = eta[:, g[0], g[1]]
        meta.append(dict(
            id=sid, title=title, a=round(a, 3), wind=wind, wavelength=wavelength,
            t=np.round(t, 1).tolist(), frames=len(t),
            eta_scale=eta_s, uv_scale=uv_s, uv_every=UV_EVERY,
            spb_level=np.round(level, 4).tolist(), spb_max=round(float(eta_max[g]), 3),
            eta_max=np.round(eta_max, 3).tolist(),
        ))
        print(f"  {title}: у Петербурга +{eta_max[g]:.2f} м ({time.perf_counter() - start:.0f} с)")

    manifest = dict(
        format="showcase/2", nx=len(x), ny=len(y),
        x=x.tolist(), y=y.tolist(),
        lon=lon.tolist(), lat=lat.tolist(), h=np.round(h, 2).tolist(),
        gauge=dict(iy=int(g[0]), ix=int(g[1]), lon=float(lon[g[1]]), lat=float(lat[g[0]])),
        places=[dict(name=n, lon=lo, lat=la) for n, lo, la in PLACES],
        floods=[dict(year=yr, level=lv) for yr, lv in FLOODS],
        fit_1975=dict(wind_for_wave_05=round(wind_for_05, 1), wave_for_wind=round(a_1975, 3),
                      wind=WIND_1975, target=FLOOD_1975),
        scenarios=meta,
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"Готово: {OUT} ({size:.1f} МБ)")


def run_custom(sid, a, wavelength, wind, wind_dir, hours, out_dir):
    """Свой 2D-расчёт из интерфейса: тот же формат, что у готовых сценариев, файлы — в out_dir."""
    x, y, h, lon, lat = load_gulf()
    g = gauge(lon, lat, h)
    t, eta, eta_max, u, v = run(a, wind, hours * HOUR, uv=True, wavelength=wavelength, wind_dir=wind_dir)
    eta_c, eta_s = pack(eta)
    uv_c, uv_s = pack(np.stack((u[::UV_EVERY], v[::UV_EVERY])))
    out_dir.mkdir(parents=True, exist_ok=True)
    eta_c.tofile(out_dir / f"{sid}_eta.bin")
    uv_c.tofile(out_dir / f"{sid}_uv.bin")
    title = f"Свой расчёт: волна {a:g} м × {wavelength / 1e3:g} км, ветер {wind:g} м/с из {wind_dir:g}°"
    return dict(id=sid, title=title, a=a, wind=wind, wind_dir=wind_dir, wavelength=wavelength,
                t=np.round(t, 1).tolist(), frames=len(t), eta_scale=eta_s, uv_scale=uv_s, uv_every=UV_EVERY,
                spb_level=np.round(eta[:, g[0], g[1]], 4).tolist(), spb_max=round(float(eta_max[g]), 3),
                eta_max=np.round(eta_max, 3).tolist(), files=dict(eta=f"/custom/{sid}_eta.bin", uv=f"/custom/{sid}_uv.bin"))


def fit_1975():
    print("Подбор под наводнение 1975 года (+2,81 м у Горного института):")
    start = time.perf_counter()
    wind_for_05 = bisect(lambda w: spb_max(0.5, w), 0.0, 30.0, FLOOD_1975, tol=0.1)
    print(f"  волна 0,5 м: нужен западный ветер {wind_for_05:.1f} м/с")
    a_1975 = bisect(lambda a: spb_max(a, WIND_1975), 0.0, 1.0, FLOOD_1975, tol=0.005)
    print(f"  ветер {WIND_1975} м/с (как в 1975 году, 25–30 м/с): нужна волна на входе {a_1975:.2f} м"
          f"  ({time.perf_counter() - start:.0f} с)")
    return wind_for_05, a_1975


if __name__ == "__main__":
    main(sys.argv[1:] or None)
