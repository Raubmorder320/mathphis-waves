"""Детальная батиметрия для подводного режима и берега без ступенек.

    python -m showcase.seabed

Источник — EMODnet Bathymetry (DTM, шаг 1/16′ ≈ 58 × 115 м, вместе с отметками суши)
через WCS, кэш в showcase/.tiles/emodnet. Результат в showcase/data/seabed/:
- gulf.bin / gulf.json — весь залив, шаг ≈ 220 м, Int16 (дм), для подводного режима;
- neva.bin / neva.json — Невская губа и Кронштадт, бикубически до 25 м, Int16 (см);
- neva_sdf.bin — знаковое расстояние до берега, Int16 (дм): > 0 — вода, < 0 — суша.
И заново showcase/data/sea.png (маска моря и глубины основной карты) из EMODnet:
R — 1 + 25·√h, G — расстояние до берега (128 + d/40 м), чтобы берег рисовался гладко.
"""
import json
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".tiles" / "emodnet"
OUT = HERE / "data" / "seabed"
WCS = ("https://ows.emodnet-bathymetry.eu/wcs?service=WCS&version=2.0.1&request=GetCoverage"
       "&coverageId=emodnet__mean&subset=Lat({s},{n})&subset=Long({w},{e})&format=image/tiff")
GULF = (21.5, 58.9, 30.6, 61.0)          # lon0, lat0, lon1, lat1
NEVA = (29.45, 59.84, 30.35, 60.10)
FINE = 25.0                              # м — шаг детальной сетки губы


def fetch(w, s, e, n):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{w:.2f}_{s:.2f}_{e:.2f}_{n:.2f}.tif"
    if not p.exists():
        for attempt in range(4):
            try:
                r = httpx.get(WCS.format(w=w, s=s, e=e, n=n), timeout=180)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                    p.write_bytes(r.content)
                    break
            except httpx.HTTPError:
                pass
            time.sleep(3)
        else:
            raise RuntimeError(f"EMODnet недоступен: {w},{s},{e},{n}")
    return np.array(Image.open(p), dtype=np.float32)    # строки сверху вниз (север сверху)


def mosaic(box, step=1.5):
    """Отметки (м, вверх положительно) на прямоугольнике box кусками по step градусов."""
    lon0, lat0, lon1, lat1 = box
    rows = []
    lat = lat1
    while lat > lat0 + 1e-9:
        s = max(lat0, lat - step)
        row, lon = [], lon0
        while lon < lon1 - 1e-9:
            e = min(lon1, lon + step)
            row.append(fetch(round(lon, 3), round(s, 3), round(e, 3), round(lat, 3)))
            lon = e
        h = min(a.shape[0] for a in row)
        rows.append(np.hstack([a[:h] for a in row]))
        lat = s
    w = min(r.shape[1] for r in rows)
    return np.vstack([r[:, :w] for r in rows])


def sdf_m(water, cell_x, cell_y):
    """Знаковое расстояние до берега, м: вода > 0, суша < 0."""
    d_in = distance_transform_edt(water, sampling=(cell_y, cell_x))
    d_out = distance_transform_edt(~water, sampling=(cell_y, cell_x))
    return d_in - d_out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kx = lambda lat: 111320 * np.cos(np.radians(lat))

    # Весь залив.
    g = mosaic(GULF)
    print(f"EMODnet, весь залив: {g.shape[1]} × {g.shape[0]}, отметки {g.min():.0f}…{g.max():.0f} м")
    lon0, lat0, lon1, lat1 = GULF
    g4 = g[::2, ::2]                                       # ≈ 115 × 230 м → для подводного режима
    np.round(g4[::-1] * 10).astype("<i2").tofile(OUT / "gulf.bin")   # строки с юга
    (OUT / "gulf.json").write_text(json.dumps(dict(bbox=[lon0, lat0, lon1, lat1], nx=g4.shape[1], ny=g4.shape[0],
                                                   unit="дм, отметка вверх", source="EMODnet Bathymetry DTM")),
                                   encoding="utf-8")

    # sea.png для основной карты: шаг ≈ 165 м, гладкий берег по расстоянию.
    target_w, target_h = 3000, 1400
    sea = zoom(g, (target_h / g.shape[0], target_w / g.shape[1]), order=3)
    water = sea < 0
    cy = (lat1 - lat0) * 110540 / target_h
    cx = (lon1 - lon0) * kx(60.0) / target_w
    dist = gaussian_filter(sdf_m(water, cx, cy), 0.8)
    depth = np.maximum(-sea, 0)
    R = np.where(dist > 0, 1 + np.minimum(254, 25 * np.sqrt(depth)), 0)
    G = np.clip(128 + dist / 40.0, 0, 255)
    rgba = np.dstack([R, G, np.zeros_like(R), np.full_like(R, 255)]).round().astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(HERE / "data" / "sea.png", optimize=True)
    (HERE / "data" / "sea.json").write_text(json.dumps(dict(
        lon0=lon0, lon1=lon1, lat0=lat0, lat1=lat1, width=target_w, height=target_h,
        encoding="R: 0 — суша, иначе 1 + 25·√h; G: 128 + расстояние до берега / 40 м",
        source="EMODnet Bathymetry DTM")), encoding="utf-8")
    print(f"sea.png {target_w} × {target_h}: море {water.mean():.0%}")

    # Невская губа: бикубически до 25 м и знаковое расстояние до берега.
    n0 = mosaic(NEVA, step=1.0)
    w, s, e, n = NEVA[0], NEVA[1], NEVA[2], NEVA[3]
    nx, ny = int((e - w) * kx(0.5 * (s + n)) / FINE), int((n - s) * 110540 / FINE)
    fine = zoom(n0, (ny / n0.shape[0], nx / n0.shape[1]), order=3)[::-1]   # строки с юга
    fwater = fine < 0
    fsdf = gaussian_filter(sdf_m(fwater, FINE, FINE), 1.0)
    np.round(fine * 100).astype("<i2").tofile(OUT / "neva.bin")
    np.round(np.clip(fsdf * 10, -32000, 32000)).astype("<i2").tofile(OUT / "neva_sdf.bin")
    (OUT / "neva.json").write_text(json.dumps(dict(bbox=list(NEVA), nx=nx, ny=ny, res=FINE,
        unit="см, отметка вверх; sdf — дм", source="EMODnet Bathymetry DTM, бикубически")), encoding="utf-8")
    deep = -fine[fwater]
    print(f"Невская губа: {nx} × {ny} по {FINE:g} м, глубины до {deep.max():.1f} м, медиана {np.median(deep):.1f} м")


if __name__ == "__main__":
    main()
