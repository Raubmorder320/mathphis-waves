"""Подробная маска моря и глубины всего Финского залива для отрисовки воды.

Берётся из тайлов рельефа Terrarium (в них батиметрия GEBCO), которые лежат в кэше
showcase/.tiles (python -m showcase.tiles). Результат — showcase/data/sea.png:
серый 8 бит, 0 — суша, иначе 1 + 25·√h (h — глубина, м), и sea.json с границами.
Расчёт моделей эта карта не трогает: она только для картинки (берег, цвет по глубине).

    python -m showcase.sea
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from showcase.tiles import GULF, path_for, tile_range

OUT = Path(__file__).resolve().parent / "data"
Z = 8
STEP_LON, STEP_LAT = 0.006, 0.003          # ≈ 330 м на 60° с. ш.


def mosaic(z=Z):
    tiles = tile_range(GULF, z)
    xs, ys = sorted({t[1] for t in tiles}), sorted({t[2] for t in tiles})
    img = np.full((len(ys) * 256, len(xs) * 256), np.nan)
    for _, x, y in tiles:
        p = path_for("dem", z, x, y)
        if not p.exists():
            continue
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=float)
        elev = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
        i, j = ys.index(y), xs.index(x)
        img[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] = elev
    return img, xs[0], ys[0], z


def main():
    img, x0, y0, z = mosaic()
    n = 2 ** z
    lon0, lat0, lon1, lat1 = GULF
    lons = np.arange(lon0, lon1, STEP_LON)
    lats = np.arange(lat1, lat0, -STEP_LAT)               # сверху вниз, как в картинке
    # Пиксель мозаики для каждой точки сетки (Меркатор).
    px = ((lons + 180) / 360 * n - x0) * 256
    py = ((1 - np.arcsinh(np.tan(np.radians(lats))) / math.pi) / 2 * n - y0) * 256
    elev = img[np.clip(py.astype(int), 0, img.shape[0] - 1)[:, None],
               np.clip(px.astype(int), 0, img.shape[1] - 1)[None, :]]
    depth = np.where(np.isfinite(elev) & (elev < 0), -elev, 0.0)
    # Где тайла рельефа нет, берём карту глубин модели (data/gulf_bathymetry.npz).
    hole = ~np.isfinite(elev)
    if hole.any():
        from backend.models.shallow_2d import load_gulf
        _, _, h, mlon, mlat = load_gulf()
        LON, LAT = np.meshgrid(lons, lats)
        ix = np.clip(np.round(np.interp(LON, mlon, np.arange(len(mlon)))).astype(int), 0, len(mlon) - 1)
        iy = np.clip(np.round(np.interp(LAT, mlat, np.arange(len(mlat)))).astype(int), 0, len(mlat) - 1)
        inside = (LON >= mlon[0]) & (LON <= mlon[-1]) & (LAT >= mlat[0]) & (LAT <= mlat[-1])
        depth = np.where(hole & inside, h[iy, ix], depth)
    # Лёгкое размытие: берег и мелководье без «ступенек» пикселей 330 м.
    from scipy.ndimage import gaussian_filter
    soft = gaussian_filter(np.where(depth > 0, 1 + 25 * np.sqrt(depth), 0.0), 1.0)
    code = np.clip(np.round(soft), 0, 255).astype(np.uint8)
    OUT.mkdir(exist_ok=True)
    Image.fromarray(code, mode="L").save(OUT / "sea.png", optimize=True)
    meta = dict(lon0=float(lons[0]), lon1=float(lons[-1]), lat0=float(lats[-1]), lat1=float(lats[0]),
                width=len(lons), height=len(lats), encoding="0 — суша, иначе 1 + 25·√h")
    (OUT / "sea.json").write_text(json.dumps(meta), encoding="utf-8")
    missing = int(np.isnan(elev).sum())
    print(f"sea.png {len(lons)}×{len(lats)}, море {np.mean(depth > 0):.0%}, "
          f"наибольшая глубина {depth.max():.0f} м, нет данных в {missing} точках")


if __name__ == "__main__":
    main()
