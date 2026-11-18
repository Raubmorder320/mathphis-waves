"""Спутниковый снимок зоны города — склейка тайлов z15 без пересэмплирования.

    python -m showcase.city_sat vo kronstadt

Тайлы Esri World Imagery z15 (Web Mercator, ~2,4 м на пиксель на 60° с. ш.) идут через
кэш карты (showcase/.tiles, в git не попадает) и склеиваются как есть:
showcase/.tiles/city/<зона>.jpg и <зона>.json — индексы крайних тайлов и точные
границы склейки (tile2lon / tile2lat), а не bbox запроса. UV земли в браузере
считаются той же формулой тайлов, поэтому снимок совпадает с OSM пиксель в пиксель.
Контрольная проверка: углы домов OSM → пиксель склейки двумя путями.
"""
import asyncio
import gzip
import io
import json
import math
import sys
from pathlib import Path

from PIL import Image

from showcase.osm_builder import AREAS
from showcase.tiles import CACHE, get_tile, tile_range

HERE = Path(__file__).resolve().parent
Z = 15
OUT = CACHE / "city"


def lon2tile(lon, z):
    return (lon + 180.0) / 360.0 * 2 ** z


def lat2tile(lat, z):
    return (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * 2 ** z


def tile2lon(x, z):
    return x / 2 ** z * 360.0 - 180.0


def tile2lat(y, z):
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / 2 ** z))))


async def fetch(tiles):
    sem = asyncio.Semaphore(10)

    async def one(t):
        async with sem:
            return t, await get_tile("sat", *t)
    return await asyncio.gather(*(one(t) for t in tiles))


def mercator_uv(lon, lat, meta):
    """Та же формула, что в city_worker.js: UV склейки по координатам тайлов Web Mercator."""
    u = (lon2tile(lon, meta["z"]) - meta["x0"]) / (meta["x1"] - meta["x0"])
    v = (lat2tile(lat, meta["z"]) - meta["y0"]) / (meta["y1"] - meta["y0"])   # 0 — верх (север)
    return u, v


def main(ids):
    OUT.mkdir(parents=True, exist_ok=True)
    for aid in ids:
        name, s, w, n, e = AREAS[aid]
        tiles = tile_range((w, s, e, n), Z)
        got = asyncio.run(fetch(tiles))
        xs, ys = sorted({t[1] for t in tiles}), sorted({t[2] for t in tiles})
        mos = Image.new("RGB", (len(xs) * 256, len(ys) * 256), (60, 70, 60))
        for (z, x, y), g in got:
            if g:
                mos.paste(Image.open(io.BytesIO(g[0])).convert("RGB"), ((x - xs[0]) * 256, (y - ys[0]) * 256))
        mos.save(OUT / f"{aid}.jpg", quality=88)
        # Границы склейки — по краям крайних тайлов, не по bbox запроса.
        meta = dict(z=Z, x0=xs[0], y0=ys[0], x1=xs[-1] + 1, y1=ys[-1] + 1,
                    width=mos.width, height=mos.height,
                    lon0=tile2lon(xs[0], Z), lon1=tile2lon(xs[-1] + 1, Z),
                    lat_top=tile2lat(ys[0], Z), lat_bottom=tile2lat(ys[-1] + 1, Z))
        (OUT / f"{aid}.json").write_text(json.dumps(meta), encoding="utf-8")
        ok = sum(g is not None for _, g in got)
        print(f"{name}: тайлов z{Z} {ok}/{len(tiles)}, склейка {mos.width} × {mos.height}, "
              f"{meta['lon0']:.5f}…{meta['lon1']:.5f} в. д., {meta['lat_bottom']:.5f}…{meta['lat_top']:.5f} с. ш.")

        # Контрольные точки: углы домов OSM → пиксель (а) по UV сцены, (б) по формуле Slippy Map.
        osm = json.loads(gzip.decompress((HERE / "data" / "osm" / f"{aid}.json.gz").read_bytes()))
        worst = 0.0
        step = max(1, len(osm["buildings"]) // 7)
        for b in osm["buildings"][::step][:7]:
            x, y = b["p"][0]
            lon, lat = osm["origin"][0] + x / osm["kx"], osm["origin"][1] + y / osm["ky"]
            u, v = mercator_uv(lon, lat, meta)
            pa = (u * meta["width"], v * meta["height"])
            pb = ((lon2tile(lon, Z) - xs[0]) * 256, (lat2tile(lat, Z) - ys[0]) * 256)
            worst = max(worst, math.hypot(pa[0] - pb[0], pa[1] - pb[1]))
        print(f"  контроль: 7 углов домов, расхождение пикселя UV и Slippy Map — {worst:.2e} пикс.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["vo", "kronstadt"])
