"""Спутниковый снимок зоны города, выровненный по сетке зоны (для земли под домами).

    python -m showcase.city_sat vo kronstadt

Тайлы Esri World Imagery z15 (~2,4 м на пиксель на 60° с. ш.) идут через тот же кэш,
что и карта (showcase/.tiles, в git не попадает), склеиваются и пересэмпливаются
в метры зоны: showcase/.tiles/city/<зона>.jpg. Сервер отдаёт его по /citysat/<зона>.jpg.
"""
import asyncio
import gzip
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from showcase.osm_builder import AREAS
from showcase.tiles import CACHE, get_tile, tile_range

HERE = Path(__file__).resolve().parent
Z = 15
PX = 2.5                                  # м на пиксель итоговой картинки
OUT = CACHE / "city"


async def fetch(tiles):
    sem = asyncio.Semaphore(10)

    async def one(t):
        async with sem:
            return t, await get_tile("sat", *t)
    return await asyncio.gather(*(one(t) for t in tiles))


def main(ids):
    OUT.mkdir(parents=True, exist_ok=True)
    for aid in ids:
        name, s, w, n, e = AREAS[aid]
        tiles = tile_range((w, s, e, n), Z)
        got = asyncio.run(fetch(tiles))
        ok = sum(g is not None for _, g in got)
        print(f"{name}: тайлов z{Z} {ok}/{len(tiles)}")
        xs, ys = sorted({t[1] for t in tiles}), sorted({t[2] for t in tiles})
        mos = Image.new("RGB", (len(xs) * 256, len(ys) * 256), (60, 70, 60))
        for (z, x, y), g in got:
            if g:
                mos.paste(Image.open(io.BytesIO(g[0])).convert("RGB"), (xs.index(x) * 256, ys.index(y) * 256))
        mos = np.asarray(mos)
        osm = json.loads(gzip.decompress((HERE / "data" / "osm" / f"{aid}.json.gz").read_bytes()))
        sx, sy = osm["size"]
        W, H = int(sx / PX), int(sy / PX)
        lon = w + (np.arange(W) + 0.5) / W * (e - w)
        lat = n - (np.arange(H) + 0.5) / H * (n - s)            # сверху север
        nt = 2 ** Z
        px = ((lon + 180) / 360 * nt - xs[0]) * 256
        py = ((1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * nt - ys[0]) * 256
        img = mos[np.clip(py.astype(int), 0, mos.shape[0] - 1)[:, None], np.clip(px.astype(int), 0, mos.shape[1] - 1)[None, :]]
        Image.fromarray(img).save(OUT / f"{aid}.jpg", quality=86)
        print(f"  {W} × {H} по {PX:g} м → {OUT / (aid + '.jpg')}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["vo", "kronstadt"])
