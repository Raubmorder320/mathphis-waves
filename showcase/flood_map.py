"""Карта высот и препятствий для затопления улиц (шаг 4 м): рельеф + здания + набережные.

    python -m showcase.flood_map vo kronstadt

Рельеф — тайлы Terrarium z14 (для Петербурга это SRTM/ГЕБКО, шаг ~30 м, сглажено),
поверх — растр зданий OSM (непроницаемые стены) и парапетов набережных. Вода —
по рельефу ниже порога и по водным полигонам, которые получаются «вычитанием»:
всё, что ниже WATER_Z и не здание и не дорога. Результат — showcase/data/flood/<зона>.bin
(Int16, см: отметка дна или земли; здания — код BUILDING) и <зона>.json.
Расчёт затопления идёт на видеокарте в браузере (static/flood.js).
"""
import asyncio
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from backend.models.shallow_2d import gauge, load_gulf
from showcase.osm_builder import AREAS
from showcase.tiles import get_tile, path_for, tile_range

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "flood"
RES = 4.0              # м на ячейку
Z = 14
WATER_Z = 0.6          # м — ниже этого по рельефу считаем водой (Нева, залив)
BED = -4.0             # м — дно рек и залива в пределах зоны (Нева 4–12 м; для затопления важна суша)
PARAPET = 1.0          # м — гранитный парапет набережной над землёй
BUILDING = 32000       # код ячейки-здания
GAUGE = {"vo": (30.26, 59.93), "kronstadt": (29.76, 59.99)}   # откуда брать уровень моря из 2D-расчёта


async def fetch_dem(bbox):
    s, w, n, e = bbox
    tiles = tile_range((w, s, e, n), Z)
    sem = asyncio.Semaphore(12)

    async def one(t):
        async with sem:
            return await get_tile("dem", *t)

    got = await asyncio.gather(*(one(t) for t in tiles))
    print(f"  рельеф z{Z}: {sum(g is not None for g in got)}/{len(tiles)} тайлов")
    return tiles


def dem_grid(bbox, tiles, nx, ny):
    s, w, n, e = bbox
    xs, ys = sorted({t[1] for t in tiles}), sorted({t[2] for t in tiles})
    img = np.full((len(ys) * 256, len(xs) * 256), np.nan)
    for _, x, y in tiles:
        p = path_for("dem", Z, x, y)
        if p.exists():
            rgb = np.asarray(Image.open(p).convert("RGB"), dtype=float)
            img[ys.index(y) * 256:(ys.index(y) + 1) * 256, xs.index(x) * 256:(xs.index(x) + 1) * 256] = \
                rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
    nt = 2 ** Z
    lon = w + (np.arange(nx) + 0.5) / nx * (e - w)
    lat = s + (np.arange(ny) + 0.5) / ny * (n - s)
    px = ((lon + 180) / 360 * nt - xs[0]) * 256 - 0.5
    py = ((1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * nt - ys[0]) * 256 - 0.5
    # Билинейно по пикселям мозаики.
    x0 = np.clip(np.floor(px).astype(int), 0, img.shape[1] - 2); fx = np.clip(px - x0, 0, 1)
    y0 = np.clip(np.floor(py).astype(int), 0, img.shape[0] - 2); fy = np.clip(py - y0, 0, 1)
    a = img[y0[:, None], x0[None, :]]; b = img[y0[:, None], x0[None, :] + 1]
    c = img[y0[:, None] + 1, x0[None, :]]; d = img[y0[:, None] + 1, x0[None, :] + 1]
    fxg, fyg = fx[None, :], fy[:, None]
    return (a * (1 - fxg) + b * fxg) * (1 - fyg) + (c * (1 - fxg) + d * fxg) * fyg


def main(ids):
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, h2, lon2, lat2 = load_gulf()
    for aid in ids:
        name, s, w, n, e = AREAS[aid]
        osm = json.loads(gzip.decompress((HERE / "data" / "osm" / f"{aid}.json.gz").read_bytes()))
        sx, sy = osm["size"]
        nx, ny = int(sx / RES), int(sy / RES)
        print(f"{name}: сетка {nx} × {ny} по {RES:g} м")
        tiles = asyncio.run(fetch_dem((s, w, n, e)))
        ground = dem_grid((s, w, n, e), tiles, nx, ny)
        ground = np.nan_to_num(ground, nan=0.0)

        # Растр в координатах зоны: x на восток, y на север; картинка — сверху север.
        def canvas():
            return Image.new("L", (nx, ny), 0)
        to_px = lambda pts: [(x / RES, ny - 1 - y / RES) for x, y in pts]
        road = canvas(); dr = ImageDraw.Draw(road)
        for r in osm["roads"]:
            if r["tu"]:
                continue
            dr.line(to_px(r["p"]), fill=255, width=max(1, int(round((r["w"] + 2 * r["sw"]) / RES))))
        bld = canvas(); db = ImageDraw.Draw(bld)
        for b in osm["buildings"]:
            if b["b"] < 2.5:                                   # консоли и арки над землёй не перегораживают
                db.polygon(to_px(b["p"]), fill=255)
        quay = canvas(); dq = ImageDraw.Draw(quay)
        for q in osm["quays"]:
            dq.line(to_px(q["p"]), fill=255, width=2)
        road, bld, quay = (np.asarray(im)[::-1] > 0 for im in (road, bld, quay))   # строки — с юга

        from scipy.ndimage import binary_dilation, gaussian_filter, label, minimum_filter
        dsm = ground                                          # SRTM в городе — вместе с крышами
        wfile = HERE / ".osm" / f"{aid}_water.json"
        if wfile.exists():
            # Контуры воды OSM (реки, каналы, береговая линия) — границы; связные области
            # между ними с низкой отметкой и без домов — вода.
            edge = canvas(); de = ImageDraw.Draw(edge)
            for el in json.loads(wfile.read_text(encoding="utf-8"))["elements"]:
                geoms = [el["geometry"]] if el.get("geometry") else [m["geometry"] for m in el.get("members", []) if m.get("geometry")]
                for g in geoms:
                    pts = [((p["lon"] - w) * osm["kx"] / RES, ny - 1 - (p["lat"] - s) * osm["ky"] / RES) for p in g if p]
                    if len(pts) > 1:
                        de.line(pts, fill=255, width=1)
            edge = np.asarray(edge)[::-1] > 0
            lab, nlab = label(~edge & ~bld)
            water = np.zeros_like(edge)
            for k in range(1, nlab + 1):
                cells = lab == k
                if cells.sum() > 200 and np.median(dsm[cells]) < 4.0 and road[cells].mean() < 0.05:
                    water |= cells
            water |= edge & binary_dilation(water, iterations=1) & ~bld
            how = "контуры воды OSM"
        else:
            water = (dsm < WATER_Z) & ~bld & ~road & ~quay
            lab, _ = label(water)
            edgel = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
            water = np.isin(lab, list(edgel))
            how = "порог по рельефу"
        # Земля без домов: минимум по окну 250 м (улицы и дворы) и сглаживание — грубый DTM из DSM.
        land_dsm = np.where(water | bld, np.nan, dsm)
        fill = np.where(np.isnan(land_dsm), np.nanpercentile(land_dsm, 20), land_dsm)
        dtm = gaussian_filter(minimum_filter(fill, size=int(250 / RES)), sigma=60 / RES)
        ground = np.maximum(dtm, 1.0)
        print(f"  вода: {how}; земля: минимум DSM по окну 250 м, сглажено")
        bed = np.where(water, BED, ground)
        bed = np.where(quay & ~water, bed + PARAPET, bed)
        code = np.round(bed * 100).astype(np.int16)
        code[bld] = BUILDING
        code.tofile(OUT / f"{aid}.bin")

        g = gauge(lon2, lat2, h2, GAUGE[aid])
        land = ~water & ~bld
        meta = dict(id=aid, name=name, bbox=[s, w, n, e], nx=nx, ny=ny, res=RES, origin=osm["origin"],
                    kx=osm["kx"], ky=osm["ky"], building=BUILDING, bed=BED, parapet=PARAPET,
                    gauge=dict(iy=int(g[0]), ix=int(g[1])), water_source=how,
                    ground_note="земля — оценка из SRTM DSM фильтром минимума по окну 250 м, точность ±2–3 м",
                    ground_p10=float(np.percentile(bed[land], 10)), ground_p50=float(np.median(bed[land])))
        (OUT / f"{aid}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        print(f"  вода {water.mean():.0%}, здания {bld.mean():.0%}, дороги {road.mean():.0%}; "
              f"суша: 10% ниже {meta['ground_p10']:.1f} м, медиана {meta['ground_p50']:.1f} м")
        Image.fromarray(np.where(bld, 255, np.where(water, 40, np.clip(bed * 25 + 80, 60, 230))).astype(np.uint8)[::-1]) \
            .save(OUT / f"{aid}_preview.png")


if __name__ == "__main__":
    main(sys.argv[1:] or ["vo", "kronstadt"])
