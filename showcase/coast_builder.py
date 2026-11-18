"""Береговое кольцо Невской губы: здания, дороги и мосты OSM в полосе 1 км вглубь суши.

    python -m showcase.coast_builder            # закачка (в 3 потока, с кэшем) и сборка
    python -m showcase.coast_builder --cached   # только сборка из уже скачанного

Квадраты запросов — только там, где по EMODnet (showcase/data/seabed/neva_sdf.bin)
есть суша не дальше BAND м от берега, плюс дамба КЗС. Результат в формате
osm_builder: showcase/data/osm/coast.json.gz. Здания внутри детальных зон
(Васильевский остров, Кронштадт) не берутся — их рисуют сами зоны.
Земля в кольце — плоская (снимок карты), основания домов — на уровне карты.
"""
import gzip
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from showcase.osm_builder import AREAS, PARTS, RAW, _post, build_height, simplify, ROAD_WIDTH, SIDEWALK, _num

HERE = Path(__file__).resolve().parent
BAND = 1000.0                     # м вглубь суши
DLAT, DLON = 0.02, 0.04           # квадрат запроса ≈ 2,2 × 2,2 км
SKIP = [AREAS["vo"][1:], AREAS["kronstadt"][1:]]


def squares():
    meta = json.loads((HERE / "data/seabed/neva.json").read_text(encoding="utf-8"))
    sdf = np.fromfile(HERE / "data/seabed/neva_sdf.bin", dtype="<i2").reshape(meta["ny"], meta["nx"]) / 10.0
    w, s, e, n = meta["bbox"]
    band = (sdf <= 0) & (sdf > -BAND)
    jj, ii = np.nonzero(band)
    lat = s + (jj + 0.5) / meta["ny"] * (n - s)
    lon = w + (ii + 0.5) / meta["nx"] * (e - w)
    keys = set(zip(np.floor((lat - s) / DLAT).astype(int), np.floor((lon - w) / DLON).astype(int)))
    out = []
    for a, b in sorted(keys):
        la, lo = s + a * DLAT, w + b * DLON
        out.append((round(la, 4), round(lo, 4), round(la + DLAT, 4), round(lo + DLON, 4)))
    return out, (s, w, n, e)


def fetch_square(job):
    part, (s, w, n, e) = job
    p = RAW / "coast" / f"{part}_{s:.4f}_{w:.4f}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    got = _post(f"[out:json][timeout:180];({PARTS[part].format(bb=f'{s},{w},{n},{e}')});out tags geom;")
    p.write_text(json.dumps(got), encoding="utf-8")
    return got


def inside_skip(lon, lat):
    return any(s < lat < n and w < lon < e for s, w, n, e in SKIP)


def main(cached=False):
    sq, (s0, w0, n0, e0) = squares()
    (RAW / "coast").mkdir(parents=True, exist_ok=True)
    jobs = [(part, q) for part in PARTS for q in sq]
    print(f"Кольцо берега: {len(sq)} квадратов, запросов {len(jobs)}")
    elements, done, start = {}, 0, time.perf_counter()
    if cached:
        for part, (s, w, n, e) in jobs:
            p = RAW / "coast" / f"{part}_{s:.4f}_{w:.4f}.json"
            if p.exists():
                for el in json.loads(p.read_text(encoding="utf-8")):
                    elements[(el["type"], el["id"])] = el
                done += 1
    else:
        with ThreadPoolExecutor(3) as pool:
            for got in pool.map(fetch_square, jobs):
                done += 1
                for el in got:
                    elements[(el["type"], el["id"])] = el
                if done % 5 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)}, элементов {len(elements)}, {time.perf_counter() - start:.0f} с", flush=True)

    kx = 111320.0 * math.cos(math.radians(0.5 * (s0 + n0)))
    ky = 110540.0
    xy = lambda g: [[round((p["lon"] - w0) * kx, 1), round((p["lat"] - s0) * ky, 1)] for p in g]
    buildings, roads, quays = [], [], []
    for el in elements.values():
        tags, g = el.get("tags", {}), el.get("geometry")
        if "building" in tags and el["type"] == "way" and g and len(g) >= 4:
            if inside_skip(g[0]["lon"], g[0]["lat"]):
                continue
            ring = simplify(xy(g), 0.7)[:-1]
            if len(ring) < 3:
                continue
            h, base = build_height(tags)
            buildings.append(dict(p=ring, h=round(h, 1), b=round(base, 1), k=tags.get("building", "yes"),
                                  c=tags.get("building:colour"), r=tags.get("roof:shape"), rc=tags.get("roof:colour"),
                                  hist=1 if tags.get("heritage") or tags.get("historic") else 0))
        elif "highway" in tags and g:
            kind = tags["highway"]
            base_kind = kind.replace("_link", "")
            if base_kind not in ROAD_WIDTH:
                continue
            if inside_skip(g[0]["lon"], g[0]["lat"]) and inside_skip(g[-1]["lon"], g[-1]["lat"]):
                continue
            lanes = _num(tags.get("lanes"))
            width = _num(tags.get("width")) or (lanes * 3.3 if lanes else ROAD_WIDTH[base_kind] * (0.7 if kind != base_kind else 1))
            roads.append(dict(p=simplify(xy(g), 1.0), w=round(width, 1), k=kind, sw=SIDEWALK.get(base_kind, 0),
                              br=1 if tags.get("bridge") and tags.get("bridge") != "no" else 0,
                              tu=1 if tags.get("tunnel") and tags.get("tunnel") != "no" else 0))
        elif g and (tags.get("man_made") or tags.get("barrier")):
            quays.append(dict(p=simplify(xy(g), 0.5), k=tags.get("man_made") or tags.get("barrier")))

    out = dict(id="coast", name="Берег Невской губы", bbox=[s0, w0, n0, e0], origin=[w0, s0], kx=kx, ky=ky,
               size=[round((e0 - w0) * kx, 1), round((n0 - s0) * ky, 1)], flat=True,
               buildings=buildings, roads=roads, quays=quays, squares=len(sq), fetched=done, total=len(jobs))
    p = HERE / "data" / "osm" / "coast.json.gz"
    p.write_bytes(gzip.compress(json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode()))
    print(f"  зданий {len(buildings)}, дорог {len(roads)} (мостов {sum(r['br'] for r in roads)}), "
          f"набережных {len(quays)} → {p.name} ({p.stat().st_size / 1e6:.1f} МБ), квадратов скачано {done}/{len(jobs)}")


if __name__ == "__main__":
    main("--cached" in sys.argv)
