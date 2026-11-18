"""Здания, улицы и набережные из OpenStreetMap для детального режима города.

    python -m showcase.osm_builder vo kronstadt      # зоны по id, без аргументов — vo и kronstadt
    python -m showcase.osm_builder --all             # все зоны из AREAS

Overpass API → кэш showcase/.osm/<зона>.json (сырой ответ) → showcase/data/osm/<зона>.json.gz:
координаты в метрах от угла зоны (запад-юг), с точностью 0,1 м.
"""
import gzip
import json
import math
import re
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
RAW = HERE / ".osm"
OUT = HERE / "data" / "osm"
MIRRORS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",   # быстрее всего отсюда
    "https://overpass-api.de/api/interpreter",
]
# Зона: (название, юг, запад, север, восток)
AREAS = {
    "vo": ("Васильевский остров", 59.921, 30.205, 59.957, 30.300),
    "kronstadt": ("Кронштадт", 59.982, 29.700, 60.012, 29.800),
    "lomonosov": ("Ломоносов", 59.903, 29.745, 59.922, 29.800),
    "peterhof": ("Петергоф", 59.872, 29.885, 59.893, 29.930),
    "petrogradka": ("Петроградская сторона", 59.952, 30.270, 59.975, 30.330),
    "admiralteysky": ("Адмиралтейский район", 59.906, 30.275, 59.940, 30.330),
    "kzs": ("Дамба КЗС", 59.905, 29.660, 60.090, 29.960),
}
ROAD_WIDTH = {   # м, полная ширина проезжей части
    "motorway": 22, "trunk": 18, "primary": 16, "secondary": 13, "tertiary": 10,
    "residential": 7, "unclassified": 6, "living_street": 5, "service": 4,
    "pedestrian": 6, "footway": 2.5, "path": 2, "cycleway": 2, "steps": 2.5,
    "track": 3.5, "corridor": 2, "bridleway": 2, "platform": 4, "raceway": 8, "busway": 7, "road": 6,
}
SIDEWALK = {"primary": 4, "secondary": 3.5, "tertiary": 3, "residential": 2.5, "unclassified": 2, "trunk": 4}
DEFAULT_HEIGHT = {   # м, если в OSM нет ни height, ни building:levels
    "house": 7, "detached": 7, "garage": 3, "garages": 3, "shed": 3, "kiosk": 3, "roof": 4,
    "industrial": 10, "warehouse": 9, "church": 22, "cathedral": 35, "apartments": 19,
    "residential": 19, "commercial": 16, "retail": 9, "office": 18, "school": 14, "university": 18,
}


CHUNK = 0.012          # ° — квадрат одного запроса (~0,7 × 1,3 км): большие Overpass не тянет
PARTS = {
    "buildings": 'way["building"]({bb}); relation["building"]["type"="multipolygon"]({bb});',
    "lines": 'way["highway"]({bb}); way["man_made"~"^(quay|embankment|pier|breakwater|groyne)$"]({bb});'
             ' way["barrier"="retaining_wall"]({bb});',
}


def _post(q):
    for url in MIRRORS:
        for attempt in range(3):
            try:
                r = httpx.post(url, data={"data": q}, timeout=300,
                               headers={"User-Agent": "mathphis-waves-showcase/1.0 (student project)"})
            except httpx.HTTPError:
                continue
            if r.status_code == 200:
                return r.json()["elements"]
            time.sleep(3 + 5 * attempt)                     # 429/504 — подождать и повторить
    raise RuntimeError("Overpass недоступен")


def fetch(area_id, force=False, cached_only=False):
    """Все элементы зоны: запросы по квадратам CHUNK, ответы кэшируются по одному."""
    _, s, w, n, e = AREAS[area_id]
    cache = RAW / area_id
    cache.mkdir(parents=True, exist_ok=True)
    lats = [s + i * CHUNK for i in range(math.ceil((n - s) / CHUNK) + 1)]
    lons = [w + i * CHUNK * 2 for i in range(math.ceil((e - w) / (CHUNK * 2)) + 1)]
    jobs = [(part, la, lo) for part in PARTS for la in lats[:-1] for lo in lons[:-1]]
    elements, start = {}, time.perf_counter()
    for k, (part, la, lo) in enumerate(jobs):
        p = cache / f"{part}_{la:.4f}_{lo:.4f}.json"
        if p.exists() and not force:
            got = json.loads(p.read_text(encoding="utf-8"))
        elif cached_only:
            continue
        else:
            bb = f"{la:.5f},{lo:.5f},{min(la + CHUNK, n):.5f},{min(lo + 2 * CHUNK, e):.5f}"
            got = _post(f"[out:json][timeout:180];({PARTS[part].format(bb=bb)});out tags geom;")
            p.write_text(json.dumps(got), encoding="utf-8")
        for el in got:
            elements[(el["type"], el["id"])] = el
        print(f"  {k + 1}/{len(jobs)} квадратов, элементов {len(elements)}, {time.perf_counter() - start:.0f} с",
              flush=True)
    return {"elements": list(elements.values())}


def simplify(pts, tol):
    """Дуглас — Пеккер: убрать точки, отклоняющиеся от хорды меньше tol (м)."""
    if len(pts) < 3:
        return pts
    if pts[0] == pts[-1]:
        # Замкнутый контур: хорда вырождена — делим по самой дальней от начала точке.
        k = max(range(len(pts)), key=lambda m: math.hypot(pts[m][0] - pts[0][0], pts[m][1] - pts[0][1]))
        if 0 < k < len(pts) - 1:
            return simplify(pts[:k + 1], tol)[:-1] + simplify(pts[k:], tol)
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        (x1, y1), (x2, y2) = pts[i], pts[j]
        L = math.hypot(x2 - x1, y2 - y1) or 1e-9
        best, k = 0.0, -1
        for m in range(i + 1, j):
            d = abs((x2 - x1) * (y1 - pts[m][1]) - (x1 - pts[m][0]) * (y2 - y1)) / L
            if d > best:
                best, k = d, m
        if best > tol:
            keep[k] = True
            stack += [(i, k), (k, j)]
    return [p for p, f in zip(pts, keep) if f]


def _num(v):
    m = re.match(r"\s*([\d.]+)", str(v or ""))
    return float(m.group(1)) if m else None


def build_height(tags):
    """Высота над землёй и высота основания (м) по тегам OSM."""
    h = _num(tags.get("height"))
    levels = _num(tags.get("building:levels"))
    if h is None and levels is not None:
        h = levels * 3.2 + (1.5 if tags.get("roof:shape") not in (None, "flat") else 0.6)
    if h is None:
        h = DEFAULT_HEIGHT.get(tags.get("building"), 15.0)
    base = _num(tags.get("min_height"))
    if base is None and _num(tags.get("building:min_level")) is not None:
        base = _num(tags.get("building:min_level")) * 3.2
    return min(max(h, 2.5), 160.0), base or 0.0


def main(ids, cached_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    for aid in ids:
        name, s, w, n, e = AREAS[aid]
        print(f"{name} ({aid}):")
        data = fetch(aid, cached_only=cached_only)
        lat0, lon0 = s, w
        kx = 111320.0 * math.cos(math.radians(0.5 * (s + n)))
        ky = 110540.0
        xy = lambda g: [[round((p["lon"] - lon0) * kx, 1), round((p["lat"] - lat0) * ky, 1)] for p in g]

        buildings, roads, quays = [], [], []
        for el in data["elements"]:
            tags = el.get("tags", {})
            if "building" in tags or tags.get("building:part"):
                rings = []
                if el["type"] == "way" and el.get("geometry"):
                    rings = [el["geometry"]]
                elif el["type"] == "relation":
                    rings = [m["geometry"] for m in el.get("members", [])
                             if m.get("role") == "outer" and m.get("geometry") and
                             m["geometry"][0] == m["geometry"][-1]]
                hgt, base = build_height(tags)
                for g in rings:
                    if len(g) < 4:
                        continue
                    ring = simplify(xy(g), 0.7)[:-1]           # замкнутый контур без повтора
                    if len(ring) < 3:
                        continue
                    buildings.append(dict(p=ring, h=round(hgt, 1), b=round(base, 1),
                                          k=tags.get("building", "yes"), c=tags.get("building:colour"),
                                          r=tags.get("roof:shape"), rc=tags.get("roof:colour"),
                                          hist=1 if tags.get("heritage") or tags.get("historic") else 0))
            elif "highway" in tags and el.get("geometry"):
                kind = tags["highway"]
                base_kind = kind.replace("_link", "")               # съезды — как основная дорога, уже
                if base_kind not in ROAD_WIDTH:
                    continue
                lanes = _num(tags.get("lanes"))
                width = _num(tags.get("width")) or (lanes * 3.3 if lanes else
                                                    ROAD_WIDTH[base_kind] * (0.7 if kind != base_kind else 1.0))
                roads.append(dict(p=simplify(xy(el["geometry"]), 1.0), w=round(width, 1), k=kind,
                                  sw=SIDEWALK.get(base_kind, 0),
                                  br=1 if tags.get("bridge") and tags.get("bridge") != "no" else 0,
                                  tu=1 if tags.get("tunnel") and tags.get("tunnel") != "no" else 0))
            elif el.get("geometry") and (tags.get("man_made") or tags.get("barrier")):
                quays.append(dict(p=simplify(xy(el["geometry"]), 0.5), k=tags.get("man_made") or tags.get("barrier")))

        out = dict(id=aid, name=name, bbox=[s, w, n, e], origin=[lon0, lat0], kx=kx, ky=ky,
                   size=[round((e - w) * kx, 1), round((n - s) * ky, 1)],
                   buildings=buildings, roads=roads, quays=quays)
        p = OUT / f"{aid}.json.gz"
        p.write_bytes(gzip.compress(json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode()))
        hs = sorted(b["h"] for b in buildings)
        print(f"  зданий {len(buildings)} (высота медиана {hs[len(hs) // 2] if hs else 0:.0f} м), "
              f"дорог {len(roads)}, набережных и стенок {len(quays)} → {p.name} ({p.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    args = sys.argv[1:]
    cached = "--cached" in args                       # собрать только из уже скачанных квадратов
    args = [a for a in args if a != "--cached"]
    main(list(AREAS) if args == ["--all"] else args or ["vo", "kronstadt"], cached)
