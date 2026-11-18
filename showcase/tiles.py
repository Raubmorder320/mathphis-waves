"""Тайлы карты через локальный кэш: показ не зависит от медленного интернета.

Сервер отдаёт /tiles/<слой>/<z>/<x>/<y>: из showcase/.tiles, а если тайла нет — скачивает
его один раз и сохраняет. Заранее прогреть кэш на весь Финский залив:

    python -m showcase.tiles             # ~1000 тайлов, при медленной сети — минут десять

Кэш в git не попадает (.gitignore): тайлы принадлежат их поставщикам, см. ATTRIBUTION.
"""
import asyncio
import math
import os
import sys
from pathlib import Path

import httpx

# Кэш: встроенный (в приложении — только чтение) и пишущий (SHOWCASE_CACHE, в приложении —
# папка пользователя). Без приложения это одна и та же папка showcase/.tiles.
BUNDLED = Path(__file__).resolve().parent / ".tiles"
CACHE = Path(os.environ.get("SHOWCASE_CACHE") or BUNDLED)
SOURCES = {
    # Спутниковый снимок и подписи — Esri; рельеф (с глубинами моря) — Mapzen/AWS Terrain Tiles.
    "sat": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "jpg"),
    "labels": ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}", "png"),
    "dem": ("https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png", "png"),
}
ATTRIBUTION = ("Снимок и подписи: Esri, Maxar, Earthstar Geographics · "
               "Рельеф: Mapzen, AWS Terrain Tiles (SRTM, GEBCO, ETOPO1) · Расчёт: mathphis-waves")
MEDIA = {"jpg": "image/jpeg", "png": "image/png"}
GULF = (21.5, 58.9, 30.9, 61.0)          # весь Финский залив: lon0, lat0, lon1, lat1
EAST = (26.8, 59.3, 30.6, 60.75)         # расчётная область
NEVA = (29.4, 59.75, 30.6, 60.15)        # Невская губа
PLAN = {                                 # слой: [(bbox, z0, z1)]
    "dem": [(GULF, 5, 8), (EAST, 9, 9)],
    "sat": [(GULF, 5, 8), (EAST, 9, 10)],
    "labels": [(GULF, 5, 8), (EAST, 9, 10)],
}

_client: httpx.AsyncClient | None = None
_locks: dict[str, asyncio.Lock] = {}


def path_for(src, z, x, y):
    return CACHE / src / str(z) / str(x) / f"{y}.{SOURCES[src][1]}"


async def get_tile(src, z, x, y, timeout=45.0):
    """Байты тайла и тип, или None, если поставщик недоступен."""
    global _client
    p = path_for(src, z, x, y)
    b = BUNDLED / src / str(z) / str(x) / f"{y}.{SOURCES[src][1]}"
    for q in (p, b):
        if q.exists():
            return q.read_bytes(), MEDIA[SOURCES[src][1]]
    lock = _locks.setdefault((str(p), id(asyncio.get_running_loop())), asyncio.Lock())
    async with lock:
        if p.exists():
            return p.read_bytes(), MEDIA[SOURCES[src][1]]
        loop = asyncio.get_running_loop()
        if _client is None or getattr(_client, "_loop", None) is not loop:
            # Клиент привязан к циклу событий: новый asyncio.run — новый клиент.
            _client = httpx.AsyncClient(timeout=timeout, follow_redirects=True,
                                        headers={"User-Agent": "mathphis-waves-showcase"})
            _client._loop = loop
        try:
            r = await _client.get(SOURCES[src][0].format(z=z, x=x, y=y))
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
        return r.content, MEDIA[SOURCES[src][1]]


def tile_range(bbox, z):
    lon0, lat0, lon1, lat1 = bbox
    n = 2 ** z
    tx = lambda lon: int((lon + 180) / 360 * n)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return [(z, x, y) for x in range(tx(lon0), tx(lon1) + 1) for y in range(ty(lat1), ty(lat0) + 1)]


async def prefetch(concurrency=24):
    jobs = sorted({(src, *t) for src, parts in PLAN.items() for bbox, z0, z1 in parts
                   for z in range(z0, z1 + 1) for t in tile_range(bbox, z)})
    todo = [j for j in jobs if not path_for(*j).exists()]
    print(f"Тайлов всего {len(jobs)}, качать {len(todo)}")
    sem = asyncio.Semaphore(concurrency)
    done = failed = 0

    async def one(job):
        nonlocal done, failed
        async with sem:
            ok = await get_tile(*job)
        done += 1
        failed += ok is None
        if done % 50 == 0 or done == len(todo):
            print(f"  {done}/{len(todo)}, не скачалось {failed}", flush=True)

    await asyncio.gather(*(one(j) for j in todo))
    print("Кэш готов" if not failed else f"Не скачалось {failed} — запустите ещё раз")


if __name__ == "__main__":
    asyncio.run(prefetch(int(sys.argv[1]) if len(sys.argv) > 1 else 24))
