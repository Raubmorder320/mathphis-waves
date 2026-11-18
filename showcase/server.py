"""Сервер 3D-эмулятора: страница, заранее посчитанные сценарии и разрез по клику.

    uvicorn showcase.server:app --port 8800      # затем http://127.0.0.1:8800

Отдельное приложение, чтобы не трогать backend/main.py: модели вызываются напрямую.
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from showcase.section import solve_section
from showcase.tiles import ATTRIBUTION, SOURCES, get_tile

HERE = Path(__file__).resolve().parent
pool: ProcessPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app):
    # Два процесса держим заранее: Сен-Венан и Буссинеск разреза считаются параллельно.
    global pool
    pool = ProcessPoolExecutor(max_workers=2)
    yield
    pool.shutdown(cancel_futures=True)


app = FastAPI(title="Финский залив в 3D", lifespan=lifespan)


@app.middleware("http")
async def no_stale_code(request, call_next):
    """Страница и скрипты — без кэша браузера: после правок всегда свежий код."""
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp
app.mount("/data", StaticFiles(directory=HERE / "data"), name="data")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/tiles/{src}/{z}/{x}/{y}")
async def tile(src: str, z: int, x: int, y: int):
    """Тайл карты из локального кэша (при промахе — скачать и сохранить)."""
    if src not in SOURCES:
        raise HTTPException(404)
    got = await get_tile(src, z, x, y)
    if got is None:
        raise HTTPException(404, "Тайла нет в кэше и поставщик недоступен")
    return Response(got[0], media_type=got[1], headers={"Cache-Control": "max-age=86400"})


@app.get("/osm/{area}.json")
def osm(area: str):
    """Здания и улицы зоны (showcase/osm_builder.py): отдаём сжатый файл как есть, gzip."""
    p = HERE / "data" / "osm" / f"{area}.json.gz"
    if not p.exists():
        raise HTTPException(404, "Нет данных OSM для этой зоны")
    return Response(p.read_bytes(), media_type="application/json", headers={"Content-Encoding": "gzip"})


@app.get("/citysat/{area}.jpg")
def citysat(area: str):
    """Спутниковый снимок зоны города (showcase/city_sat.py; кэш, в git не попадает)."""
    p = HERE / ".tiles" / "city" / f"{area}.jpg"
    if not p.exists():
        raise HTTPException(404, "Снимок зоны не скачан: python -m showcase.city_sat")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/attribution")
def attribution():
    return {"text": ATTRIBUTION}


@app.post("/section")
async def section(lon: float = Body(), lat: float = Body(), a: float = Body(0.5),
                  wavelength: float = Body(10e3)):
    """Путь волны через точку (lon, lat) и три модели на нём."""
    if not (0.05 <= a <= 3.0 and 2e3 <= wavelength <= 40e3):
        raise HTTPException(400, "Амплитуда 0,05–3 м, длина волны 2–40 км")
    try:
        return await run_in_threadpool(solve_section, lon, lat, a, wavelength, pool)
    except ValueError as e:
        raise HTTPException(400, str(e))
