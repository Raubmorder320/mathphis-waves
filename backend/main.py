from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse

from backend.contract import Params, Result
from backend.registry import MODELS

INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

app = FastAPI(title="Волны у берега Финского залива")


@app.get("/models")
def list_models():
    """Список моделей для страницы: что показать в списке и какие поля построить."""
    return [
        {"id": m.id, "name": m.name, "description": m.description,
         "dim": m.dim, "extra_params": m.extra_params}
        for m in MODELS.values()
    ]


@app.post("/simulate")
def simulate(model_id: str = Body(), params: Params = Body()) -> Result:
    """Тело запроса: {"model_id": "...", "params": {...поля Params...}}."""
    if model_id not in MODELS:
        raise HTTPException(404, f"Нет модели {model_id!r}")
    try:
        return MODELS[model_id].solve(params)
    except (NotImplementedError, ValueError) as e:
        # Модель не умеет такую постановку или параметры недопустимы — показать текст на странице.
        raise HTTPException(400, str(e))


@app.get("/")
def index():
    return FileResponse(INDEX)
