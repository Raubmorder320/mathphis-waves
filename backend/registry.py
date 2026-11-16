"""Список доступных моделей. Новая модель регистрируется одной строкой внизу файла."""
from backend.contract import WaveModel
from backend.models.boussinesq import BoussinesqModel
from backend.models.demo import DemoModel
from backend.models.kdv import KdVModel
from backend.models.linear import LinearModel
from backend.models.saint_venant import SaintVenantModel
from backend.models.shallow_2d import Shallow2DModel

MODELS: dict[str, WaveModel] = {}  # id модели -> экземпляр модели


def register(model: WaveModel) -> None:
    """Добавить модель в список; после этого она появится в GET /models."""
    MODELS[model.id] = model


register(DemoModel())
register(LinearModel())
register(BoussinesqModel())
register(KdVModel())
register(SaintVenantModel())
register(Shallow2DModel())
