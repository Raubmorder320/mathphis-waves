"""Список доступных моделей. Новая модель регистрируется одной строкой внизу файла."""
from backend.contract import WaveModel
from backend.models.demo import DemoModel
from backend.models.saint_venant import SaintVenantModel

MODELS: dict[str, WaveModel] = {}  # id модели -> экземпляр модели


def register(model: WaveModel) -> None:
    """Добавить модель в список; после этого она появится в GET /models."""
    MODELS[model.id] = model


register(DemoModel())
register(SaintVenantModel())
