"""Контракт между моделями, сервером и страницей.

Этот файл читают все. Любая модель получает Params и возвращает Result.
Меняйте поля только по договорённости с командой: от них зависят и модели, и фронтенд.
"""
from dataclasses import dataclass, field


@dataclass
class Params:
    """Входные параметры расчёта (все величины в СИ)."""
    a: float = 0.5             # амплитуда волны, м
    wavelength: float = 100.0  # длина волны, м
    depth: float = 5.0         # глубина воды (для slope — у левого края), м
    bottom: str = "flat"       # дно: "flat" — ровное, "slope" — склон к берегу, "gulf" — разрез залива
    depth_shore: float = 1.0   # для slope — глубина у правого края, м; 0 и меньше — сухой берег
    L: float = 1000.0          # длина расчётной области по x, м (для gulf берётся из данных)
    t_end: float = 60.0        # конечное время расчёта, с
    nx: int = 400              # число узлов сетки по x
    n_frames: int = 100        # число кадров по времени, которые вернёт модель
    extra: dict[str, float] = field(default_factory=dict)  # параметры отдельной модели, см. WaveModel.extra_params


@dataclass
class Result:
    """Результат расчёта, который рисует страница.

    Модель с dim = 1 возвращает eta[кадр][ix] и h[ix].
    Модель с dim = 2 возвращает ещё y, а eta[кадр][iy][ix] и h[iy][ix] — карты.
    """
    x: list[float]             # координаты узлов, м
    t: list[float]             # моменты времени кадров, с; длина n_frames
    eta: list                  # возвышение свободной поверхности η, м
    h: list                    # глубина под невозмущённой водой, м; на суше 0 и меньше
    y: list[float] | None = None  # координаты узлов по y, м; только у dim = 2


class WaveModel:
    """Базовый класс модели. Новая модель наследуется от него и реализует solve."""
    id: str            # короткий идентификатор латиницей, используется в API (например, "linear")
    name: str          # название для выпадающего списка на странице
    description: str   # пара предложений: какая физика, какие допущения
    dim: int = 1       # 1 — профиль η(x), 2 — карта η(x, y)
    # Параметры, нужные только этой модели. Страница строит для них поля, значения
    # приходят в Params.extra. Формат элемента:
    # {"name": "wind_speed", "label": "Скорость ветра", "unit": "м/с", "default": 0, "min": 0, "max": 40}
    extra_params: list[dict] = []

    def solve(self, p: Params) -> Result:
        """Посчитать эволюцию волны по параметрам p и вернуть кадры η."""
        raise NotImplementedError

    def extras(self, p: Params) -> dict[str, float]:
        """Значения extra_params: присланные в p.extra, остальные — по умолчанию."""
        values = {e["name"]: e["default"] for e in self.extra_params}
        values.update(p.extra)
        return values
