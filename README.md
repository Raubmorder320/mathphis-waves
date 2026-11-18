# Волны у берега Финского залива

Веб-сервис: задаём параметры, выбираем модель волны, смотрим анимацию профиля воды η(x, t).
«Модель» здесь — математическая модель: система уравнений, её допущения и численный метод решения (машинного обучения в проекте нет).
Тема: «Волны залива набегают на берег. Определите, где нелинейность важна». Теория — в `docs/theory.md`, задания — в `TASKS.md`, архитектура, результаты и план доклада — в `docs/report.md`.

## Запуск

Из папки `waves-gulf/` (Python 3.11):

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Открыть http://127.0.0.1:8000

## Как это устроено

```
frontend/index.html ──POST /simulate {model_id, params}──▶ backend/main.py
                                                              │
                                              registry.MODELS[model_id].solve(Params)
                                                              │
        график η(x) по кадрам ◀──────── JSON Result {x, t, eta, h} ◀┘
```

- `backend/contract.py` — **контракт**: `Params` (вход), `Result` (выход), `WaveModel` (базовый класс). Начинать чтение отсюда.
- `backend/common.py` — **общая постановка**: сетка, дно (`flat`, `slope`, `gulf`) и начальное условие. Все одномерные модели берут их отсюда, иначе решения нельзя сравнивать.
- `backend/registry.py` — список моделей.
- `backend/models/` — по одному файлу на модель. `demo.py` — заглушка для проверки связки, не физика.
- `backend/main.py` — API: `GET /models`, `POST /simulate`, `/` (страница). Если модель бросает `NotImplementedError` или `ValueError`, страница получает ответ 400 с текстом ошибки.
- `data/` — карта глубин и разрез залива, `analysis/` — сравнение моделей и рисунки, `docs/` — теория.

## Как добавить модель

1. Создать файл `backend/models/<имя>.py`.
2. В нём унаследовать класс от `WaveModel` из `backend/contract.py`, задать `id`, `name`, `description` и реализовать `solve(self, p: Params) -> Result` (образец — `demo.py`).
   - Сетку, дно и η₀ брать из `setup(p)` в `backend/common.py`, начальную скорость — из `u_linear` или `u_simple_wave` там же.
   - Если модель не умеет какое-то дно, бросить `NotImplementedError` с понятным текстом.
   - Двумерная модель задаёт `dim = 2` и заполняет `Result.y`.
   - Свои параметры (ветер, трение) объявить в `extra_params`, значения брать через `self.extras(p)`.
3. В `backend/registry.py` импортировать класс и добавить строку `register(МояМодель())`.
4. Перезапустить сервер — модель появится в выпадающем списке.
