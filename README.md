# Волны у берега Финского залива

Веб-сервис: задаём параметры, выбираем модель волны, смотрим анимацию профиля воды η(x, t).
«Модель» здесь — математическая модель: система уравнений, её допущения и численный метод решения (машинного обучения в проекте нет).
Цель проекта — понять, при каких амплитудах и временах важны нелинейные эффекты.

## Запуск

Из папки `waves-gulf/` (Python 3.11):

```
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
- `backend/registry.py` — список моделей.
- `backend/models/` — по одному файлу на модель. `demo.py` — заглушка для проверки связки, не физика.
- `backend/main.py` — API: `GET /models`, `POST /simulate`, `/` (страница).

## Как добавить модель

1. Создать файл `backend/models/<имя>.py`.
2. В нём унаследовать класс от `WaveModel` из `backend/contract.py`, задать `id`, `name`, `description` и реализовать `solve(self, p: Params) -> Result` (образец — `demo.py`).
3. В `backend/registry.py` импортировать класс и добавить строку `register(МояМодель())`.
4. Перезапустить сервер — модель появится в выпадающем списке.


