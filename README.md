# Mortgage Approval ML Service

Лабораторная работа №10. ML-сервис, предсказывающий одобрение/отказ по заявке на ипотеку.
Backend на **FastAPI**, фронтенд — статическая HTML-страница на **Tailwind (CDN)**, модель — sklearn-пайплайн, сериализованный в `.pkl`.

---

## Структура проекта

```
.
├── app/
│   ├── main.py          # FastAPI: /upload-model, /predict, /predict-from-csv, /health
│   ├── schemas.py       # Pydantic-схемы запроса/ответа
│   └── services.py      # Загрузка .pkl, предсказания, расчёт ROC-AUC
├── models/              # сюда сохраняются загруженные .pkl
├── static/
│   └── index.html       # минималистичный UI
├── tests/
│   └── test_api.py      # тесты на fastapi.testclient
├── .sourcecraft/
│   └── ci.yaml          # CI: ruff + pytest
├── train.py             # обучение sklearn-пайплайна (ML-часть)
├── research.ipynb       # EDA / эксперименты
├── loan_data.csv        # исходный датасет
├── mortgage_pipeline.pkl # уже обученная модель (RandomForest)
├── sample_single_client.json  # пример запроса для /predict
├── test_with_target.csv       # CSV с колонкой loan_status (для ROC-AUC)
├── test_without_target.csv    # CSV без таргета
└── pyproject.toml
```

---

## Запуск

Нужен [uv](https://docs.astral.sh/uv/).

```bash
# 1. установить зависимости
uv sync --all-groups

# 2. (опционально) положить обученную модель в models/, чтобы она подхватилась на старте
cp mortgage_pipeline.pkl models/

# 3. поднять сервер
uv run uvicorn app.main:app --reload
```

- UI: <http://127.0.0.1:8000/>
- Swagger: <http://127.0.0.1:8000/docs>
- Healthcheck: <http://127.0.0.1:8000/health>

На старте сервис ищет любой `.pkl` в `models/` и автоматически подгружает его.
Если модели нет — `/predict` и `/predict-from-csv` вернут `400 "Model is not loaded"`. Загрузить модель можно через `POST /upload-model` или прямо из UI.

### Тесты и линт

```bash
uv run ruff check .
uv run pytest
```

---

## API

| Метод  | Путь                | Описание |
|--------|---------------------|----------|
| `GET`  | `/health`           | Статус сервиса и факт загрузки модели |
| `GET`  | `/`                 | Отдаёт `static/index.html` |
| `POST` | `/upload-model`     | `multipart/form-data`, поле `file` — `.pkl`. Сохраняет в `models/` и подгружает |
| `POST` | `/predict`          | JSON `{ "objects": [ { ...признаки... } ] }` → массив `{features, loan_status}` |
| `POST` | `/predict-from-csv` | `multipart/form-data` с CSV. Если есть `loan_status` — считает ROC-AUC. Возвращает JSON со строками + `predicted_loan_status` |

Пример тела для `/predict` — см. `sample_single_client.json`.

---

## Что сделано

**ML (готово, отдельный коммит `ML part completed`)**
- EDA в `research.ipynb`, очистка датасета от выбросов.
- `train.py`: `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) → `RandomForestClassifier` в едином `Pipeline`.
- Сериализация в `mortgage_pipeline.pkl` через `joblib`.
- Сгенерированы артефакты для проверки API: `sample_single_client.json`, `test_with_target.csv`, `test_without_target.csv`.

**Backend / API**
- FastAPI-приложение с lifespan-инициализацией модели из `models/`.
- Все три обязательных эндпоинта + `/health` + раздача UI.
- Корректная обработка ошибок: `400` при ненагруженной модели, неверном расширении, пустом / невалидном CSV.
- Pydantic-схемы для валидации входа клиента.

**Frontend**
- Одностраничный UI (`static/index.html`) на Tailwind CDN: форма клиента → `/predict`, аплоад `.pkl`, аплоад CSV с превью результата и ROC-AUC, индикатор статуса модели.

**Качество / CI**
- `tests/test_api.py` — 9 тестов на `TestClient`, покрывают все эндпоинты и негативные сценарии.
- `ruff` сконфигурирован в `pyproject.toml` (исключены `*.ipynb`, `.venv`, `models`).
- `.sourcecraft/ci.yaml` — пайплайн `uv sync` → `ruff check` → `pytest` на каждый push/PR.
- `README.md`, `.gitignore` (игнор кэшей и `models/*.pkl`).

---

## Что осталось / возможные улучшения

- **Сравнение моделей по ТЗ.** Сейчас в `train.py` обучается только `RandomForestClassifier`. По заданию нужно сравнить ≥ 2 алгоритмов (LogReg / RF / GBM) и выбрать лучший по ROC-AUC.
- **Feature selection.** В пайплайне используются все признаки, явного отбора нет.
- **Документация ML-части** в README (метрики, какие модели сравнили, итоговый ROC-AUC на тесте).
- **Чистка корня репозитория.** Заглушка `main.py` (`print("Hello from pylab10-ml!")`) больше не нужна. Дубликат `mortgage_pipeline.pkl` в корне можно удалить, оставив только копию в `models/`.
- **Опциональные пункты ТЗ:** Docker-контейнеризация, расширенное логирование, визуализация метрик (ROC-кривая) на фронте.
- **Линтинг фронтенда в CI** (по ТЗ требуется). Сейчас CI линтит только Python. Можно добавить `prettier --check static/` или `eslint`.
- **CORS / hosting.** Если фронт будет деплоиться отдельно от API — нужен `CORSMiddleware`.
