# Mortgage Approval ML Service

Лабораторная работа №10. Консультирующий ML-сервис, предсказывающий **одобрение или отказ в выдаче ипотеки** по анкете клиента.

Backend на **FastAPI**, фронтенд — статическая HTML-страница на **Tailwind (CDN)**, модель — sklearn-`Pipeline` (предобработка + классификатор), сериализованный в `.pkl` через `joblib`.

## Что должно получиться

Готовое веб-приложение, у которого есть:

- **Обученная модель** (`models/mortgage_pipeline.pkl`) — на вход принимает признаки клиента (возраст, доход, сумма кредита, цель, кредитный рейтинг и т. д.), на выходе даёт `loan_status` (0 — отказ, 1 — одобрено) и `predict_proba`.
- **REST API** на FastAPI с тремя обязательными методами:
  - `POST /upload-model` — загрузить новый `.pkl` (например, после переобучения) без перезапуска сервиса.
  - `POST /predict` — получить предсказание по одному или нескольким клиентам в формате JSON.
  - `POST /predict-from-csv` — прогнать весь CSV: вернуть исходные строки с колонкой `predicted_loan_status` и, если в файле есть `loan_status`, посчитать ROC-AUC на лету.
- **Минималистичный UI** на `/`: форма «один клиент → предсказание», аплоад `.pkl`, аплоад CSV с превью результата и метрикой, индикатор «модель загружена / не загружена».

## Как это работает

1. На старте FastAPI ищет любой `.pkl` в каталоге `models/` и подгружает его в `app.state.model`. Если модели нет, эндпоинты предсказания отвечают `400 "Model is not loaded"` — фронт показывает соответствующий статус.
2. ML-инженер обучает модель скриптом `uv run python -m ml.train`: скрипт читает `data/loan_data.csv`, чистит выбросы, строит `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) → `RandomForestClassifier` в едином `Pipeline`, сохраняет его в `models/mortgage_pipeline.pkl` и обновляет тестовые артефакты в `data/`.
3. Клиент (фронт или curl) шлёт JSON со списком объектов в `/predict`. Бэкенд собирает `pandas.DataFrame`, прогоняет через `model.predict(df)` и возвращает массив `{features, loan_status}`.
4. Для пакетной проверки клиент шлёт CSV в `/predict-from-csv`. Сервис читает его в DataFrame; если есть колонка `loan_status`, отделяет её, считает `predict_proba` и `roc_auc_score`; затем добавляет `predicted_loan_status` и возвращает весь набор данных в JSON.
5. Новую обученную модель можно «прокатить» в прод одним запросом `POST /upload-model` — файл сохраняется в `models/`, тут же подгружается в память, и следующий запрос на предсказание уже идёт через неё.

---

## Структура проекта

```
.
├── app/                          # FastAPI backend
│   ├── main.py                   # endpoints: /upload-model, /predict, /predict-from-csv, /health
│   ├── schemas.py                # Pydantic-схемы запроса/ответа
│   └── services.py               # загрузка .pkl, предсказания, расчёт ROC-AUC
├── ml/
│   └── train.py                  # обучение sklearn-пайплайна
├── notebooks/
│   └── research.ipynb            # EDA / эксперименты
├── data/                         # датасет и тестовые артефакты
│   ├── loan_data.csv             # исходный датасет
│   ├── sample_single_client.json # пример тела для /predict
│   ├── test_with_target.csv      # CSV с loan_status (для проверки ROC-AUC)
│   └── test_without_target.csv   # CSV без таргета
├── models/                       # сюда сохраняются .pkl (gitignored)
├── static/
│   └── index.html                # минималистичный UI
├── tests/
│   └── test_api.py               # тесты на fastapi.testclient
├── .sourcecraft/
│   └── ci.yaml                   # CI: ruff + pytest
├── pyproject.toml
├── README.md
└── TASK.md
```

---

## Запуск

Нужен [uv](https://docs.astral.sh/uv/).

```bash
# 1. установить зависимости
uv sync --all-groups

# 2. обучить модель (положит mortgage_pipeline.pkl в models/ и обновит data/*.csv|json)
uv run python -m ml.train

# 3. поднять сервер
uv run uvicorn app.main:app --reload
```

На старте сервис ищет любой `.pkl` в `models/`. Если предобученной модели нет — её можно загрузить через `POST /upload-model` или прямо из UI.

- UI: <http://127.0.0.1:8000/>
- Swagger: <http://127.0.0.1:8000/docs>
- Healthcheck: <http://127.0.0.1:8000/health>

Если модели нет — `/predict` и `/predict-from-csv` вернут `400 "Model is not loaded"`.

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

Пример тела для `/predict` — см. `data/sample_single_client.json`.

---

## Что сделано

**ML (готово, отдельный коммит `ML part completed`)**
- EDA в `notebooks/research.ipynb`, очистка датасета от выбросов.
- `ml/train.py`: `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) → `RandomForestClassifier` в едином `Pipeline`.
- Сериализация в `models/mortgage_pipeline.pkl` через `joblib`.
- Сгенерированы артефакты для проверки API в `data/`: `sample_single_client.json`, `test_with_target.csv`, `test_without_target.csv`.

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

- **Сравнение моделей по ТЗ.** Сейчас в `ml/train.py` обучается только `RandomForestClassifier`. По заданию нужно сравнить ≥ 2 алгоритмов (LogReg / RF / GBM) и выбрать лучший по ROC-AUC.
- **Feature selection.** В пайплайне используются все признаки, явного отбора нет.
- **Документация ML-части** в README (метрики, какие модели сравнили, итоговый ROC-AUC на тесте).
- **Опциональные пункты ТЗ:** Docker-контейнеризация, расширенное логирование, визуализация метрик (ROC-кривая) на фронте.
- **Линтинг фронтенда в CI** (по ТЗ требуется). Сейчас CI линтит только Python. Можно добавить `prettier --check static/` или `eslint`.
- **CORS / hosting.** Если фронт будет деплоиться отдельно от API — нужен `CORSMiddleware`.
