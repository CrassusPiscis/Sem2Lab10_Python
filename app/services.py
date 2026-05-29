"""Бизнес-логика: загрузка модели и формирование предсказаний."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TARGET_COLUMN = "loan_status"
PREDICTED_COLUMN = "predicted_loan_status"


def ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def save_uploaded_model(filename: str, payload: bytes) -> Path:
    """Сохраняет .pkl-файл в models/ и возвращает путь."""
    ensure_models_dir()
    target = MODELS_DIR / Path(filename).name
    target.write_bytes(payload)
    return target


def load_model_from_disk(path: Path) -> Any:
    logger.info("Загрузка модели из %s", path)
    return joblib.load(path)


def find_default_model() -> Path | None:
    """Ищет любой .pkl в каталоге models/ для авто-подгрузки на старте."""
    ensure_models_dir()
    pkls = sorted(MODELS_DIR.glob("*.pkl"))
    return pkls[0] if pkls else None


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def predict_records(model: Any, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Возвращает список словарей с признаками и предсказанным loan_status."""
    df = records_to_dataframe(records)
    preds = model.predict(df)
    out: list[dict[str, Any]] = []
    for record, pred in zip(records, preds):
        out.append({"features": record, "loan_status": int(pred)})
    return out


def predict_dataframe(model: Any, df: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    """Прогоняет DataFrame через модель, считает ROC-AUC если есть target."""
    roc_auc: float | None = None
    if TARGET_COLUMN in df.columns:
        y_true = df[TARGET_COLUMN]
        features = df.drop(columns=[TARGET_COLUMN])
        proba = model.predict_proba(features)[:, 1]
        try:
            roc_auc = float(roc_auc_score(y_true, proba))
        except ValueError:
            roc_auc = None
        preds = (proba >= 0.5).astype(int)
    else:
        features = df
        preds = model.predict(features)

    result = df.copy()
    result[PREDICTED_COLUMN] = preds.astype(int)
    return result, roc_auc
