"""Тесты API. Используют фикстуру с пред-загруженной моделью из models/."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import find_default_model, load_model_from_disk

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SAMPLE_JSON = DATA_DIR / "sample_single_client.json"
CSV_WITH_TARGET = DATA_DIR / "test_with_target.csv"
CSV_WITHOUT_TARGET = DATA_DIR / "test_without_target.csv"
MODEL_PATH = MODELS_DIR / "mortgage_pipeline.pkl"


@pytest.fixture()
def client_no_model():
    """Клиент без загруженной модели."""
    with TestClient(app) as c:
        c.app.state.model = None
        c.app.state.model_path = None
        yield c


@pytest.fixture()
def client_with_model():
    """Клиент с подгруженной обученной моделью."""
    path = find_default_model() or MODEL_PATH
    if not path or not Path(path).exists():
        pytest.skip("Модель не найдена для интеграционных тестов")
    with TestClient(app) as c:
        c.app.state.model = load_model_from_disk(path)
        c.app.state.model_path = str(path)
        yield c


def test_health_ok(client_no_model):
    r = client_no_model.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_without_model_returns_400(client_no_model):
    payload = {"objects": [json.loads(SAMPLE_JSON.read_text())]}
    r = client_no_model.post("/predict", json=payload)
    assert r.status_code == 400
    assert r.json()["detail"] == "Model is not loaded"


def test_predict_from_csv_without_model_returns_400(client_no_model):
    files = {"file": ("a.csv", b"a,b\n1,2\n", "text/csv")}
    r = client_no_model.post("/predict-from-csv", files=files)
    assert r.status_code == 400


def test_upload_model_rejects_non_pkl(client_no_model):
    files = {"file": ("model.txt", b"not a model", "text/plain")}
    r = client_no_model.post("/upload-model", files=files)
    assert r.status_code == 400


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="нет mortgage_pipeline.pkl")
def test_upload_model_ok(client_no_model):
    payload = MODEL_PATH.read_bytes()
    files = {"file": ("uploaded.pkl", payload, "application/octet-stream")}
    r = client_no_model.post("/upload-model", files=files)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_predictions(client_with_model):
    sample = json.loads(SAMPLE_JSON.read_text())
    r = client_with_model.post("/predict", json={"objects": [sample, sample]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["loan_status"] in (0, 1)
    assert 0.0 <= body["predictions"][0]["loan_probability"] <= 1.0
    assert body["predictions"][0]["features"]["person_age"] == sample["person_age"]


def test_predict_from_csv_with_target_returns_roc_auc(client_with_model):
    if not CSV_WITH_TARGET.exists():
        pytest.skip("нет test_with_target.csv")
    files = {"file": ("with_target.csv", CSV_WITH_TARGET.read_bytes(), "text/csv")}
    r = client_with_model.post("/predict-from-csv", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] > 0
    assert body["roc_auc"] is None or 0.0 <= body["roc_auc"] <= 1.0
    assert "predicted_loan_status" in body["data"][0]


def test_predict_from_csv_without_target(client_with_model):
    if not CSV_WITHOUT_TARGET.exists():
        pytest.skip("нет test_without_target.csv")
    files = {"file": ("no_target.csv", CSV_WITHOUT_TARGET.read_bytes(), "text/csv")}
    r = client_with_model.post("/predict-from-csv", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["roc_auc"] is None
    assert "predicted_loan_status" in body["data"][0]


def test_predict_from_csv_empty(client_with_model):
    df = pd.DataFrame()
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    files = {"file": ("empty.csv", buf.getvalue(), "text/csv")}
    r = client_with_model.post("/predict-from-csv", files=files)
    assert r.status_code == 400
