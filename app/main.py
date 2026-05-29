"""FastAPI-приложение для предсказания одобрения ипотеки."""
from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from app.schemas import PredictRequest, PredictResponse
from app.services import (
    find_default_model,
    load_model_from_disk,
    predict_dataframe,
    predict_records,
    save_uploaded_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_path = None
    default = find_default_model()
    if default is not None:
        try:
            app.state.model = load_model_from_disk(default)
            app.state.model_path = str(default)
            logger.info("Модель по умолчанию загружена: %s", default)
        except Exception as exc:
            logger.warning("Не удалось загрузить модель по умолчанию: %s", exc)
    yield


app = FastAPI(
    title="Mortgage Approval Service",
    description="ML-сервис для предсказания одобрения ипотеки",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model_or_400(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=400, detail="Model is not loaded")
    return model


@app.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
        "model_path": request.app.state.model_path,
    }


@app.get("/")
def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html не найден")
    return FileResponse(index_path)


@app.post("/upload-model")
async def upload_model(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Ожидается файл с расширением .pkl")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Пустой файл")

    try:
        path = save_uploaded_model(file.filename, payload)
        model = load_model_from_disk(path)
    except Exception as exc:
        logger.exception("Ошибка загрузки модели")
        raise HTTPException(status_code=400, detail=f"Не удалось загрузить модель: {exc}") from exc

    request.app.state.model = model
    request.app.state.model_path = str(path)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "filename": file.filename, "path": str(path)},
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    model = get_model_or_400(request)
    records = [obj.model_dump() for obj in payload.objects]
    try:
        results = predict_records(model, records)
    except Exception as exc:
        logger.exception("Ошибка предсказания")
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {exc}") from exc
    return PredictResponse(predictions=results)


@app.post("/predict-from-csv")
async def predict_from_csv(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    model = get_model_or_400(request)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Пустой файл")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV не содержит строк")

    try:
        result_df, roc_auc = predict_dataframe(model, df)
    except Exception as exc:
        logger.exception("Ошибка предсказания по CSV")
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {exc}") from exc

    return JSONResponse(
        content={
            "roc_auc": roc_auc,
            "rows": int(len(result_df)),
            "data": result_df.to_dict(orient="records"),
        }
    )
