import logging
import os
import secrets
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from db import (
    DatabaseUnavailableError,
    get_sales_history,
    get_sales_history_batch,
    get_simulated_date,
)
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from inference import build_prediction_row, build_prediction_rows, split_valid_pairs
from mangum import Mangum
from schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    SeriesError,
    SeriesPrediction,
)

from features import HISTORY_DAYS

MODEL_PATH = Path(__file__).parent / "model.pkl"
API_KEY = os.getenv("API_KEY")

# --- Tudo aqui fora roda só uma vez, no cold start ---
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
store_categories = bundle["categories"]["store"]
item_categories = bundle["categories"]["item"]

store_dtype = pd.api.types.CategoricalDtype(categories=store_categories)
item_dtype = pd.api.types.CategoricalDtype(categories=item_categories)
store_set = set(store_categories)
item_set = set(item_categories)

logger = logging.getLogger(__name__)

if API_KEY is None:
    logger.warning("API_KEY is not configured.")

app = FastAPI()

def require_api_key(x_api_key: str = Header(default=None)):
    if not API_KEY or x_api_key is None or not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.exception_handler(DatabaseUnavailableError)
def database_unavailable_handler(request, exc):
    logger.error("Database is unavailable.")
    return JSONResponse(
        status_code=503,
        content={"detail": "Database is unavailable."}
    )

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict(request: PredictionRequest):
    if request.store not in store_categories or request.item not in item_categories:
        raise HTTPException(status_code=422, detail="Store and item must be valid.")

    simulated_date = get_simulated_date()

    if simulated_date is None:
        raise HTTPException(
            status_code=500, detail="Simulated date not found in the database."
        )

    prediction_date = simulated_date + timedelta(days=1)

    history = get_sales_history(
        request.store, request.item, simulated_date, HISTORY_DAYS
    )

    if history.empty:
        raise HTTPException(
            status_code=422,
            detail="Sales history is empty for the given store and item.",
        )

    first_expected = pd.Timestamp(simulated_date) - pd.Timedelta(days=HISTORY_DAYS - 1)
    complete = (
        len(history) == HISTORY_DAYS
        and history["date"].max() == pd.Timestamp(simulated_date)
        and history["date"].min() == first_expected
    )

    if not complete:
        raise HTTPException(
            status_code=422,
            detail=f"Sales history is incomplete. Expected {HISTORY_DAYS} days of data from {first_expected.date()} to {simulated_date}.",
        )

    row = build_prediction_row(history, request.store, request.item, prediction_date)

    if row.empty or row["date"].iloc[0] != pd.Timestamp(prediction_date):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build prediction row for store {request.store} and item {request.item}.",
        )

    row["store"] = row["store"].astype(store_dtype)
    row["item"] = row["item"].astype(item_dtype)

    X = row[feature_columns]
    dmatrix = xgb.DMatrix(X, enable_categorical=True)
    prediction = model.predict(dmatrix)[0]

    return PredictionResponse(
        store=request.store,
        item=request.item,
        prediction_date=prediction_date,
        demand_forecast=float(prediction),
    )


def predict_from_rows(rows):
    """Prevê para as linhas já preparadas e devolve uma lista de SeriesPrediction."""
    if rows.empty:
        return []

    rows = rows.copy()
    rows["store"] = rows["store"].astype(store_dtype)
    rows["item"] = rows["item"].astype(item_dtype)

    dmatrix = xgb.DMatrix(rows[feature_columns], enable_categorical=True)
    values = model.predict(dmatrix)

    # store e item vêm da própria linha prevista, então a previsão nunca troca de série
    return [
        SeriesPrediction(store=int(store), item=int(item), demand_forecast=float(value))
        for store, item, value in zip(rows["store"], rows["item"], values)
    ]


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(require_api_key)],
)
def predict_batch(request: BatchPredictionRequest):
    pairs = [(s.store, s.item) for s in request.series]

    # 1. Categoria desconhecida: separa ANTES de ir ao banco
    known = [(s, i) for s, i in pairs if s in store_set and i in item_set]
    known_set = set(known)
    errors = [
        SeriesError(store=s, item=i, reason="unknown_category")
        for s, i in pairs
        if (s, i) not in known_set
    ]

    # 2. Data simulada
    simulated_date = get_simulated_date()
    if simulated_date is None:
        raise HTTPException(
            status_code=500, detail="Simulated date not found in the database."
        )
    prediction_date = simulated_date + timedelta(days=1)

    # 3. Histórico em lote e validação por série
    history = get_sales_history_batch(known, simulated_date, HISTORY_DAYS)
    valid, history_errors = split_valid_pairs(history, known, simulated_date, HISTORY_DAYS)
    errors += [SeriesError(store=s, item=i, reason=r) for s, i, r in history_errors]

    # 4. Features e previsão
    rows = build_prediction_rows(history, valid, prediction_date)
    predictions = predict_from_rows(rows)

    # pares que passaram na validação mas não voltaram das features
    returned = {(p.store, p.item) for p in predictions}
    errors += [
        SeriesError(store=s, item=i, reason="feature_row_mismatch")
        for s, i in valid
        if (s, i) not in returned
    ]

    # 5. Mesmo formato de resposta sempre (200), mesmo que nenhuma série dê certo
    return BatchPredictionResponse(
        prediction_date=prediction_date,
        predictions=predictions,
        errors=errors,
    )


handler = Mangum(app)