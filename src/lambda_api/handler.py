from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from db import DatabaseUnavailableError, get_sales_history, get_simulated_date
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from inference import build_prediction_row
from mangum import Mangum
from schemas import PredictionRequest, PredictionResponse

from features import HISTORY_DAYS

MODEL_PATH = Path(__file__).parent / "model.pkl"

# --- Tudo aqui fora roda só uma vez, no cold start ---
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
store_categories = bundle["categories"]["store"]
item_categories = bundle["categories"]["item"]

store_dtype = pd.api.types.CategoricalDtype(categories=store_categories)
item_dtype = pd.api.types.CategoricalDtype(categories=item_categories)

app = FastAPI()

@app.exception_handler(DatabaseUnavailableError)
def database_unavailable_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": "Database is unavailable."}
    )

@app.post("/predict", response_model=PredictionResponse)
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


handler = Mangum(app)
