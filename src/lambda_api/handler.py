from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from db import get_connection, get_engine, get_sales_history, get_simulated_date
from fastapi import FastAPI
from inference import build_prediction_row
from mangum import Mangum
from schemas import PredictionRequest, PredictionResponse

MODEL_PATH = Path(__file__).parent.parent.parent / "artifacts" / "model.pkl"

# --- Tudo aqui fora roda só uma vez, no cold start ---
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
store_categories = bundle["categories"]["store"]
item_categories = bundle["categories"]["item"]

store_dtype = pd.api.types.CategoricalDtype(categories=store_categories)
item_dtype = pd.api.types.CategoricalDtype(categories=item_categories)

conn = get_connection()
engine = get_engine()

app = FastAPI()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    simulated_date = get_simulated_date(conn)
    prediction_date = simulated_date + timedelta(days=1)

    history = get_sales_history(engine, request.store, request.item, simulated_date)
    row = build_prediction_row(history, request.store, request.item, prediction_date)

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