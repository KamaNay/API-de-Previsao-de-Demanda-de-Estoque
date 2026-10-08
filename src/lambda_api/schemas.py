from datetime import date

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    # loja e item
    store: int
    item: int


class PredictionResponse(BaseModel):
    # loja, item, data prevista e previsão de demanda
    store: int
    item: int
    prediction_date: date
    demand_forecast: float