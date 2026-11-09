from datetime import date

from pydantic import BaseModel, Field


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
    
class BatchPredictionRequest(BaseModel):
    # lista de lojas e itens
    series: list[PredictionRequest] = Field(min_length=1, max_length=1000, description="Lista de lojas e itens para previsão de demanda")
    
class SeriesPrediction(BaseModel):
    # loja, item, data prevista e previsão de demanda
    store: int
    item: int
    prediction_date: date
    demand_forecast: float
    
class SeriesError(BaseModel):
    # loja, item e mensagem de erro
    store: int
    item: int
    error_message: str
    
class BatchPredictionResponse(BaseModel):
    # lista de previsões
    prediction_date: date
    predictions: list[SeriesPrediction]
    errors: list[SeriesError] = []