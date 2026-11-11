from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    store: int
    item: int


class PredictionResponse(BaseModel):
    store: int
    item: int
    prediction_date: date
    demand_forecast: float


class BatchPredictionRequest(BaseModel):
    series: list[PredictionRequest] = Field(
        min_length=1,
        max_length=1000,
        description="Lista de lojas e itens para previsão de demanda",
    )

    @model_validator(mode="after")
    def reject_duplicate_series(self):
        pairs = [(s.store, s.item) for s in self.series]
        if len(pairs) != len(set(pairs)):
            raise ValueError("series must not contain duplicate (store, item) pairs")
        return self


class SeriesPrediction(BaseModel):
    store: int
    item: int
    demand_forecast: float


SeriesErrorReason = Literal[
    "unknown_category",
    "empty_history",
    "incomplete_history",
    "feature_row_mismatch",
]


class SeriesError(BaseModel):
    store: int
    item: int
    reason: SeriesErrorReason


class BatchPredictionResponse(BaseModel):
    prediction_date: date
    predictions: list[SeriesPrediction]
    errors: list[SeriesError] = Field(default_factory=list)