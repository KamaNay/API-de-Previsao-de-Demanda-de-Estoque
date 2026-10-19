from pathlib import Path

import joblib
import pandas as pd
from db import get_engine, get_sales_history
from inference import build_prediction_row

from features import build_feature_frame

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "train.csv"
STORE, ITEM = 1, 1
TARGET_DATE = pd.Timestamp("2017-12-31")

# Caminho do treino: features calculadas sobre o CSV inteiro
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
full = build_feature_frame(df)
train_row = full[(full["store"] == STORE) & (full["item"] == ITEM) & (full["date"] == TARGET_DATE)]

# Caminho da Lambda: histórico só até o dia ANTERIOR ao alvo
history = get_sales_history(get_engine(), STORE, ITEM, TARGET_DATE - pd.Timedelta(days=1))
lambda_row = build_prediction_row(history, STORE, ITEM, TARGET_DATE.date())

cols = joblib.load(Path(__file__).parent.parent.parent / "artifacts" / "model.pkl")["feature_columns"]

print(train_row[cols].reset_index(drop=True))
print(lambda_row[cols].reset_index(drop=True))