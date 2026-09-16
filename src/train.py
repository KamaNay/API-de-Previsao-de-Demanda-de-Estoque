import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import build_feature_frame

DATA_PATH = Path(__file__).parent.parent / "data" / "train.csv"
ARTIFACTS_PATH = Path(__file__).parent.parent / "artifacts"

TEST_START = "2017-10-01"  # último trimestre como teste

def split_train_test(df, test_start):
    train = df[df["date"] < test_start].copy()
    test = df[df["date"] >= test_start].copy()
    return train, test

FEATURE_COLUMNS = [
    "store", "item", "day_of_week", "month", "year", "is_weekend",
    "lag_7", "lag_14", "lag_365",
    "rolling_mean_7", "rolling_mean_14",
]

def prepare_categorical(df):
    df = df.copy()
    df["store"] = df["store"].astype("category")
    df["item"] = df["item"].astype("category")
    return df

def naive_baseline_predictions(test_df):
    # A previsão mais simples possível: "vai vender igual vendeu na mesma
    # semana passada".
    return test_df["lag_7"]

def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label}: MAE={mae:.2f}  RMSE={rmse:.2f}")
    
    return mae, rmse
    
def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        n_estimators=500,     
        max_depth=7,          
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True, 
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, feature_columns, test_start, path):
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "test_start": test_start,
    }
    joblib.dump(bundle, path)
    
def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = build_feature_frame(df)
    train, test = split_train_test(df, TEST_START)
    
    train = prepare_categorical(train)
    test = prepare_categorical(test)
    
    x_train = train[FEATURE_COLUMNS]
    y_train = train["sales"]
    x_test = test[FEATURE_COLUMNS]
    y_test = test["sales"]
    
    model = train_xgboost(x_train, y_train, x_test, y_test)
    save_model(model, FEATURE_COLUMNS, TEST_START, ARTIFACTS_PATH / "model.pkl")
    
    # --- Baseline ---
    test["naive_pred"] = naive_baseline_predictions(test)
    maeB, rmseB = evaluate(test["sales"], test["naive_pred"], "Baseline")
    maeXG, rmseXG = evaluate(test["sales"], model.predict(x_test), "XGBoost")
    
    with open(ARTIFACTS_PATH / "metrics.json", "w") as metrics_file:
        json.dump(
            {
                "baseline": {"mae": maeB, "rmse": rmseB},
                "xgboost": {"mae": maeXG, "rmse": rmseXG},
            },
            metrics_file,
        )
    
if __name__ == "__main__":
    main()