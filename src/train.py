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

def prepare_categorical(df, store_categories, item_categories):
    df = df.copy()
    df["store"] = df["store"].astype(pd.CategoricalDtype(categories=store_categories))
    df["item"] = df["item"].astype(pd.CategoricalDtype(categories=item_categories))
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
    
def train_xgboost_native(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    
    params = {
        "max_depth": 7,
        "eta": 0.05,               
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "seed": 42,
    }
    
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
    )
    return bst

def save_model(model, feature_columns, test_start, path, store_categories, item_categories):
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "test_start": test_start,
        "categories": {"store": store_categories, "item": item_categories},
    }
    joblib.dump(bundle, path)
    
def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = build_feature_frame(df)
    
    store_categories = sorted(df["store"].unique())
    item_categories = sorted(df["item"].unique())
    
    train, test = split_train_test(df, TEST_START)
    
    train = prepare_categorical(train, store_categories, item_categories)
    test = prepare_categorical(test, store_categories, item_categories)
    
    x_train = train[FEATURE_COLUMNS]
    y_train = train["sales"]
    x_test = test[FEATURE_COLUMNS]
    
    model = train_xgboost_native(x_train, y_train)
    save_model(model, FEATURE_COLUMNS, TEST_START, ARTIFACTS_PATH / "model.pkl", store_categories, item_categories)
    
    # --- Baseline ---
    test["naive_pred"] = naive_baseline_predictions(test)
    maeB, rmseB = evaluate(test["sales"], test["naive_pred"], "Baseline")
    maeXG, rmseXG = evaluate(test["sales"], model.predict(xgb.DMatrix(x_test, enable_categorical=True)), "XGBoost")
    
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