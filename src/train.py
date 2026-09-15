import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from pathlib import Path
import xgboost as xgb
import joblib

DATA_PATH = Path(__file__).parent.parent / "data" / "train.csv"

LAGS = [7, 14, 365]


def add_calendar_features(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)  # sábado é 5, domingo é 6
    return df


def add_lag_features(df):
    df = df.sort_values(["store", "item", "date"]).copy()

    grouped = df.groupby(["store", "item"])["sales"]

    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    return df


def add_rolling_features(df):
    df = df.sort_values(["store", "item", "date"]).copy()

    # Passo 1: pegar a série de vendas "atrasada em 1 dia" -- ou seja,
    # o valor que aparece na posição de hoje é, na verdade, o de ontem.
    grouped = df.groupby(["store", "item"])["sales"]
    shifted = grouped.shift(1)

    # Passo 2: agora aplicar rolling EM CIMA da série já deslocada,
    # ainda respeitando os grupos de loja-item
    for window in [7, 14]:
        df[f"rolling_mean_{window}"] = (
            shifted.groupby([df["store"], df["item"]])
            .rolling(window)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

    return df


def build_feature_frame(df):
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Uma linha de código que remove as linhas com NaN nas colunas de feature (dica: dropna tem um parâmetro subset).
    df = df.dropna(subset=[col for col in df.columns if col.startswith(("lag_", "rolling_"))])
    return df

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
    save_model(model, FEATURE_COLUMNS, TEST_START, "model.pkl")
    
    # --- Baseline ---
    test["naive_pred"] = naive_baseline_predictions(test)
    evaluate(test["sales"], test["naive_pred"], "Baseline")
    evaluate(test["sales"], model.predict(x_test), "XGBoost")
    
if __name__ == "__main__":
    main()