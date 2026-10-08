import numpy as np
import pandas as pd

from features import build_feature_frame


def build_prediction_row(history_df, store, item, prediction_date):
    """
    history_df: últimos 365 dias reais de (store, item), colunas date/store/item/sales
    Retorna um dataframe de 1 linha com as features prontas para o modelo.
    """
    new_row = pd.DataFrame([{
        "date": prediction_date,
        "store": store,
        "item": item,
        "sales": np.nan, 
    }])
    
    new_row["date"] = pd.to_datetime(new_row["date"])

    combined = pd.concat([history_df, new_row], ignore_index=True)
    combined = build_feature_frame(combined)

    return combined.tail(1)