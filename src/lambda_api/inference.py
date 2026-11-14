import numpy as np
import pandas as pd

from features import HISTORY_DAYS, build_feature_frame


def build_prediction_row(history_df, store, item, prediction_date):
    """
    history_df: últimos 365 dias reais de (store, item), colunas date/store/item/sales
    Retorna um dataframe de 1 linha com as features prontas para o modelo.
    """
    new_row = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(prediction_date),
                "store": store,
                "item": item,
                "sales": np.nan,
            }
        ]
    )

    combined = pd.concat([history_df, new_row], ignore_index=True)
    combined = build_feature_frame(combined)
    combined["date"] = pd.to_datetime(combined["date"])
    
    return combined.tail(1)


def split_valid_pairs(history_df, pairs, simulated_date, history_days=HISTORY_DAYS):
    """
    Separa as séries pedidas em válidas e inválidas, ANTES de calcular features.

    Uma série é válida quando tem exatamente `history_days` dias consecutivos terminando em
    `simulated_date`. Como (store, item, sale_date) é chave primária, contagem correta mais as duas
    pontas corretas garantem que não falta nenhum dia no meio.

    Retorna (pares_validos, erros), com erros = [(store, item, reason)].
    """
    last_day = pd.Timestamp(simulated_date)
    first_day = last_day - pd.Timedelta(days=history_days - 1)

    if history_df.empty:
        stats = {}
    else:
        stats = (
            history_df.groupby(["store", "item"])["date"]
            .agg(["count", "min", "max"])
            .to_dict("index")
        )

    valid, errors = [], []
    for store, item in pairs:
        s = stats.get((store, item))
        if s is None:
            errors.append((store, item, "empty_history"))
        elif (
            s["count"] == history_days
            and s["min"] == first_day
            and s["max"] == last_day
        ):
            valid.append((store, item))
        else:
            errors.append((store, item, "incomplete_history"))
    return valid, errors


def build_prediction_rows(history_df, pairs, prediction_date):
    """
    Versão em lote de build_prediction_row, para séries já validadas por split_valid_pairs.
    Retorna um DataFrame com uma linha por série, só com as linhas do dia previsto.
    """
    columns = ["date", "store", "item", "sales"]
    if not pairs:
        return pd.DataFrame(columns=columns)

    target = pd.Timestamp(prediction_date)

    # Só entram as séries válidas; o histórico de séries inválidas fica de fora.
    keys = pd.MultiIndex.from_frame(history_df[["store", "item"]])
    history = history_df[keys.isin(pairs)]

    new_rows = pd.DataFrame(
        {
            "date": target,
            "store": [store for store, _ in pairs],
            "item": [item for _, item in pairs],
            "sales": np.nan,
        }
    )

    combined = pd.concat([history, new_rows], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = build_feature_frame(combined)

    return combined[combined["date"] == target].reset_index(drop=True)
