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