"""Testes das features e da validação de séries. Rodam sem banco, sem rede e sem o modelo."""
import numpy as np
import pandas as pd
import pytest
from inference import build_prediction_row, build_prediction_rows, split_valid_pairs

from features import HISTORY_DAYS, LAGS, build_feature_frame

START = pd.Timestamp("2016-01-01")
DAYS = 400  # mais que os 365 dias do maior lag


def make_series(store, item, seed, days=DAYS):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "date": pd.date_range(START, periods=days),
            "store": store,
            "item": item,
            "sales": rng.integers(1, 100, days).astype(float),
        }
    )


@pytest.fixture
def two_series():
    return pd.concat([make_series(1, 1, 0), make_series(2, 7, 1)], ignore_index=True)


def lookup_sales(frame, feat, shift_days):
    """Venda da MESMA série em `data - shift_days`, alinhada às linhas de `feat`."""
    s = frame.set_index(["store", "item", "date"])["sales"]
    idx = pd.MultiIndex.from_arrays(
        [feat["store"], feat["item"], feat["date"] - pd.Timedelta(days=shift_days)]
    )
    return s.reindex(idx).to_numpy()


def test_lags_use_the_same_series_only(two_series):
    feat = build_feature_frame(two_series)
    assert not feat.empty
    for lag in LAGS:
        np.testing.assert_allclose(feat[f"lag_{lag}"].to_numpy(), lookup_sales(two_series, feat, lag))


@pytest.mark.parametrize("window", [7, 14])
def test_rolling_mean_excludes_the_predicted_day(two_series, window):
    feat = build_feature_frame(two_series)
    past = np.vstack([lookup_sales(two_series, feat, k) for k in range(1, window + 1)])
    np.testing.assert_allclose(feat[f"rolling_mean_{window}"].to_numpy(), past.mean(axis=0))


def feature_cols(df):
    return [c for c in df.columns if c.startswith(("lag_", "rolling_"))] + [
        "day_of_week", "month", "year", "is_weekend",
    ]


def history_before(series, target):
    start = target - pd.Timedelta(days=HISTORY_DAYS)
    return series[(series["date"] >= start) & (series["date"] < target)]


def test_api_row_matches_training_row(two_series):
    """A linha montada na API (sem a venda do dia) é igual à do pipeline de treino."""
    one = two_series[two_series["store"] == 1]
    target = one["date"].iloc[380]
    train_row = build_feature_frame(one).query("date == @target")
    api_row = build_prediction_row(history_before(one, target), 1, 1, target)
    cols = feature_cols(train_row)
    np.testing.assert_allclose(
        api_row[cols].to_numpy(dtype=float), train_row[cols].to_numpy(dtype=float)
    )


def test_batch_matches_single(two_series):
    target = two_series["date"].iloc[380]
    history = pd.concat(
        [history_before(two_series[two_series["store"] == s], target) for s in (1, 2)],
        ignore_index=True,
    )
    pairs = [(1, 1), (2, 7)]
    batch = build_prediction_rows(history, pairs, target).set_index(["store", "item"])
    for store, item in pairs:
        h = history[(history["store"] == store) & (history["item"] == item)]
        single = build_prediction_row(h, store, item, target).iloc[0]
        cols = feature_cols(batch)
        np.testing.assert_allclose(
            batch.loc[(store, item), cols].to_numpy(dtype=float), single[cols].to_numpy(dtype=float)
        )


def test_split_valid_pairs_flags_empty_and_incomplete():
    last = START + pd.Timedelta(days=HISTORY_DAYS - 1)
    full = make_series(1, 1, 0, days=HISTORY_DAYS)
    gap = make_series(1, 2, 1, days=HISTORY_DAYS).drop(index=100)  # falta um dia no meio
    history = pd.concat([full, gap], ignore_index=True)

    valid, errors = split_valid_pairs(history, [(1, 1), (1, 2), (1, 3)], last)

    assert valid == [(1, 1)]
    assert errors == [(1, 2, "incomplete_history"), (1, 3, "empty_history")]
