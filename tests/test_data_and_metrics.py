import numpy as np
import pandas as pd
import pytest

from stock_forecast.data import FEATURES, LOOKBACK, add_features, load_prices, make_dataset
from stock_forecast.metrics import evaluate
from stock_forecast.models import Drift, LinearAR, Naive


@pytest.fixture(scope="module")
def ds():
    return make_dataset(load_prices())


def synthetic_prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 10 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    return pd.DataFrame({
        "Date": pd.bdate_range("2000-01-03", periods=n),
        "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Adj Close": close, "Volume": rng.integers(1_000, 10_000, n),
    })


def test_dataset_loads_full_history():
    df = load_prices()
    assert len(df) == 9909
    assert df["Date"].is_monotonic_increasing
    assert df["Date"].iloc[0] == pd.Timestamp("1980-12-12")


def test_splits_are_chronological_and_disjoint(ds):
    assert ds.train.idx.max() < ds.val.idx.min()
    assert ds.val.idx.max() < ds.test.idx.min()
    assert ds.test.idx.max() == len(ds.frame) - 1
    assert ds.train.dates.max() < ds.val.dates.min() < ds.test.dates.min()


def test_windows_contain_only_past_data():
    """Changing the target day's (or any later) data must not change its input window."""
    df = synthetic_prices()
    a = make_dataset(df)
    k = 5
    target_row = a.test.idx[k]
    df2 = df.copy()
    # add_features drops the first row, so frame row r = raw row r + 1
    df2.loc[target_row + 1:, ["Close", "High", "Low"]] *= 3
    b = make_dataset(df2)
    np.testing.assert_array_equal(a.test.X[k], b.test.X[k])
    assert a.test.y[k] != b.test.y[k]


def test_window_shape_and_target_alignment(ds):
    assert ds.test.X.shape[1:] == (LOOKBACK, len(FEATURES))
    ret = ds.frame["ret"].to_numpy("float32")
    np.testing.assert_allclose(ds.test.y, ret[ds.test.idx])
    np.testing.assert_allclose(np.log(ds.test.close / ds.test.prev_close), ds.test.y, atol=1e-5)


def test_scaler_fitted_on_train_only(ds):
    train_rows = ds.frame.loc[: ds.boundaries[0] - 1, FEATURES]
    np.testing.assert_allclose(ds.scaler.mean_, train_rows.mean().to_numpy(), rtol=1e-6)


def test_features_are_finite():
    f = add_features(load_prices())
    assert np.isfinite(f[FEATURES].to_numpy()).all()


def test_naive_has_zero_skill_and_perfect_reconstruction(ds):
    m = evaluate(Naive().predict(ds.test), ds.test)
    assert m["Skill vs naive (%)"] == pytest.approx(0)
    assert "Directional Acc (%)" not in m  # a flat forecast has no direction
    exact = evaluate(ds.test.y, ds.test)
    assert exact["Price RMSE"] == pytest.approx(0, abs=1e-3)
    assert exact["Skill vs naive (%)"] == pytest.approx(100)


def test_drift_direction_equals_up_day_rate(ds):
    m = evaluate(Drift().fit(ds).predict(ds.test), ds.test)
    moved = ds.test.y != 0
    assert m["Directional Acc (%)"] == pytest.approx(100 * np.mean(ds.test.y[moved] > 0))


def test_linear_ar_learns_a_real_signal():
    """On synthetic AR(1) returns, the ridge model should clearly beat the naive forecast."""
    rng = np.random.default_rng(1)
    n, r = 1500, np.zeros(1500)
    for t in range(1, n):
        r[t] = 0.6 * r[t - 1] + rng.normal(0, 0.01)
    close = 50 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"Date": pd.bdate_range("2000-01-03", periods=n), "Open": close,
                       "High": close * 1.01, "Low": close * 0.99, "Close": close,
                       "Adj Close": close, "Volume": np.full(n, 5000)})
    syn = make_dataset(df)
    m = evaluate(LinearAR().fit(syn).predict(syn.test), syn.test)
    assert m["Skill vs naive (%)"] > 20
