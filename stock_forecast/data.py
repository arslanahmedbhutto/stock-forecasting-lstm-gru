"""Loading, features, chronological split and leak-free windows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from stock_forecast import DATA_PATH

FEATURES = ["ret", "hl_range", "vol_chg"]
LOOKBACK = 60
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def load_prices(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stationary features. Row t only uses prices up to and including day t.

    ret       log(Close_t / Close_{t-1})   <- also the prediction target
    hl_range  log(High_t / Low_t)          intraday volatility
    vol_chg   log(Volume_t / Volume_{t-1})
    """
    out = df.copy()
    out["ret"] = np.log(out["Close"]).diff()
    out["hl_range"] = np.log(out["High"] / out["Low"])
    out["vol_chg"] = np.log(out["Volume"].clip(lower=1)).diff()
    return out.dropna().reset_index(drop=True)


@dataclass
class Split:
    """Windows for one split. Target y[k] is the return on day idx[k];
    X[k] holds features from the LOOKBACK days before it (never day idx[k] itself)."""

    X: np.ndarray          # (n, LOOKBACK, n_features), scaled
    y: np.ndarray          # (n,) raw log returns
    idx: np.ndarray        # row index of the target day
    prev_close: np.ndarray  # close on the day before the target
    close: np.ndarray       # actual close on the target day
    dates: np.ndarray


@dataclass
class Dataset:
    train: Split
    val: Split
    test: Split
    scaler: StandardScaler
    frame: pd.DataFrame
    boundaries: tuple[int, int]


def split_bounds(n: int) -> tuple[int, int]:
    return int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))


def make_dataset(df: pd.DataFrame, lookback: int = LOOKBACK) -> Dataset:
    frame = add_features(df)
    n = len(frame)
    train_end, val_end = split_bounds(n)

    # Fit scaling on training rows only
    scaler = StandardScaler().fit(frame.loc[: train_end - 1, FEATURES])
    feats = scaler.transform(frame[FEATURES]).astype("float32")
    ret = frame["ret"].to_numpy("float32")
    close = frame["Close"].to_numpy("float64")
    dates = frame["Date"].to_numpy()

    def windows(start: int, stop: int) -> Split:
        idx = np.arange(max(start, lookback), stop)
        X = np.stack([feats[i - lookback:i] for i in idx])
        return Split(X, ret[idx], idx, close[idx - 1], close[idx], dates[idx])

    return Dataset(windows(0, train_end), windows(train_end, val_end), windows(val_end, n),
                   scaler, frame, (train_end, val_end))
