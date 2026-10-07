"""Evaluation of next-day return forecasts, in return and price space."""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import r2_score


def evaluate(pred_ret: np.ndarray, split, up_rate: float | None = None) -> dict:
    """Metrics for one-step-ahead forecasts.

    Price forecast = previous close * exp(predicted log return).
    `skill` is the % reduction in squared error vs the naive forecast
    (tomorrow's price = today's), which predicts a return of 0.
    """
    y = split.y.astype("float64")
    pred_ret = np.asarray(pred_ret, dtype="float64")
    pred_price = split.prev_close * np.exp(pred_ret)
    err = split.close - pred_price

    mse = np.mean((y - pred_ret) ** 2)
    mse_naive = np.mean(y ** 2)

    moved = y != 0
    out = {
        "Return RMSE (%)": float(100 * np.sqrt(mse)),
        "Skill vs naive (%)": float(100 * (1 - mse / mse_naive)),
        "Price RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Price MAE": float(np.mean(np.abs(err))),
        "Price MAPE (%)": float(100 * np.mean(np.abs(err) / split.close)),
        "Price R2": float(r2_score(split.close, pred_price)),
    }
    if np.any(pred_ret != 0):
        hits = int(np.sum(np.sign(pred_ret[moved]) == np.sign(y[moved])))
        n = int(moved.sum())
        out["Directional Acc (%)"] = 100 * hits / n
        # Is the hit rate better than always predicting "up"?
        if up_rate is not None:
            out["p-value vs always-up"] = float(binomtest(hits, n, up_rate, alternative="greater").pvalue)
    return out
