"""Train every model, evaluate on the test period, write reports and figures.

    python -m stock_forecast.train            # full run (~15 min on CPU)
    python -m stock_forecast.train --seeds 1  # quicker
"""

import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from stock_forecast import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED
from stock_forecast.data import FEATURES, LOOKBACK, load_prices, make_dataset
from stock_forecast.metrics import evaluate
from stock_forecast.models import Drift, LinearAR, Naive, RNNForecaster
from stock_forecast.original import OriginalPriceGRU
from stock_forecast import plots

ORIGINAL = "Original: GRU on raw prices"
SUMMARY_COLS = ["Skill vs naive (%)", "Directional Acc (%)", "Return RMSE (%)", "Price RMSE",
                "Price MAPE (%)", "Price R2"]


def up_rate(split) -> float:
    moved = split.y != 0
    return float(np.mean(split.y[moved] > 0))


def main(n_seeds: int):
    ds = make_dataset(load_prices())
    f = ds.frame
    print(f"{len(f):,} trading days | train {len(ds.train.y):,} | val {len(ds.val.y):,} | test {len(ds.test.y):,}")
    for name, s in [("train", ds.train), ("val", ds.val), ("test", ds.test)]:
        print(f"  {name:<5} {pd.Timestamp(s.dates[0]).date()} -> {pd.Timestamp(s.dates[-1]).date()}"
              f"   up days {up_rate(s):.1%}")

    rows, preds, per_seed = [], {}, []

    def record(name, family, model, seed=None):
        r = {"Model": name, "Family": family,
             **{f"val {k}": v for k, v in evaluate(model.predict(ds.val), ds.val, up_rate(ds.val)).items()},
             **evaluate(model.predict(ds.test), ds.test, up_rate(ds.test))}
        if seed is not None:
            r["seed"] = seed
        return r

    for name, model in [("Naive (tomorrow = today)", Naive()), ("Drift (average return)", Drift()),
                        ("Linear AR (ridge)", LinearAR())]:
        model.fit(ds)
        rows.append(record(name, "Baseline", model))
        preds[name] = model.predict(ds.test)
        if isinstance(model, LinearAR):
            ridge = model
    print(f"  ridge alpha chosen on val: {ridge.alpha}")

    best = {}
    for cell in ["LSTM", "GRU"]:
        for s in range(n_seeds):
            t0 = time.time()
            m = RNNForecaster(cell, seed=SEED + s).fit(ds)
            r = record(cell, "Recurrent", m, seed=SEED + s)
            per_seed.append(r)
            print(f"  {cell} seed {SEED + s}: test skill {r['Skill vs naive (%)']:+.2f}%  "
                  f"dir acc {r['Directional Acc (%)']:.1f}%  ({time.time() - t0:.0f}s)")
            if cell not in best or m.best_val_loss < best[cell].best_val_loss:
                best[cell] = m
        seed_rows = [r for r in per_seed if r["Model"] == cell]
        agg = {"Model": cell, "Family": "Recurrent", "seeds": n_seeds}
        for k, v in seed_rows[0].items():
            if isinstance(v, float):
                vals = [r[k] for r in seed_rows]
                agg[k], agg[f"{k} std"] = float(np.mean(vals)), float(np.std(vals))
        rows.append(agg)
        preds[cell] = best[cell].predict(ds.test)

    t0 = time.time()
    orig = OriginalPriceGRU(seed=SEED).fit(ds)
    rows.append(record(ORIGINAL, "Original", orig))
    print(f"  {ORIGINAL}: test price R2 {rows[-1]['Price R2']:.2f}  ({time.time() - t0:.0f}s)")

    # ---------- save models ----------
    MODELS_DIR.mkdir(exist_ok=True)
    for cell, m in best.items():
        m.model.save(MODELS_DIR / f"{cell.lower()}.keras")
    joblib.dump({"scaler": ds.scaler, "features": FEATURES, "lookback": LOOKBACK, "ridge": ridge.model,
                 "target_mean_std": {c: (m.mu, m.sd) for c, m in best.items()}},
                MODELS_DIR / "preprocessing_and_ridge.joblib")

    # ---------- reports ----------
    write_reports(ds, rows, per_seed, n_seeds)
    write_figures(ds, rows, preds, orig)


def fmt(r, k):
    v = r.get(k)
    if v is None:
        return "—"
    d = 4 if "R2" in k else 3 if "Return" in k else 2
    s = f"{v:.{d}f}" if abs(v) < 1000 else f"{v:,.0f}"
    if f"{k} std" in r:
        s += f" ± {r[f'{k} std']:.{d}f}"
    return s


def write_reports(ds, rows, per_seed, n_seeds):
    REPORTS_DIR.mkdir(exist_ok=True)
    meta = {
        "data": "AAPL daily OHLCV, Yahoo Finance export",
        "periods": {n: [str(pd.Timestamp(s.dates[0]).date()), str(pd.Timestamp(s.dates[-1]).date())]
                    for n, s in [("train", ds.train), ("val", ds.val), ("test", ds.test)]},
        "up_day_rate_test": up_rate(ds.test),
        "lookback": LOOKBACK, "features": FEATURES, "seeds": n_seeds,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps({**meta, "results": rows, "per_seed": per_seed}, indent=2), encoding="utf-8")

    head = "| Model | " + " | ".join(SUMMARY_COLS) + " |\n|" + "---|" * (len(SUMMARY_COLS) + 1) + "\n"
    body = "".join("| " + r["Model"] + " | " + " | ".join(fmt(r, k) for k in SUMMARY_COLS) + " |\n" for r in rows)
    (REPORTS_DIR / "model_comparison.md").write_text(
        f"# Test-period results ({meta['periods']['test'][0]} to {meta['periods']['test'][1]})\n\n"
        f"Recurrent rows are mean ± std over {n_seeds} seeds. "
        f"Always predicting 'up' would score {100 * meta['up_day_rate_test']:.1f}% directional accuracy.\n\n"
        + head + body, encoding="utf-8"
    )


def write_figures(ds, rows, preds, orig):
    plots.original_failure(ds, orig.predict_price(ds.test), orig.train_max, FIGURES_DIR / "original_failure.png")

    fair = [r for r in rows if r["Family"] != "Original" and not r["Model"].startswith("Naive")]
    plots.hbar({r["Model"]: r["Skill vs naive (%)"] for r in fair},
               "Error reduction vs the naive forecast (test period)",
               "Skill vs naive (%) — positive = better than 'tomorrow = today'",
               FIGURES_DIR / "skill_vs_naive.png", highlight={"LSTM", "GRU"},
               errors={r["Model"]: r.get("Skill vs naive (%) std", 0) for r in fair})
    plots.hbar({r["Model"]: r["Directional Acc (%)"] for r in fair if "Directional Acc (%)" in r},
               "Directional accuracy (test period)", "Days where the predicted direction was right (%)",
               FIGURES_DIR / "directional_accuracy.png", ref=100 * up_rate(ds.test), xmin=40, ref_label="always 'up'", fmt="{:.1f}%", highlight={"LSTM", "GRU"},
               errors={r["Model"]: r.get("Directional Acc (%) std", 0) for r in fair})
    plots.returns_vs_predictions(ds.test, {"GRU": preds["GRU"], "Linear AR": preds["Linear AR (ridge)"]},
                                 FIGURES_DIR / "returns_vs_predictions.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    main(ap.parse_args().seeds)
