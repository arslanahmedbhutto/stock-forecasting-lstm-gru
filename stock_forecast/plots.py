"""Report figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PERIOD_FILL = {"Train": "#f0efec", "Validation": "#e8f0fb", "Test": "#fdece4"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def original_failure(ds, orig_pred_price, train_max, path):
    """Close price with split periods, the training-range ceiling, and the
    original model's test predictions."""
    f = ds.frame
    t_end, v_end = ds.boundaries
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for name, a, b in [("Train", 0, t_end), ("Validation", t_end, v_end), ("Test", v_end, len(f) - 1)]:
        ax.axvspan(f["Date"].iloc[a], f["Date"].iloc[b], color=PERIOD_FILL[name], lw=0)
        ax.text(f["Date"].iloc[(a + b) // 2], 335, name, ha="center", color=INK_2, fontsize=9.5)
    ax.plot(f["Date"], f["Close"], color=INK, lw=1.1, label="Actual close")
    ax.plot(ds.test.dates, orig_pred_price, color=ORANGE, lw=2, label="Original model's test prediction")
    ax.axhline(train_max, color=BLUE, lw=1.2, ls="--")
    ax.text(f["Date"].iloc[200], train_max + 7, f"Highest price seen in training: ${train_max:.2f}",
            color=BLUE, fontsize=9.5)
    ax.set_ylim(0, 350)
    ax.set_ylabel("Close price ($)")
    ax.set_title("Why the original model failed: test prices are far outside the training range")
    ax.legend(frameon=False, loc="center left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, path)


def hbar(values: dict[str, float], title: str, xlabel: str, path: Path, ref: float | None = None,
         ref_label: str | None = None, fmt: str = "{:+.2f}", highlight: set[str] = frozenset(),
         errors: dict[str, float] | None = None, xmin: float | None = None):
    names = list(values)
    vals = [values[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.3))
    y = np.arange(len(names))[::-1]
    colors = [BLUE if n in highlight else MUTED for n in names]
    xerr = [errors.get(n, 0) for n in names] if errors else None
    left = xmin or 0
    ax.barh(y, [v - left for v in vals], left=left, color=colors, height=0.6, xerr=xerr,
            error_kw={"ecolor": INK_2, "lw": 1, "capsize": 3})
    ax.set_yticks(y, names)
    lo, hi = ax.get_xlim()
    pad = (hi - lo) * 0.015
    for yi, v, e in zip(y, vals, xerr or [0] * len(vals)):
        x = v + e + pad if v >= 0 else v - e - pad
        ax.text(x, yi, fmt.format(v), va="center", ha="left" if v >= 0 else "right", fontsize=9.5, color=INK)
    if xmin is None:
        ax.axvline(0, color=INK_2, lw=0.8)
    if ref is not None:
        ax.axvline(ref, color=ORANGE, lw=1.5, ls="--")
        ax.text(ref, len(names) - 0.4, f" {ref_label}", color=ORANGE, fontsize=9, va="bottom")
    ax.set_xlim(xmin if xmin is not None else lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.12)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=14)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _save(fig, path)


def returns_vs_predictions(split, preds: dict[str, np.ndarray], path: Path, last: int = 250):
    """Actual daily returns vs model predictions over the last `last` test days."""
    d = split.dates[-last:]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(d, split.y[-last:] * 100, color=MUTED, width=1.0, label="Actual daily return")
    for (name, p), c in zip(preds.items(), [BLUE, ORANGE]):
        ax.plot(d, p[-last:] * 100, color=c, lw=1.6, label=f"{name} prediction")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("Daily return (%)")
    ax.set_title("Predicted vs actual daily returns (last 250 test days)")
    ax.legend(frameon=False, loc="lower left", ncols=3)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _save(fig, path)
