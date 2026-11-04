"""Plot training / validation losses from the CSV logs written by the trainer.

    python scripts/plot_losses.py English
    python scripts/plot_losses.py Telugu

Writes Results/<Lang>/training_loss.png, validation_loss.png, train_vs_val.png
and loss_summary.json (numbers quoted in the report come from this file).
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else None


def ema(x, a=0.9):
    out, m = np.empty_like(x), x[0]
    for i, v in enumerate(x):
        m = a * m + (1 - a) * v
        out[i] = m
    return out


def panel(ax_grid, data, title_prefix, smooth):
    for ax, n in zip(ax_grid.flat, NAMES):
        y = data[n]
        if smooth:
            ax.plot(data["step"], y, color="0.8", lw=0.6, label="raw")
            ax.plot(data["step"], ema(y), lw=1.4, label="EMA 0.9")
        else:
            ax.plot(data["step"], y, "o-", ms=2.5, lw=1.2)
        ax.set_title("{} {} loss".format(title_prefix, n), fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)


def main(lang, out=None):
    out = out or os.path.join(ROOT, "Results", lang)
    log = os.path.join(out, "logs")
    tr = read(os.path.join(log, "train_log.csv"))
    va = read(os.path.join(log, "val_log.csv"))

    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    panel(axs, tr, "train", smooth=True)
    axs.flat[0].legend(fontsize=7)
    fig.suptitle("{}: training losses (logged every 100 steps)".format(lang))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_loss.png"), dpi=150)
    plt.close(fig)

    summary = {"last_train_step": int(tr["step"][-1])}
    if va is not None:
        fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
        panel(axs, va, "val", smooth=False)
        fig.suptitle("{}: validation losses (full validation set)".format(lang))
        fig.tight_layout()
        fig.savefig(os.path.join(out, "validation_loss.png"), dpi=150)
        plt.close(fig)

        fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
        for ax, n in zip(axs.flat, NAMES):
            # average the train loss over each validation interval for a fair overlay
            tv = [tr[n][(tr["step"] > s - (va["step"][1] - va["step"][0] if len(va["step"]) > 1 else 1000)) & (tr["step"] <= s)].mean() for s in va["step"]]
            ax.plot(va["step"], tv, label="train (mean over interval)")
            ax.plot(va["step"], va[n], label="validation")
            ax.set_title(n, fontsize=9)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
        axs.flat[0].legend(fontsize=7)
        fig.suptitle("{}: train vs validation".format(lang))
        fig.tight_layout()
        fig.savefig(os.path.join(out, "train_vs_val.png"), dpi=150)
        plt.close(fig)

        i = int(np.argmin(va["total"]))
        summary.update({
            "val_points": len(va["step"]),
            "best_val_step": int(va["step"][i]),
            "best_val": {n: round(float(va[n][i]), 4) for n in NAMES},
            "final_val_step": int(va["step"][-1]),
            "final_val": {n: round(float(va[n][-1]), 4) for n in NAMES},
        })
    last = tr["step"] > tr["step"][-1] - 1000
    summary["final_train_mean_last_1k"] = {n: round(float(tr[n][last].mean()), 4) for n in NAMES}
    if "steps_per_sec" in tr:
        summary["median_steps_per_sec"] = round(float(np.median(tr["steps_per_sec"])), 2)
        summary["train_wallclock_h_last_session"] = round(float(tr["elapsed_h"][-1]), 2)
    with open(os.path.join(out, "loss_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # optional 2nd arg: results dir containing logs/ (default Results/<Lang>)
    main(sys.argv[1] if len(sys.argv) > 1 else "English", sys.argv[2] if len(sys.argv) > 2 else None)
