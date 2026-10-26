"""Pacing gains of the target, classical, quantum and classical twin heads on one Tokyo
trace, drawn as plain lines.

Same trace and window as generate_pacing_gain_timeseries_seed.py
(downlink-sequential-logs, trace 10, pool_index 3014-3313,
multiseed_head_comparison_v1). The x-axis is the interval end time t_i (s)
from the Tokyo pool.

Run: python3 reports/generate_pacing_gain_lines_seed.py [seed]
"""
import csv
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS_ROOT = ROOT / "data/processed/evaluation/multiseed_head_comparison_v1"
POOL_PATH = ROOT / "data/processed/evaluation/tokyo_8model_comparison_v1/tokyo.pkl"
FIG_DIR = ROOT / "reports" / "figures" / "paper"

STREAM_GROUP = "downlink-sequential-logs"
POOL_START, POOL_END = 3014, 3313
T_MAX = 100.0

CLASSICAL_COLOR = "#0072B2"
QUANTUM_COLOR = "#D55E00"
TARGET_COLOR = "#000000"
TWIN_COLOR = "#009E73"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Times", "serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def load_window(seed, model_key):
    path = RESULTS_ROOT / f"seed{seed}" / model_key / "predictions.csv"
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r["stream_group"] == STREAM_GROUP
                and POOL_START <= int(r["pool_index"]) <= POOL_END]
    rows.sort(key=lambda r: int(r["pool_index"]))
    return rows


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 400003
    with open(POOL_PATH, "rb") as f:
        pool = pickle.load(f)
    c = load_window(seed, "lfm2_5_350m_classical")
    q = load_window(seed, "lfm2_5_350m_quantum")
    tw = load_window(seed, "lfm2_5_350m_classical_twin")
    assert [r["sample_id"] for r in c] == [r["sample_id"] for r in q]
    assert [r["sample_id"] for r in c] == [r["sample_id"] for r in tw]
    assert all(pool.sample_ids[int(r["pool_index"])] == r["sample_id"] for r in c)

    t_all = [float(pool.states[int(r["pool_index"])][2]) for r in c]
    keep = [i for i, t in enumerate(t_all) if t <= T_MAX]
    t = [t_all[i] for i in keep]

    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    ax.plot(t, [float(c[i]["target_gain"]) for i in keep], color=TARGET_COLOR,
            lw=0.9, ls="--", label="Heuristic target", zorder=3)
    ax.plot(t, [float(c[i]["predicted_gain"]) for i in keep], color=CLASSICAL_COLOR,
            lw=0.9, label="Linear head", alpha=0.9, zorder=4)
    ax.plot(t, [float(q[i]["predicted_gain"]) for i in keep], color=QUANTUM_COLOR,
            lw=0.9, label="VQC head", alpha=0.85, zorder=5)
    ax.plot(t, [float(tw[i]["predicted_gain"]) for i in keep], color=TWIN_COLOR,
            lw=1.2, ls=":", label="Classical twin", zorder=6)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0.85, 1.30)
    ax.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Predicted gain (offline)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, handletextpad=0.3, columnspacing=1.0,
              borderaxespad=0.0, handlelength=1.6)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(FIG_DIR / f"pacing_gain_lines_seed{seed}.{ext}",
                    bbox_inches="tight", **kw)
    print("written:", FIG_DIR / f"pacing_gain_lines_seed{seed}.pdf")


if __name__ == "__main__":
    main()
