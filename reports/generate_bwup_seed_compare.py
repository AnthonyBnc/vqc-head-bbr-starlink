"""Figs. 5-7: BW_UP decisions per location, seeds 100003 and 400003.

Same quantities as evaluate_seen_locations.py `plot` (predicted gains, mean
gain difference per trace, surrogate retransmissions per trace relative to the
heuristic target), redrawn with the paper's 2x3 layout:
  - "Expert" renamed to "Target" everywhere (the label is a heuristic target),
  - the LFM2.5 linear head is labelled "LFM2.5 (linear)",
  - the same y-axis limits are used for each panel type across Tokyo, Mumbai
    and Sao Paulo, so the three figures can be compared directly.
The classical twin is not shown: it was evaluated on Tokyo only, not on the
seen-location validation traces.

Run from the repo root: python3 reports/generate_bwup_seed_compare.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_seen_locations as esl  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SEEDS = [100003, 400003]
LOCATIONS = [("Tokyo", "tokyo", "Tokyo (unseen location)"),
             ("Mumbai", "mumbai", "Mumbai (seen location)"),
             ("SaoPaulo", "saopaulo", "São Paulo (seen location)")]
ORDER = ["granite_4_0_350m_classical", "pleias_rag_350m_classical",
         "lfm2_5_350m_classical", "gemma_3_270m_classical", "lfm2_5_350m_quantum"]
NAME = {"target": "Target", "granite_4_0_350m_classical": "Granite",
        "pleias_rag_350m_classical": "Pleias", "lfm2_5_350m_classical": "LFM2.5\n(linear)",
        "gemma_3_270m_classical": "Gemma3", "lfm2_5_350m_quantum": "LFM2.5\n+VQC"}
LEGEND = {"target": "Target", "granite_4_0_350m_classical": "Granite",
          "pleias_rag_350m_classical": "Pleias", "lfm2_5_350m_classical": "LFM2.5 (linear)",
          "gemma_3_270m_classical": "Gemma3", "lfm2_5_350m_quantum": "LFM2.5+VQC"}
COLOR = {"target": "#5A5A5A", "granite_4_0_350m_classical": "#4C9A9A",
         "pleias_rag_350m_classical": "#C9504C", "lfm2_5_350m_classical": "#E0A23C",
         "gemma_3_270m_classical": "#5E9E4C", "lfm2_5_350m_quantum": "#7E78C2"}
FIG_DIR = ROOT / "reports" / "figures" / "paper"

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def collect():
    """stats[(seed, loc)] = dict of per-model arrays for the three panels."""
    import os
    os.chdir(ROOT)
    val_retx, val_trace = esl.pool_reference(esl.VAL_POOL)
    tok_retx, tok_trace = esl.pool_reference(esl.TOKYO_POOL)
    stats = {}
    for seed in SEEDS:
        tokyo_dirs = {key: tdir for key, _, tdir in esl.models(seed)}
        for loc, _, _ in LOCATIONS:
            gains, bias, ratio = {}, {}, {}
            for i, key in enumerate(ORDER):
                if loc == "Tokyo":
                    d = pd.read_csv(tokyo_dirs[key] / "predictions.csv")
                    d["e"] = tok_retx[d.pool_index]; d["trace"] = tok_trace[d.pool_index]
                else:
                    d = pd.read_csv(esl.OUT / f"seed{seed}" / key / "predictions.csv")
                    d = d[d.location == loc]
                    d["e"] = val_retx[d.pool_index]; d["trace"] = val_trace[d.pool_index]
                u = d[d.phase == "BW_UP"]
                y = u.target_gain.round(2).to_numpy(); p = u.predicted_gain.round(2).to_numpy()
                if i == 0:
                    gains["target"] = y
                gains[key] = p
                g = pd.DataFrame({"t": u.trace.to_numpy(), "d": p - y,
                                  "r": u.surrogate_retransmissions.to_numpy(),
                                  "e": u.e.to_numpy()}).groupby("t")
                bias[key] = g["d"].mean().to_numpy()
                ratio[key] = ((g["r"].sum() + 1) / (g["e"].sum() + 1)).to_numpy()
            stats[(seed, loc)] = (gains, bias, ratio)
    return stats


def box(ax, data, keys, whis=1.5):
    bp = ax.boxplot([data[k] for k in keys], widths=0.6, patch_artist=True,
                    showfliers=False, whis=whis,
                    medianprops=dict(color="black", linewidth=1.4))
    for patch, k in zip(bp["boxes"], keys):
        patch.set_facecolor(COLOR[k]); patch.set_edgecolor("#333")
    ax.set_xticks(range(1, len(keys) + 1)); ax.set_xticklabels([NAME[k] for k in keys])
    ax.grid(axis="y", alpha=0.25)


def main():
    stats = collect()
    # shared y-limits per panel type across all locations and seeds
    allb = np.concatenate([np.concatenate(list(s[1].values())) for s in stats.values()])
    blo, bhi = np.percentile(allb, 0.5) - 0.02, np.percentile(allb, 99.5) + 0.02
    for loc, slug, title in LOCATIONS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.6),
                                 gridspec_kw={"width_ratios": [1.2, 1, 1]})
        letters = iter("abcdef")
        for r, seed in enumerate(SEEDS):
            gains, bias, ratio = stats[(seed, loc)]
            ax = axes[r][0]
            box(ax, gains, ["target"] + ORDER, whis=(0, 100))
            ax.set_ylim(1.04, 1.26); ax.set_ylabel("Pacing gain")
            ax.set_title(f"({next(letters)}) Seed {seed}: predicted vs. target gain")
            ax = axes[r][1]
            box(ax, bias, ORDER)
            ax.axhline(0, ls="--", lw=1, color="#777", zorder=0)
            ax.set_ylim(blo, bhi); ax.set_ylabel("Predicted − target")
            ax.set_title(f"({next(letters)}) Seed {seed}: gain difference per trace")
            ax = axes[r][2]
            box(ax, ratio, ORDER)
            ax.axhline(1, ls="--", lw=1, color="#777", zorder=0)
            ax.set_yscale("log"); ax.set_ylim(0.25, 12)
            ax.set_yticks([0.25, 0.5, 1, 2, 4, 8])
            ax.set_yticklabels(["0.25×", "0.5×", "1×", "2×", "4×", "8×"]); ax.minorticks_off()
            ax.set_ylabel("Ratio to target (log scale)")
            ax.set_title(f"({next(letters)}) Seed {seed}: retransmissions vs. target")
        handles = [Patch(fc=COLOR[k], ec="#333", label=LEGEND[k]) for k in ["target"] + ORDER]
        fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
                   title=f"{title}, BW_UP intervals", bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(h_pad=2.0, rect=(0, 0, 1, 0.9))
        for ext in ("pdf", "png"):
            kw = {"dpi": 220} if ext == "png" else {}
            fig.savefig(FIG_DIR / f"bwup_seed_compare_{slug}.{ext}", bbox_inches="tight", **kw)
        plt.close(fig)
        # medians, to check the numbers quoted in Section V
        for seed in SEEDS:
            _, bias, ratio = stats[(seed, loc)]
            q = "lfm2_5_350m_quantum"
            lin = [np.median(bias[k]) for k in ORDER[:-1]]
            linr = [np.median(ratio[k]) for k in ORDER[:-1]]
            print(f"{loc:9s} seed {seed}: VQC median bias {np.median(bias[q]):+.3f} ratio "
                  f"{np.median(ratio[q]):.2f} | linear bias {min(lin):+.3f}..{max(lin):+.3f} "
                  f"ratio {min(linr):.2f}..{max(linr):.2f}")


if __name__ == "__main__":
    main()
