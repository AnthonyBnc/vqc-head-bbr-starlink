"""Generate 2 single-axes bar charts across all 9 models -- our 5 (the 4
classical backbones plus gpt_quantum, all from the 100-epoch run
four_model_100epoch_v1) and the source paper's 4 SLMs
(GPT-2/T5/GPT-Neo/SmolLM2):

1. Trainable LoRA parameter percentage
2. Mean VRAM usage

Companion to generate_100epoch_line_charts.py (mean loss / mean accuracy).
generate_nine_model_charts.py produces the same two charts keyed to the
earlier 16-epoch run and is left untouched.

PROVENANCE, do not strip when editing:
- Trainable %% for our 5 is computed from each 100-epoch run's own
  run.manifest.json (lora_trainable_parameters / backbone_parameters).
- VRAM for our 5 is the Apple MPS measurement from
  reports/measure_vram_usage.py, carried in nine_model_comparison.json. It was
  NOT measured on the RTX 4090 that actually ran the 100-epoch training --
  no VRAM profiling was instrumented there. It is hardware-transferred, and
  labelled as such on the chart.
- Paper rows are published reference values (Table II / Fig. 8b), measured on
  2x NVIDIA RTX 6000 over a 150-epoch budget.
Different hardware and different epoch budgets on both sides: read the two
colors as two separate groups, do not compute cross-group deltas.

Run: .venv/bin/python reports/generate_100epoch_bar_charts.py
"""

import json

import matplotlib.pyplot as plt

from chart_style import DISPLAY, ROOT

NINE_MODEL_DATA = ROOT / "reports" / "assets" / "nine_model_comparison.json"
FIG_DIR_100EPOCH = ROOT / "reports" / "figures" / "100epoch"

CLASSICAL_MODELS = ["granite_4_0_350m", "pleias_rag_350m", "lfm2_5_350m", "gemma_3_270m"]
MODELS_100EPOCH = CLASSICAL_MODELS + ["gpt_quantum"]

_RUN_ROOT = ROOT / "data/processed/lora_training/four_model_100epoch_v1"
RUN_MANIFESTS_100EPOCH = {
    model: _RUN_ROOT / f"{model}_classical/run.manifest.json" for model in CLASSICAL_MODELS
}
RUN_MANIFESTS_100EPOCH["gpt_quantum"] = _RUN_ROOT / "lfm2_5_350m_quantum/run.manifest.json"

# gpt_quantum reuses the LFM2.5-350M backbone and LoRA config unchanged, so its
# trainable %% and VRAM land on top of that model's bars. The VRAM measurement
# carried in nine_model_comparison.json was taken on the 16-epoch run; the head
# config is identical, so it transfers.

OURS_COLOR = "#4C72B0"
PAPER_COLOR = "#DD8452"

# Font family + base sizes are set once, centrally, in
# chart_style.py's plt.rcParams.update(...) block (imported
# below) -- edit font/size there, not here, so both this script and
# generate_100epoch_line_charts.py stay in sync.


def _ours_rows():
    """Trainable %% straight from the 100-epoch manifests; VRAM joined in from
    the MPS measurement (see PROVENANCE above)."""
    vram_by_key = {
        row["model_key"]: row["mean_vram_usage_gb"]
        for row in json.loads(NINE_MODEL_DATA.read_text())["this_work"]
    }
    rows = []
    for model in MODELS_100EPOCH:
        manifest = json.loads(RUN_MANIFESTS_100EPOCH[model].read_text())
        percent = 100.0 * manifest["lora_trainable_parameters"] / manifest["backbone_parameters"]
        rows.append(
            {
                "display_name": DISPLAY[model],
                "lora_trainable_parameter_percent": percent,
                "mean_vram_usage_gb": vram_by_key[model],
            }
        )
    return rows


def _paper_rows():
    return json.loads(NINE_MODEL_DATA.read_text())["published_reference"]


SHORT_NAMES = {
    "Granite-4.0-350M": "Granite-4.0\n350M",
    "Pleias-RAG-350M": "Pleias-RAG\n350M",
    "LFM2.5-350M": "LFM2.5\n350M",
    "Gemma-3-270M": "Gemma-3\n270M",
    "LFM2.5-350M + VQC head": "LFM2.5\n+ VQC head",
}
FIG_DIR_PAPER = ROOT / "reports" / "figures" / "paper"


def _bar_chart(values_key, ylabel, value_format, footnote, filename):
    """Larger labels, horizontal two-line tick labels, and a
    visible split between values measured here and values reported in SLM-BBR."""
    ours, paper = _ours_rows(), _paper_rows()
    rows = ours + paper
    names = [SHORT_NAMES.get(r["display_name"], r["display_name"]) for r in rows]
    values = [r[values_key] for r in rows]
    colors = [OURS_COLOR] * len(ours) + [PAPER_COLOR] * len(paper)
    hatches = [""] * len(ours) + ["//"] * len(paper)

    fig, ax = plt.subplots(figsize=(13, 5.6))
    x = list(range(len(rows)))
    # small gap between the two groups
    x = [i if i < len(ours) else i + 0.6 for i in x]
    bars = ax.bar(x, values, color=colors, width=0.7, hatch=hatches, edgecolor="white")
    for bar, value in zip(bars, values):
        ax.annotate(
            value_format.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=15,
        )
    split = len(ours) - 0.5 + 0.3
    ax.axvline(split, color="#444444", linestyle="--", linewidth=1.2)
    top = max(values) * 1.32
    ax.set_ylim(0, top)
    ax.text((len(ours) - 1) / 2, top * 0.93, "This work (measured)", ha="center", fontsize=16, weight="bold")
    ax.text(len(ours) + 0.6 + (len(paper) - 1) / 2, top * 0.93, "SLM-BBR (reported)", ha="center", fontsize=16, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    FIG_DIR_100EPOCH.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        kwargs = {"dpi": 200} if ext == "png" else {}
        fig.savefig(FIG_DIR_100EPOCH / f"{filename}.{ext}", bbox_inches="tight", **kwargs)
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 200} if ext == "png" else {}
        fig.savefig(FIG_DIR_PAPER / f"{filename}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"saved {filename}.png/svg/pdf")


def generate_trainable_parameter_bar():
    _bar_chart(
        "lora_trainable_parameter_percent",
        "Trainable Parameters (%)",
        "{:.2f}%",
        "This work: all 5 runs share LoRA rank 128, so differences here are backbone architecture only.\n"
        "gpt_quantum reuses the LFM2.5-350M backbone and LoRA config unchanged, so its bar coincides with it.\n"
        "The paper does not publish its LoRA rank / target modules -- the two groups are not on a common\n"
        "configuration. Read them as two separate groups; do not compute cross-group deltas.",
        "100epoch_trainable_parameters_combined",
    )


def generate_vram_bar():
    _bar_chart(
        "mean_vram_usage_gb",
        "Peak training memory (GB)",
        "{:.2f} GB",
        "This work: peak memory over one real training step, measured on Apple MPS "
        "(reports/measure_vram_usage.py) -- NOT\nmeasured on the RTX 4090 that ran the 100-epoch training, "
        "where no VRAM profiling was instrumented.\nPaper: Fig. 8(b), 2x NVIDIA RTX 6000. Different hardware "
        "on both sides; do not compute cross-group deltas.",
        "100epoch_vram_combined",
    )


def main():
    generate_trainable_parameter_bar()
    generate_vram_bar()


if __name__ == "__main__":
    main()
