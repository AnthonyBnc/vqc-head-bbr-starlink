"""Fig. 2: VQC head circuit with the four stages labelled.

Matches plm_special/quantum_head.py with n_q = 8, D = 1, no re-uploading:
R_Y(theta_i) angle encoding on |0>, one trainable R_Y(omega_i) per qubit,
CNOT(i, (i+1) mod n_q) for i = 0..n_q-1 (ring), Pauli-Z expectation per qubit.

Run: python3 reports/generate_vqc_circuit_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures" / "paper"
NQ = 8

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Times", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
})

ENC_C, VAR_C, MEAS_C = "#DCE9F7", "#FBE3CF", "#E3F1DF"
EDGE = "#333333"


def gate(ax, x, y, text, face):
    w, h = 0.62, 0.62
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=face, ec=EDGE, lw=0.7, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5, zorder=4)


def cnot(ax, x, yc, yt):
    ax.plot([x, x], [yc, yt], color=EDGE, lw=0.8, zorder=2)
    ax.add_patch(Circle((x, yc), 0.09, color=EDGE, zorder=3))
    ax.add_patch(Circle((x, yt), 0.2, fc="white", ec=EDGE, lw=0.8, zorder=3))
    ax.plot([x - 0.2, x + 0.2], [yt, yt], color=EDGE, lw=0.8, zorder=4)
    ax.plot([x, x], [yt - 0.2, yt + 0.2], color=EDGE, lw=0.8, zorder=4)


def bracket(ax, x0, x1, y, title, sub):
    ax.plot([x0, x0, x1, x1], [y - 0.15, y, y, y - 0.15], color=EDGE, lw=0.7)
    ax.text((x0 + x1) / 2, y + 0.12, sub, ha="center", va="bottom", fontsize=7.5)
    ax.text((x0 + x1) / 2, y + 0.62, title, ha="center", va="bottom",
            fontsize=8, weight="bold")


def main():
    x_enc, x_var = 1.0, 2.7
    x_cnot = [3.9 + 0.55 * k for k in range(NQ)]
    x_meas = x_cnot[-1] + 1.2
    ys = [NQ - 1 - i for i in range(NQ)]  # qubit 0 on top

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    for i, y in enumerate(ys):
        ax.plot([0.35, x_meas], [y, y], color=EDGE, lw=0.7, zorder=1)
        ax.text(0.3, y, rf"$q_{i}$: $|0\rangle$", ha="right", va="center", fontsize=7.5)
        gate(ax, x_enc, y, r"$R_Y$", ENC_C)
        gate(ax, x_var, y, r"$R_Y$", VAR_C)
        gate(ax, x_meas, y, r"$Z$", MEAS_C)
    for k in range(NQ):
        cnot(ax, x_cnot[k], ys[k], ys[(k + 1) % NQ])

    top = NQ - 0.35
    bracket(ax, x_enc - 0.35, x_enc + 0.35, top, "Encoding", r"$R_Y(\theta_i)$")
    bracket(ax, x_var - 0.35, x_var + 0.35, top, "Variational", r"$R_Y(\omega_i)$")
    bracket(ax, x_cnot[0] - 0.25, x_cnot[-1] + 0.25, top, "Entanglement", "CNOT ring")
    bracket(ax, x_meas - 0.35, x_meas + 0.35, top, "Measurement", r"$\langle Z_i\rangle$")

    ax.set_xlim(-0.9, x_meas + 0.6)
    ax.set_ylim(-0.6, NQ + 1.1)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "quantum_circuit.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "quantum_circuit.png", dpi=300, bbox_inches="tight")
    print("written", FIG_DIR / "quantum_circuit.pdf")


if __name__ == "__main__":
    main()
