# VQC Head for BBR over Starlink

Code for the paper **"Modular Quantum-Classic AI Framework for BBR Congestion
Control over Starlink Internet"** (Chung Bao An Nguyen, Shiva Raj Pokhrel and
Jonathan Kua, Deakin University).

The framework follows the SLM-BBR pipeline: Starlink BBR traces are turned into
a token sequence for a LoRA-adapted language model, and an action head predicts
one of eleven pacing gains. The action head is interchangeable, and three heads
are compared on the same backbone, data, labels and training settings:

| Head | Path | Trainable parameters |
|---|---|---|
| Linear head | `Linear(1024, 11)` | 11,275 |
| VQC head | `LayerNorm -> Linear(1024, 8) -> pi*tanh(./4) -> 8-qubit VQC -> Linear(8, 11)` | 10,355 |
| Classical twin | `LayerNorm -> Linear(1024, 8) -> tanh(./4) -> Linear(8, 11)` | 10,347 |

The VQC uses `R_Y` angle encoding, one layer of trainable `R_Y` rotations
(initialised to zero), a CNOT ring and per-qubit Pauli-Z measurement. It is
simulated exactly (statevector, no shots or noise): a PyTorch implementation is
used for training and the Qiskit `EstimatorQNN` implementation for the Tokyo
evaluation; the two agree to within 1e-5.

All heads are evaluated **offline** on the prediction of heuristic target gains.
No head is run as a live congestion controller.

## Repository layout

```text
.
|-- generate_bbr_experience_pool.py   # raw iperf3 traces -> development experience pool
|-- generate_bbr_dev_split.py         # trace-level train/validation split
|-- train_modern_lora.py              # training entry point (all three heads)
|-- freeze_tokyo_comparison.py        # hash and freeze final checkpoints
|-- prepare_tokyo_evaluation.py       # build the held-out Tokyo pool (one-time gate)
|-- evaluate_tokyo_models.py          # Tokyo evaluation, linear-head backbones
|-- evaluate_tokyo_gpt_quantum.py     # Tokyo evaluation, VQC head
|-- run_multiseed_head_comparison.py  # multi-seed train -> freeze -> Tokyo evaluation
|-- evaluate_seen_locations.py        # validation-trace predictions (Mumbai, Sao Paulo)
|-- config.py                         # backbone registry with pinned revisions
|-- utils/                            # action space, phase detection, labels, metrics
|-- plm_special/                      # encoder, policy, LoRA, linear/VQC/twin heads
|-- reports/                          # tables and figures used in the paper
`-- tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Backbones are loaded from a local directory set by `LM_BBR_LOCAL_MODEL_ROOT`.
The Hugging Face IDs and pinned revisions are listed in `config.py`:

| Backbone | Hugging Face ID |
|---|---|
| LFM2.5-350M (used for the head comparison) | `LiquidAI/LFM2.5-350M` |
| Granite-4.0-350M | `ibm-granite/granite-4.0-350m` |
| Pleias-RAG-350M | `PleIAs/Pleias-RAG-350M` |
| Gemma-3-270M | `google/gemma-3-270m` |

## Data

The Starlink BBR traces are those of SLM-BBR (six locations: Ohio, Sao Paulo,
London, Mumbai, Sydney and Tokyo). They are not included in this repository.
Place them under `tcp-cc-starlink-main/` with the layout

```text
tcp-cc-starlink-main/<stream-group>/<Location>/**/bbr_<Location>__*.json
```

where `<stream-group>` is one of `downlink-sequential-logs`,
`uplink-sequential-logs`, `downlink-competitive-logs` and
`uplink-competitive-logs`. Tokyo is held out: the development pipeline refuses
Tokyo traces, and the Tokyo pool can only be built through
`prepare_tokyo_evaluation.py`.

## Running the pipeline

```bash
# 1. Development experience pool (Ohio, Sao Paulo, London, Mumbai, Sydney)
python generate_bbr_experience_pool.py \
  --raw-root tcp-cc-starlink-main \
  --output data/processed/dev_pool.pkl \
  --locations Ohio SaoPaulo London Mumbai Sydney \
  --stream-groups downlink-sequential-logs uplink-sequential-logs \
                  downlink-competitive-logs uplink-competitive-logs \
  --max-files 1000

# 2. Trace-level split (2 of 10 traces per location and stream group for validation)
python generate_bbr_dev_split.py \
  --input data/processed/dev_pool.pkl \
  --output-dir data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003 \
  --seed 100003

# 3. Train one model (example: LFM2.5-350M with the VQC head)
python train_modern_lora.py \
  --model-key lfm2_5_350m --output-dir runs/lfm2_5_350m_quantum \
  --rank 128 --alpha 32 --dropout 0.05 --epochs 100 \
  --sequence-length 20 --sample-step 20 --learning-rate 1e-4 \
  --weight-decay 1e-4 --grad-accum-steps 32 --seed 100003 \
  --head-type quantum --n-qubits 8 --quantum-depth 1 \
  --quantum-ansatz trainable_ry_layers --head-input-layernorm \
  --bottleneck-temperature 4 --quantum-angle-scale pi --quantum-backend torch

# 4. Freeze checkpoints, then open the Tokyo gate once
python freeze_tokyo_comparison.py --run-dir runs/lfm2_5_350m_quantum
python prepare_tokyo_evaluation.py --tokyo-location-flag 5 \
  --acknowledge-final-held-out-evaluation

# 5. Multi-seed comparison of the three heads (train, freeze, evaluate on Tokyo)
python run_multiseed_head_comparison.py run --dry-run
python run_multiseed_head_comparison.py run --device cuda --parallel 4
python reports/summarize_multiseed_head_comparison.py
```

The exact training arguments of each head are in `COMMON_TRAIN_ARGS` and
`HEAD_TRAIN_ARGS` in `run_multiseed_head_comparison.py`. The final-epoch
checkpoint is always used; Tokyo is never used for training, model selection or
any setting.

## Reproducing the paper

| Paper item | Script |
|---|---|
| Table IV (Tokyo results, four seeds) | `reports/summarize_multiseed_head_comparison.py` |
| Macro-F1, balanced accuracy and bootstrap CIs (Section V-B) | `reports/compute_per_action_metrics.py` |
| Table V (sending rate, surrogate retransmissions) | per-interval surrogate values written to each `predictions.csv` by `utils/tokyo_evaluation.py` |
| Fig. 2 (VQC circuit) | `reports/generate_vqc_circuit_figure.py` |
| Fig. 3 (trace locations) | `reports/generate_locations_map.py` |
| Fig. 4 (training loss and accuracy) | `reports/generate_100epoch_line_charts.py` |
| Figs. 5-7 (`BW_UP` decisions per location) | `reports/generate_bwup_seed_compare.py` |
| Fig. 8 (pacing gains of one Tokyo trace) | `reports/generate_pacing_gain_lines_seed.py` |
| Fig. 9 (trainable parameters, peak memory) | `reports/generate_100epoch_bar_charts.py`, `reports/measure_vram_usage.py` |

The SLM-BBR curves in Fig. 4 are digitised estimates from the SLM-BBR paper
(`reports/digitize_paper_fig7.py`, values in
`reports/assets/digitized_paper_fig7.json`). They are not exact published
numbers.

## Notes

- **Seeds.** The head comparison uses seeds 100003, 300003, 400003 and 500003.
  A fifth seed (200003) is excluded because training of its linear head stopped
  with a non-finite loss in five attempts; its VQC run also predicts only
  `g = 1.25` in `BW_UP`.
- **Non-causal preprocessing.** Phase labels, target gains and return-to-go use
  future intervals of each trace, and the model receives earlier target gains
  through teacher forcing. Input features are not normalised.
- **Surrogate metrics.** Retransmissions and sending rate are estimated with the
  SLM-BBR surrogate model, not measured on a live connection.

## Tests

```bash
python -m unittest discover -s tests -t .
```
