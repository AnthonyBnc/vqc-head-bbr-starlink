"""Shared metrics for phase-masked BBR action prediction."""

from collections import Counter

import torch
import torch.nn.functional as F

from utils.bbr import ACTION_LEVELS, BBR_PHASES, mask_sequence_logits, validate_phase_action

LOSS_WEIGHTING_NONE = "none"
LOSS_WEIGHTING_PHASE_BALANCED = "phase_balanced"
LOSS_WEIGHTING_OPTIONS = (LOSS_WEIGHTING_NONE, LOSS_WEIGHTING_PHASE_BALANCED)


def phase_balanced_weights(phases, batch_size, device):
    """Return equal-total phase weights for a flattened batch/sequence."""
    repeated_phases = tuple(phases) * batch_size
    counts = Counter(repeated_phases)
    present_phases = [phase for phase in BBR_PHASES if counts[phase]]
    if not present_phases:
        raise ValueError("Cannot build phase-balanced weights without phases")
    total = len(repeated_phases)
    weights = [
        total / (len(present_phases) * counts[phase])
        for phase in repeated_phases
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def masked_cross_entropy(logits, labels, phases, loss_weighting=LOSS_WEIGHTING_NONE):
    """Compute phase-masked CE with an optional training-only weighting scheme."""
    if loss_weighting not in LOSS_WEIGHTING_OPTIONS:
        raise ValueError("Unknown loss_weighting: {}".format(loss_weighting))
    masked_logits = mask_sequence_logits(logits, phases)
    flat_logits = masked_logits.reshape(-1, ACTION_LEVELS).float()
    flat_labels = labels.reshape(-1).long()
    if flat_labels.numel() != len(phases) * logits.shape[0]:
        raise ValueError("Label count does not match the masked logits")
    per_sample_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    if loss_weighting == LOSS_WEIGHTING_NONE:
        return masked_logits, per_sample_loss.mean()
    weights = phase_balanced_weights(phases, logits.shape[0], flat_logits.device)
    return masked_logits, (per_sample_loss * weights).sum() / weights.sum()


class BBRMetricAccumulator:
    def __init__(self):
        self.loss_sum = 0.0
        self.samples = 0
        self.correct = 0
        self.phase_samples = Counter()
        self.phase_correct = Counter()
        self.label_distribution = Counter()
        self.prediction_distribution = Counter()
        self.phase_label_distribution = {
            phase: Counter() for phase in BBR_PHASES
        }
        self.phase_prediction_distribution = {
            phase: Counter() for phase in BBR_PHASES
        }
        # Per-step results kept on the device and copied to the CPU in one go by
        # compute(); reading them every step would stall the CPU on the GPU.
        self._pending = []

    def update(self, logits, labels, phases, loss_weighting=LOSS_WEIGHTING_NONE):
        masked_logits, loss = masked_cross_entropy(
            logits, labels, phases, loss_weighting=loss_weighting
        )
        flat_logits = masked_logits.reshape(-1, ACTION_LEVELS).float()
        flat_labels = labels.reshape(-1).long()
        loss_sum = F.cross_entropy(flat_logits, flat_labels, reduction="sum")
        predictions = flat_logits.argmax(dim=-1)
        repeated_phases = tuple(phases) * logits.shape[0]
        self._pending.append(
            (
                repeated_phases,
                flat_labels.detach(),
                predictions.detach(),
                loss_sum.detach(),
                flat_labels.numel(),
            )
        )
        return masked_logits, loss

    def _flush(self):
        """Fold the pending per-step results in; same arithmetic as before."""
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        all_labels = torch.cat([item[1] for item in pending]).cpu().tolist()
        all_predictions = torch.cat([item[2] for item in pending]).cpu().tolist()
        loss_values = torch.stack([item[3] for item in pending]).cpu().tolist()
        offset = 0
        for (repeated_phases, _, _, _, count), loss_value in zip(pending, loss_values):
            labels = all_labels[offset:offset + count]
            predictions = all_predictions[offset:offset + count]
            offset += count
            for phase, label, prediction in zip(repeated_phases, labels, predictions):
                validate_phase_action(phase, label)
                validate_phase_action(phase, prediction)
                self.samples += 1
                self.loss_sum += loss_value / count
                self.correct += int(label == prediction)
                self.phase_samples[phase] += 1
                self.phase_correct[phase] += int(label == prediction)
                self.label_distribution[label] += 1
                self.prediction_distribution[prediction] += 1
                self.phase_label_distribution[phase][label] += 1
                self.phase_prediction_distribution[phase][prediction] += 1

    def compute(self):
        self._flush()
        if not self.samples:
            raise ValueError("Cannot compute BBR metrics without samples")
        per_phase_accuracy = {
            phase: (
                self.phase_correct[phase] / self.phase_samples[phase]
                if self.phase_samples[phase]
                else None
            )
            for phase in BBR_PHASES
        }
        present_phase_accuracies = [
            value for value in per_phase_accuracy.values() if value is not None
        ]
        return {
            "loss": self.loss_sum / self.samples,
            "accuracy": self.correct / self.samples,
            "macro_phase_accuracy": (
                sum(present_phase_accuracies) / len(present_phase_accuracies)
            ),
            "samples": self.samples,
            "per_phase_accuracy": per_phase_accuracy,
            "per_phase_samples": {
                phase: self.phase_samples[phase] for phase in BBR_PHASES
            },
            "label_distribution": {
                str(action): self.label_distribution[action]
                for action in range(ACTION_LEVELS)
            },
            "prediction_distribution": {
                str(action): self.prediction_distribution[action]
                for action in range(ACTION_LEVELS)
            },
            "per_phase_label_distribution": {
                phase: {
                    str(action): self.phase_label_distribution[phase][action]
                    for action in range(ACTION_LEVELS)
                }
                for phase in BBR_PHASES
            },
            "per_phase_prediction_distribution": {
                phase: {
                    str(action): self.phase_prediction_distribution[phase][action]
                    for action in range(ACTION_LEVELS)
                }
                for phase in BBR_PHASES
            },
        }


def evaluate_validation_criteria(
    metrics,
    overall_accuracy=None,
    up_accuracy=None,
    macro_phase_accuracy=None,
    max_up_prediction_share=None,
):
    """Evaluate declared reporting thresholds without using them for optimization."""
    from utils.bbr import BW_UP

    observed_up_accuracy = metrics["per_phase_accuracy"][BW_UP]
    up_predictions = metrics.get("per_phase_prediction_distribution", {}).get(
        BW_UP, {}
    )
    up_prediction_total = sum(up_predictions.values())
    observed_max_up_share = (
        max(up_predictions.values()) / up_prediction_total
        if up_prediction_total
        else None
    )
    declared = {
        "overall_accuracy": overall_accuracy,
        "up_accuracy": up_accuracy,
        "macro_phase_accuracy": macro_phase_accuracy,
        "max_up_prediction_share": max_up_prediction_share,
    }
    observed = {
        "overall_accuracy": metrics["accuracy"],
        "up_accuracy": observed_up_accuracy,
        "macro_phase_accuracy": metrics["macro_phase_accuracy"],
        "max_up_prediction_share": observed_max_up_share,
    }
    checks = {}
    for name, threshold in declared.items():
        if threshold is None:
            continue
        value = observed[name]
        checks[name] = (
            value <= threshold
            if name == "max_up_prediction_share"
            else value >= threshold
        )
    return {
        "declared": declared,
        "observed": observed,
        "checks": checks,
        "all_declared_criteria_pass": all(checks.values()) if checks else None,
    }
