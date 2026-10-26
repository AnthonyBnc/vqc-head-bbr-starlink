"""Batch preparation and seeding for training and evaluation."""

import random

import numpy as np
import torch

from bbr_lm.bbr import ACTION_LEVELS, validate_phase_action


def _to_device(tensor, device):
    """Copy a CPU tensor to ``device`` (values identical to ``tensor.to(device)``).

    For CUDA the copy goes through pinned memory with ``non_blocking=True``: a
    plain pageable host-to-device copy makes the CPU wait until every queued GPU
    kernel has finished, which serialises CPU and GPU work at batch size 1.
    """
    device = torch.device(device)
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def _process_tensors_cpu(batch):
    """Build and validate the batch tensors on the CPU (no GPU synchronisation).

    ``actions`` is returned as raw action indices (float32); the scaling to
    (a + 1) / ACTION_LEVELS happens on the target device in
    ``_finish_on_device`` so the values are bit-identical to the old code
    (CUDA divides by a scalar via its reciprocal, the CPU divides exactly).
    """
    states, actions, returns, timesteps = batch

    states = torch.cat(states, dim=0).unsqueeze(0).float()
    actions = torch.as_tensor(actions, dtype=torch.float32).reshape(1, -1)
    labels = actions.long()
    if torch.any(actions != labels) or torch.any(labels < 0) or torch.any(labels >= ACTION_LEVELS):
        raise ValueError(
            f"Action labels must be integer indices in [0, {ACTION_LEVELS - 1}]"
        )
    returns = torch.as_tensor(returns, dtype=torch.float32).reshape(1, -1, 1)
    timesteps = torch.as_tensor(timesteps, dtype=torch.int32).unsqueeze(0)

    return states, actions, returns, timesteps, labels


def _finish_on_device(cpu_tensors, device):
    states, actions, returns, timesteps, labels = (
        _to_device(tensor, device) for tensor in cpu_tensors
    )
    actions = ((actions + 1) / ACTION_LEVELS).unsqueeze(2)
    return states, actions, returns, timesteps, labels


def process_bbr_batch(batch, device='cpu'):
    """Process a BBR batch and preserve its per-sample macro-phases."""
    if len(batch) != 5:
        raise ValueError(
            "BBR batches require states, actions, returns, timesteps, and phases"
        )

    states, actions, returns, timesteps, collated_phases = batch
    cpu_tensors = _process_tensors_cpu((states, actions, returns, timesteps))
    tensors = _finish_on_device(cpu_tensors, device)
    phases = []
    for item in collated_phases:
        if isinstance(item, str):
            phases.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], str):
            phases.append(item[0])
        else:
            raise ValueError("Unsupported collated BBR phase value: {!r}".format(item))

    labels = cpu_tensors[-1].reshape(-1).tolist()  # CPU copy: no GPU synchronisation
    if len(phases) != len(labels):
        raise ValueError(
            "BBR phase count {} does not match action count {}".format(
                len(phases), len(labels)
            )
        )

    for phase, action in zip(phases, labels):
        validate_phase_action(phase, action)

    return (*tensors, tuple(phases))


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
