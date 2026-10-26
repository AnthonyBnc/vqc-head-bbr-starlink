"""Per-feature state encoder (Eq. 17, first part).

Each of the nine state features is passed through its own
``Linear(1, embed_dim) -> LeakyReLU`` block. The projection to the backbone
hidden size is done by ``OfflineRLPolicy.embed_state1..9``.
"""

import torch.nn as nn

STATE_FEATURES = 9


class EncoderNetwork(nn.Module):
    """Encode the nine state features separately.

    State order: location, stream, interval end time, throughput,
    retransmissions, cwnd, rwnd, RTT, RTT variation.
    """

    def __init__(self, embed_dim=128, **unused):
        super().__init__()
        self.embed_dim = embed_dim
        # The attribute names fc1..fc9 are part of the saved checkpoints.
        for index in range(1, STATE_FEATURES + 1):
            setattr(self, "fc{}".format(index), nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU()))

    def forward(self, state):
        """Map ``(batch, seq_len, 9)`` states to nine ``(batch, seq_len, embed_dim)`` tensors."""
        batch_size, seq_len = state.shape[0], state.shape[1]
        state = state.reshape(batch_size * seq_len, STATE_FEATURES, 1)
        return tuple(
            getattr(self, "fc{}".format(index + 1))(state[..., index:index + 1, :]).reshape(
                batch_size, seq_len, -1
            )
            for index in range(STATE_FEATURES)
        )
