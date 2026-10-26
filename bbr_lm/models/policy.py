"""Return-conditioned policy: token sequence -> backbone -> action head.

Each interval becomes 11 tokens ``[R_t, s^1_t, ..., s^9_t, a_t]`` (Section
III-B). The hidden state at the last state token ``s^9_t`` is passed to the
action head, which returns one logit per pacing gain (Section III-D).
"""

import torch
import torch.nn as nn

from bbr_lm.bbr import ACTION_LEVELS
from bbr_lm.heads.classical_twin import ClassicalBottleneckActionHead
from bbr_lm.heads.vqc import QuantumActionHead

STATE_FEATURES = 9
TOKENS_PER_INTERVAL = STATE_FEATURES + 2  # return, nine state features, action


class OfflineRLPolicy(nn.Module):
    """Language-model policy with an interchangeable action head.

    ``head_type`` selects the head: ``"classical"`` (linear head),
    ``"classical_twin"`` or ``"quantum"`` (VQC head).
    """

    def __init__(
        self,
        state_feature_dim,
        action_levels,
        state_encoder,
        plm,
        plm_embed_size,
        max_length=None,
        max_ep_len=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_out=None,
        head_type="classical",
        quantum_config=None,
        **unused,
    ):
        super().__init__()
        if action_levels != ACTION_LEVELS:
            raise ValueError(
                f"OfflineRLPolicy requires exactly {ACTION_LEVELS} BBR actions; "
                f"got {action_levels}"
            )

        self.action_levels = action_levels
        self.max_length = max_length
        self.plm = plm
        self.plm_embed_size = plm_embed_size
        self.device = device
        self.device_out = device if device_out is None else device_out

        # Token embeddings. The creation order fixes the random initialisation,
        # so it must not change.
        self.state_encoder = state_encoder
        self.state_feature_dim = state_feature_dim
        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        self.embed_action = nn.Linear(1, plm_embed_size).to(device)
        for index in range(1, STATE_FEATURES + 1):
            setattr(
                self,
                "embed_state{}".format(index),
                nn.Linear(state_feature_dim, plm_embed_size).to(device),
            )
        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)

        self.head_type = head_type
        self.quantum_config = None
        self.action_head = self._build_action_head(head_type, quantum_config or {}).to(device)
        if head_type == "quantum":
            self.quantum_config = self.action_head.manifest_config()

        # Everything except the backbone, in the order used by saved checkpoints.
        self.modules_except_plm = nn.ModuleList(
            [
                self.state_encoder,
                self.embed_timestep,
                self.embed_return,
                self.embed_action,
                self.embed_ln,
                *(getattr(self, "embed_state{}".format(i)) for i in range(1, STATE_FEATURES + 1)),
                self.action_head,
            ]
        )

    def _build_action_head(self, head_type, config):
        if head_type == "classical":
            return nn.Linear(self.plm_embed_size, self.action_levels)
        if head_type == "classical_twin":
            return ClassicalBottleneckActionHead(
                self.plm_embed_size,
                self.action_levels,
                bottleneck_dim=int(config.get("n_qubits", 4)),
                input_layernorm=bool(config.get("input_layernorm", False)),
                temperature=float(config.get("temperature", 1.0)),
            )
        if head_type == "quantum":
            return QuantumActionHead(
                self.plm_embed_size,
                self.action_levels,
                n_qubits=int(config.get("n_qubits", 4)),
                depth=int(config.get("depth", 2)),
                ansatz=config.get("ansatz", "trainable_ry_layers"),
                input_layernorm=bool(config.get("input_layernorm", False)),
                temperature=float(config.get("temperature", 1.0)),
                angle_scale=config.get("angle_scale", "pi"),
                backend=config.get("backend", "qiskit"),
            )
        raise ValueError(
            "Unknown head_type {!r}; expected classical, classical_twin, or quantum".format(head_type)
        )

    def _action_head_input_dtype(self):
        return next(self.action_head.parameters()).dtype

    @staticmethod
    def _last_hidden_state(outputs):
        """Return the final hidden states for the different backbone output types."""
        if isinstance(outputs, dict):
            if "last_hidden_state" in outputs:
                return outputs["last_hidden_state"]
            if "hidden_states" in outputs and len(outputs["hidden_states"]) > 0:
                return outputs["hidden_states"][-1]
            return next(iter(outputs.values()))
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and len(outputs.hidden_states) > 0:
            return outputs.hidden_states[-1]
        return outputs[0]

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        """Return action logits of shape ``(1, seq_len, 11)`` for one window."""
        assert actions.shape[0] == 1, "batch size must be 1"

        actions = actions.to(self.device)  # (1, seq_len, 1)
        returns = returns.to(self.device)  # (1, seq_len, 1)
        timesteps = timesteps.to(self.device)  # (1, seq_len)

        time_embeddings = self.embed_timestep(timesteps)
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_return(returns) + time_embeddings

        states_features = self.state_encoder(states.to(self.device))
        states_embeddings = [
            getattr(self, "embed_state{}".format(index + 1))(states_features[index]) + time_embeddings
            for index in range(STATE_FEATURES)
        ]

        # Sequence (R_1, s^1_1, ..., s^9_1, a_1, R_2, ...): stacking on a new
        # axis and reshaping interleaves the 11 tokens of each interval.
        seq_len = returns_embeddings.shape[1]
        stacked_inputs = torch.stack(
            (returns_embeddings, *states_embeddings, action_embeddings), dim=2
        ).reshape(1, seq_len * TOKENS_PER_INTERVAL, -1)
        max_context = getattr(self.plm.config, "max_position_embeddings", None)
        if max_context is not None and stacked_inputs.shape[1] > max_context:
            raise ValueError(
                f"Structured sequence has {stacked_inputs.shape[1]} tokens, "
                f"exceeding model context length {max_context}"
            )
        stacked_inputs_ln = self.embed_ln(stacked_inputs)

        if attention_mask is None:
            attention_mask = torch.ones(
                stacked_inputs_ln.shape[:2], dtype=torch.long, device=self.device
            )
        plm_dtype = next(self.plm.parameters()).dtype
        outputs = self.plm(
            inputs_embeds=stacked_inputs_ln.to(dtype=plm_dtype),
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = self._last_hidden_state(outputs)

        # Hidden state at the last state token s^9_t: positions 9, 20, 31, ...
        hidden_at_state = hidden[:, TOKENS_PER_INTERVAL - 2::TOKENS_PER_INTERVAL].contiguous()
        if hidden_at_state.shape[1] != seq_len:
            raise RuntimeError("Unexpected backbone output length for action positions")
        hidden_at_state = hidden_at_state.to(dtype=self._action_head_input_dtype())
        return self.action_head(hidden_at_state)
