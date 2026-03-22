from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from simulator.rl_interface import resolve_policy_action, sanitize_action

HIERARCHICAL_MACRO_MODES = (
    "engage",
    "request_help",
    "follow_chem",
    "patrol",
)

HIERARCHICAL_MICRO_ACTIONS = (
    ("move", (-1, 0)),
    ("move", (1, 0)),
    ("move", (0, -1)),
    ("move", (0, 1)),
    ("move", (0, 0)),
    ("attack", None),
)

ENGAGE_MODE_INDEX = HIERARCHICAL_MACRO_MODES.index("engage")
REQUEST_HELP_MODE_INDEX = HIERARCHICAL_MACRO_MODES.index("request_help")
FOLLOW_CHEM_MODE_INDEX = HIERARCHICAL_MACRO_MODES.index("follow_chem")
PATROL_MODE_INDEX = HIERARCHICAL_MACRO_MODES.index("patrol")
MICRO_STAY_INDEX = HIERARCHICAL_MICRO_ACTIONS.index(("move", (0, 0)))
MICRO_ATTACK_INDEX = HIERARCHICAL_MICRO_ACTIONS.index(("attack", None))
NO_TARGET_COMPARTMENT_INDEX = config.N_COMPARTMENTS


def macro_mode_to_index(mode: str) -> int:
    return HIERARCHICAL_MACRO_MODES.index(str(mode))


def micro_action_to_index(action: tuple[str, Any]) -> int:
    if action in HIERARCHICAL_MICRO_ACTIONS:
        return HIERARCHICAL_MICRO_ACTIONS.index(action)
    if action[0] in {"signal", "signal_low", "signal_medium", "signal_high"}:
        return MICRO_STAY_INDEX
    raise ValueError(f"Unsupported hierarchical micro action label: {action}")


def macro_index_to_mode(index: int) -> str:
    idx = max(0, min(len(HIERARCHICAL_MACRO_MODES) - 1, int(index)))
    return HIERARCHICAL_MACRO_MODES[idx]


def micro_index_to_action(index: int) -> tuple[str, Any]:
    idx = max(0, min(len(HIERARCHICAL_MICRO_ACTIONS) - 1, int(index)))
    return HIERARCHICAL_MICRO_ACTIONS[idx]


def build_macro_mask(env) -> np.ndarray:
    mask = np.ones(len(HIERARCHICAL_MACRO_MODES), dtype=np.float32)
    if not any(
        action[0] in {"signal", "signal_low", "signal_medium", "signal_high"}
        for action in env.get_macrophage_actions()
    ):
        mask[REQUEST_HELP_MODE_INDEX] = 0.0
    return mask


def build_micro_action_masks(env) -> np.ndarray:
    valid_actions = set(env.get_macrophage_actions())
    masks = np.zeros((len(HIERARCHICAL_MACRO_MODES), len(HIERARCHICAL_MICRO_ACTIONS)), dtype=np.float32)

    for idx, action in enumerate(HIERARCHICAL_MICRO_ACTIONS):
        if action in valid_actions:
            masks[ENGAGE_MODE_INDEX, idx] = 1.0
            if action != ("attack", None):
                masks[FOLLOW_CHEM_MODE_INDEX, idx] = 1.0
                masks[PATROL_MODE_INDEX, idx] = 1.0

    masks[REQUEST_HELP_MODE_INDEX, MICRO_STAY_INDEX] = 1.0

    # Safety fallback: keep stay valid if the env exposes it.
    if ("move", (0, 0)) in valid_actions:
        masks[:, MICRO_STAY_INDEX] = np.maximum(masks[:, MICRO_STAY_INDEX], 1.0)

    return masks.astype(np.float32)


def resolve_hierarchical_action(env, macro_index: int, micro_index: int) -> tuple[str, Any]:
    macro_mode = macro_index_to_mode(macro_index)
    if macro_mode == "request_help":
        resolved = resolve_policy_action(env, ("request_help", None))
        safe_action, _ = sanitize_action(env, resolved)
        return safe_action

    candidate_action = micro_index_to_action(micro_index)
    if macro_mode != "engage" and candidate_action == ("attack", None):
        candidate_action = ("move", (0, 0))
    safe_action, _ = sanitize_action(env, candidate_action)
    return safe_action


def _bacteria_burden_weight(bacterium) -> float:
    return {
        "planktonic": 1.0,
        "attached": 1.4,
        "microcolony": 2.0,
        "biofilm": 2.6,
    }.get(getattr(bacterium, "state", "planktonic"), 1.0)


def _highest_burden_compartment(env) -> int:
    burdens = np.zeros(config.N_COMPARTMENTS, dtype=np.float32)
    for bacterium in env.bacteria:
        bx, by = bacterium.position
        cid = int(env.compartment_map[by, bx])
        if cid >= 0:
            burdens[cid] += _bacteria_burden_weight(bacterium)
    if float(burdens.sum()) <= 1e-6:
        return NO_TARGET_COMPARTMENT_INDEX
    return int(np.argmax(burdens))


def compute_privileged_signals(env) -> dict[str, Any]:
    mx, my = env.macrophage.position
    extended_radius = max(config.MACROPHAGE_SENSE_RADIUS + 2, config.MACROPHAGE_SENSE_RADIUS * 2)

    local_hidden_burden = 0.0
    global_burden = 0.0
    nearest_distance = float(env.width + env.height)

    for bacterium in env.bacteria:
        bx, by = bacterium.position
        distance = abs(bx - mx) + abs(by - my)
        burden = _bacteria_burden_weight(bacterium)
        global_burden += burden
        if distance <= extended_radius:
            local_hidden_burden += burden
        nearest_distance = min(nearest_distance, float(distance))

    nearby_neutrophils = env.get_local_neutrophils((mx, my), config.MACROPHAGE_SENSE_RADIUS)
    local_peak = env.get_local_chemokine_peak((mx, my), config.MACROPHAGE_SIGNAL_TRIGGER_RADIUS)
    queue_norm = float(len(env.recruitment_queue) / max(1, config.MAX_NEUTROPHIL_POOL))
    global_burden_norm = float(global_burden / max(1.0, config.INITIAL_BACTERIA_COUNT * 3.5))
    local_hidden_burden_norm = float(local_hidden_burden / max(1.0, config.INITIAL_BACTERIA_COUNT * 2.5))
    nearest_distance_norm = float(
        np.clip(nearest_distance / max(1.0, env.width + env.height), 0.0, 1.0)
    )
    target_compartment = _highest_burden_compartment(env)
    target_one_hot = np.zeros(config.N_COMPARTMENTS + 1, dtype=np.float32)
    target_one_hot[target_compartment] = 1.0

    escalation_needed = float(
        (
            local_hidden_burden_norm >= 0.45
            or global_burden_norm >= 0.55
            or (
                local_peak >= config.MACROPHAGE_SIGNAL_LOCAL_CHEMOKINE_THRESHOLD
                and local_hidden_burden_norm >= 0.25
            )
        )
        and len(nearby_neutrophils) == 0
    )

    privileged_scalars = np.asarray(
        [
            local_hidden_burden_norm,
            global_burden_norm,
            nearest_distance_norm,
            float(np.clip(len(nearby_neutrophils) / max(1, config.MAX_NEUTROPHIL_POOL), 0.0, 1.0)),
            queue_norm,
            float(
                np.clip(
                    env.recent_chemokine_burden / max(config.NEUTROPHIL_RECRUITMENT_THRESHOLD * 1.5, 1e-6),
                    0.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    local_peak / max(config.NEUTROPHIL_RECRUITMENT_THRESHOLD * 1.5, 1e-6),
                    0.0,
                    1.0,
                )
            ),
            *target_one_hot.tolist(),
        ],
        dtype=np.float32,
    )

    return {
        "privileged_scalars": privileged_scalars,
        "local_hidden_burden": local_hidden_burden_norm,
        "escalation_needed": escalation_needed,
        "target_compartment": int(target_compartment),
    }


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_tensor = mask.to(device=logits.device, dtype=logits.dtype)
    if mask_tensor.ndim < logits.ndim:
        for _ in range(logits.ndim - mask_tensor.ndim):
            mask_tensor = mask_tensor.unsqueeze(0)
    return logits.masked_fill(mask_tensor <= 0.0, torch.finfo(logits.dtype).min)


class HierarchicalRecurrentPolicy(nn.Module):
    """Recurrent hierarchical actor-critic for partial-observation host control."""

    def __init__(
        self,
        scalar_dim: int,
        privileged_dim: int,
        grid_channels: int = 6,
        hidden_size: int = 128,
        recurrent_layers: int = 1,
    ):
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.privileged_dim = int(privileged_dim)
        self.grid_channels = int(grid_channels)
        self.hidden_size = int(hidden_size)
        self.recurrent_layers = int(recurrent_layers)

        self.grid_encoder = nn.Sequential(
            nn.Conv2d(self.grid_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 192),
            nn.ReLU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(self.scalar_dim, 64),
            nn.ReLU(),
        )
        self.recurrent = nn.LSTM(
            input_size=192 + 64,
            hidden_size=self.hidden_size,
            num_layers=self.recurrent_layers,
        )
        self.macro_head = nn.Linear(self.hidden_size, len(HIERARCHICAL_MACRO_MODES))
        self.micro_head = nn.Linear(
            self.hidden_size,
            len(HIERARCHICAL_MACRO_MODES) * len(HIERARCHICAL_MICRO_ACTIONS),
        )
        self.aux_burden_head = nn.Linear(self.hidden_size, 1)
        self.aux_escalation_head = nn.Linear(self.hidden_size, 1)
        self.aux_target_head = nn.Linear(self.hidden_size, config.N_COMPARTMENTS + 1)
        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_size + self.privileged_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(
            self.recurrent_layers,
            int(batch_size),
            self.hidden_size,
            dtype=torch.float32,
            device=device,
        )
        return zeros.clone(), zeros.clone()

    def _prepare_sequence_observations(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        time_steps, batch_size = grid.shape[0], grid.shape[1]
        flat_grid = grid.reshape(time_steps * batch_size, *grid.shape[2:])
        flat_scalars = scalars.reshape(time_steps * batch_size, scalars.shape[-1])
        grid_features = self.grid_encoder(flat_grid)
        scalar_features = self.scalar_encoder(flat_scalars)
        features = torch.cat([grid_features, scalar_features], dim=-1)
        return features.reshape(time_steps, batch_size, -1)

    def forward_sequence(
        self,
        grid: torch.Tensor,
        scalars: torch.Tensor,
        privileged_scalars: torch.Tensor | None = None,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        if grid.ndim == 4:
            grid = grid.unsqueeze(1)
        if scalars.ndim == 2:
            scalars = scalars.unsqueeze(1)
        features = self._prepare_sequence_observations(grid, scalars)
        if hidden_state is None:
            hidden_state = self.initial_state(features.shape[1], features.device)
        recurrent_output, next_state = self.recurrent(features, hidden_state)
        macro_logits = self.macro_head(recurrent_output)
        micro_logits = self.micro_head(recurrent_output).view(
            recurrent_output.shape[0],
            recurrent_output.shape[1],
            len(HIERARCHICAL_MACRO_MODES),
            len(HIERARCHICAL_MICRO_ACTIONS),
        )
        burden = self.aux_burden_head(recurrent_output).squeeze(-1)
        escalation = self.aux_escalation_head(recurrent_output).squeeze(-1)
        target_logits = self.aux_target_head(recurrent_output)

        if privileged_scalars is None:
            privileged_scalars = torch.zeros(
                recurrent_output.shape[0],
                recurrent_output.shape[1],
                self.privileged_dim,
                dtype=recurrent_output.dtype,
                device=recurrent_output.device,
            )
        elif privileged_scalars.ndim == 2:
            privileged_scalars = privileged_scalars.unsqueeze(1)
        value_input = torch.cat([recurrent_output, privileged_scalars], dim=-1)
        values = self.critic_head(value_input).squeeze(-1)

        return {
            "macro_logits": macro_logits,
            "micro_logits": micro_logits,
            "burden": burden,
            "escalation": escalation,
            "target_logits": target_logits,
            "values": values,
            "hidden_state": next_state,
        }

    def act(
        self,
        observation: dict[str, np.ndarray | torch.Tensor],
        macro_mask: np.ndarray | torch.Tensor,
        micro_masks: np.ndarray | torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        privileged_scalars: np.ndarray | torch.Tensor | None = None,
        deterministic: bool = True,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        if device is None:
            device = next(self.parameters()).device

        grid = observation["grid"]
        scalars = observation["scalars"]
        if not torch.is_tensor(grid):
            grid = torch.as_tensor(grid, dtype=torch.float32, device=device)
        if not torch.is_tensor(scalars):
            scalars = torch.as_tensor(scalars, dtype=torch.float32, device=device)
        if privileged_scalars is not None and not torch.is_tensor(privileged_scalars):
            privileged_scalars = torch.as_tensor(privileged_scalars, dtype=torch.float32, device=device)
        if not torch.is_tensor(macro_mask):
            macro_mask = torch.as_tensor(macro_mask, dtype=torch.float32, device=device)
        if not torch.is_tensor(micro_masks):
            micro_masks = torch.as_tensor(micro_masks, dtype=torch.float32, device=device)

        with torch.no_grad():
            outputs = self.forward_sequence(
                grid.unsqueeze(0),
                scalars.unsqueeze(0),
                None if privileged_scalars is None else privileged_scalars.unsqueeze(0),
                hidden_state=hidden_state,
            )
            macro_logits = outputs["macro_logits"][0, 0]
            masked_macro_logits = mask_logits(macro_logits, macro_mask)
            macro_dist = torch.distributions.Categorical(logits=masked_macro_logits)
            if deterministic:
                macro_index = int(torch.argmax(masked_macro_logits).item())
            else:
                macro_index = int(macro_dist.sample().item())

            micro_logits = outputs["micro_logits"][0, 0, macro_index]
            masked_micro_logits = mask_logits(micro_logits, micro_masks[macro_index])
            micro_dist = torch.distributions.Categorical(logits=masked_micro_logits)
            if deterministic:
                micro_index = int(torch.argmax(masked_micro_logits).item())
            else:
                micro_index = int(micro_dist.sample().item())

            return {
                "macro_index": macro_index,
                "micro_index": micro_index,
                "macro_mode": macro_index_to_mode(macro_index),
                "micro_action": micro_index_to_action(micro_index),
                "log_prob": float((macro_dist.log_prob(torch.tensor(macro_index, device=device)) + micro_dist.log_prob(torch.tensor(micro_index, device=device))).item()),
                "value": float(outputs["values"][0, 0].item()),
                "hidden_state": outputs["hidden_state"],
                "aux": {
                    "burden": float(outputs["burden"][0, 0].item()),
                    "escalation": float(torch.sigmoid(outputs["escalation"][0, 0]).item()),
                    "target_compartment": int(torch.argmax(outputs["target_logits"][0, 0]).item()),
                },
            }


def save_hierarchical_checkpoint(
    path: str | Path,
    model: HierarchicalRecurrentPolicy,
    observation_config: dict[str, Any],
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "hierarchical_macrophage_v1",
        "stage": str(stage),
        "model_state": model.state_dict(),
        "model_config": {
            "scalar_dim": model.scalar_dim,
            "privileged_dim": model.privileged_dim,
            "grid_channels": model.grid_channels,
            "hidden_size": model.hidden_size,
            "recurrent_layers": model.recurrent_layers,
        },
        "observation_config": dict(observation_config),
        "metadata": {} if metadata is None else dict(metadata),
    }
    torch.save(payload, output_path)
    return output_path


def load_hierarchical_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format") != "hierarchical_macrophage_v1":
        raise ValueError(f"Unsupported hierarchical checkpoint format: {path}")
    return payload
