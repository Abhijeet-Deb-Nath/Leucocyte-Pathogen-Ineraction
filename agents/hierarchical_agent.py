from __future__ import annotations

from pathlib import Path

import torch

from agents.hierarchical_policy import (
    HierarchicalRecurrentPolicy,
    build_macro_mask,
    build_micro_action_masks,
    compute_privileged_signals,
    load_hierarchical_checkpoint,
    resolve_hierarchical_action,
)
from simulator.rl_interface import MacrophageObservationBuilder


class HierarchicalMacrophageAgent:
    """Inference wrapper for hierarchical recurrent learned-controller checkpoints."""

    def __init__(
        self,
        model_path,
        observation_mode="partial_state",
        deterministic=True,
    ):
        if observation_mode != "partial_state":
            raise ValueError("Only partial_state inference is supported for hierarchical checkpoints")

        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Hierarchical model not found: {model_path}")

        payload = load_hierarchical_checkpoint(resolved_model_path, map_location="cpu")
        model_config = dict(payload["model_config"])
        self.policy = HierarchicalRecurrentPolicy(**model_config)
        self.policy.load_state_dict(payload["model_state"])
        self.policy.eval()

        observation_config = dict(payload.get("observation_config", {}))
        self.observer = MacrophageObservationBuilder(
            mode="partial_state",
            include_belief_features=observation_config.get("include_belief_features", True),
            partial_radius=observation_config.get("partial_radius"),
            include_action_mask=False,
        )
        self.deterministic = bool(deterministic)
        self._initialized = False
        self._hidden_state = None

    def reset(self, env):
        self.observer.reset(env)
        self._initialized = True
        self._hidden_state = None

    def choose_action(self, env):
        if not self._initialized:
            self.reset(env)

        observation = self.observer.observe(env)
        macro_mask = build_macro_mask(env)
        micro_masks = build_micro_action_masks(env)
        privileged = compute_privileged_signals(env)["privileged_scalars"]
        output = self.policy.act(
            observation=observation,
            macro_mask=macro_mask,
            micro_masks=micro_masks,
            hidden_state=self._hidden_state,
            privileged_scalars=privileged,
            deterministic=self.deterministic,
        )
        self._hidden_state = output["hidden_state"]
        return resolve_hierarchical_action(
            env,
            macro_index=output["macro_index"],
            micro_index=output["micro_index"],
        )
