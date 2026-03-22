"""Stable-Baselines3 PPO inference agent for macrophage control."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from agents.hierarchical_agent import HierarchicalMacrophageAgent
from agents.hierarchical_policy import load_hierarchical_checkpoint
from simulator.rl_interface import (
    LEGACY_MACROPHAGE_ACTIONS,
    MACROPHAGE_ACTIONS,
    MacrophageObservationBuilder,
    action_from_index,
    is_action_available,
    resolve_policy_action,
    sanitize_action,
)


class RLMacrophageAgent:
    """Loads a PPO checkpoint and emits environment-compatible macrophage actions."""

    def __init__(
        self,
        model_path,
        observation_mode="partial_state",
        include_belief_features=None,
        partial_radius=None,
        deterministic=True,
    ):
        model_path = Path(model_path)
        resolved_model_path = model_path
        if not resolved_model_path.exists():
            for suffix in (".pt", ".zip"):
                candidate = resolved_model_path.with_suffix(suffix)
                if candidate.exists():
                    resolved_model_path = candidate
                    break
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"RL model not found: {model_path}")

        self._delegate = None
        if resolved_model_path.suffix == ".pt":
            payload = load_hierarchical_checkpoint(resolved_model_path, map_location="cpu")
            if payload.get("format") == "hierarchical_macrophage_v1":
                self._delegate = HierarchicalMacrophageAgent(
                    model_path=resolved_model_path,
                    observation_mode=observation_mode,
                    deterministic=deterministic,
                )
                return

        if observation_mode != "partial_state":
            raise ValueError("Only partial_state learned-controller inference is supported")
        try:
            importlib.import_module("stable_baselines3")
        except ImportError as exc:
            raise ImportError("stable-baselines3 is required for RL policy inference") from exc

        self.model, self._is_recurrent = self._load_model(resolved_model_path)
        self._action_catalog = self._detect_action_catalog()
        model_obs_space = getattr(self.model, "observation_space", None)
        include_action_mask = bool(
            getattr(model_obs_space, "spaces", None)
            and "action_mask" in model_obs_space.spaces
        )
        self.observer = MacrophageObservationBuilder(
            mode=observation_mode,
            include_belief_features=include_belief_features,
            partial_radius=partial_radius,
            include_action_mask=include_action_mask,
            action_catalog=self._action_catalog,
        )
        self._uses_action_mask = include_action_mask
        self.deterministic = deterministic
        self._initialized = False
        self._recurrent_state = None
        self._episode_start = np.array([True], dtype=np.bool_)

    def _load_model(self, resolved_model_path):
        load_attempts = []

        try:
            recurrent_mod = importlib.import_module("sb3_contrib")
            load_attempts.append(("recurrent", recurrent_mod.RecurrentPPO))
        except ImportError:
            pass

        ppo_module = importlib.import_module("stable_baselines3")
        load_attempts.append(("feedforward", ppo_module.PPO))

        errors = []
        for model_kind, loader in load_attempts:
            try:
                return loader.load(str(resolved_model_path)), model_kind == "recurrent"
            except Exception as exc:  # pragma: no cover - depends on runtime checkpoint format
                errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")

        joined = "; ".join(errors) if errors else "no available loader"
        raise RuntimeError(f"Unable to load RL model from {resolved_model_path}: {joined}")

    def _detect_action_catalog(self):
        action_n = int(getattr(self.model.action_space, "n", len(MACROPHAGE_ACTIONS)))
        if action_n == len(MACROPHAGE_ACTIONS):
            return list(MACROPHAGE_ACTIONS)
        if action_n == len(LEGACY_MACROPHAGE_ACTIONS):
            return list(LEGACY_MACROPHAGE_ACTIONS)
        raise ValueError(f"Unsupported RL action-space size: {action_n}")

    def reset(self, env):
        if self._delegate is not None:
            self._delegate.reset(env)
            return
        self.observer.reset(env)
        self._initialized = True
        self._recurrent_state = None
        self._episode_start = np.array([True], dtype=np.bool_)

    def choose_action(self, env):
        if self._delegate is not None:
            return self._delegate.choose_action(env)
        if not self._initialized:
            self.reset(env)

        observation = self.observer.observe(env)
        if self._is_recurrent:
            action_idx, self._recurrent_state = self.model.predict(
                observation,
                state=self._recurrent_state,
                episode_start=self._episode_start,
                deterministic=self.deterministic,
            )
            self._episode_start = np.array([False], dtype=np.bool_)
            action = action_from_index(int(np.asarray(action_idx).item()), self._action_catalog)
        elif self.deterministic and not self._uses_action_mask:
            action = self._predict_best_valid_action(observation, env)
        else:
            action_idx, _ = self.model.predict(observation, deterministic=self.deterministic)
            action = action_from_index(int(action_idx), self._action_catalog)
        resolved_action = resolve_policy_action(env, action)
        safe_action, _ = sanitize_action(env, resolved_action)
        return safe_action

    def _predict_best_valid_action(self, observation, env):
        torch = importlib.import_module("torch")
        obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.detach().cpu().numpy().reshape(-1)

        best_idx = None
        best_prob = -1.0
        for idx, action in enumerate(self._action_catalog):
            if is_action_available(env, action) and probs[idx] > best_prob:
                best_idx = idx
                best_prob = float(probs[idx])

        if best_idx is None:
            return ("move", (0, 0))
        return action_from_index(best_idx, self._action_catalog)
