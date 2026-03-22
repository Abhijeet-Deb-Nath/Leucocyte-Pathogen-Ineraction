"""RL interface utilities for macrophage-only learning.

This module keeps the original simulator API intact while adding a Gymnasium-compatible
adapter that always passes explicit macrophage actions into the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Optional

import numpy as np

import config
from simulator.environment import Environment

_gym_module = None
_spaces_module = None
_GYM_BASE = object
_torch_module = None
_nn_module = None
_base_features_extractor = None
_multi_input_actor_critic_policy_cls = None
_categorical_distribution_cls = None
_recurrent_multi_input_actor_critic_policy_cls = None
_rnn_states_cls = None


def _load_gym_modules():
    global _gym_module, _spaces_module
    if _gym_module is not None and _spaces_module is not None:
        return _gym_module, _spaces_module

    try:
        _gym_module = importlib.import_module("gymnasium")
        _spaces_module = importlib.import_module("gymnasium.spaces")
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("gymnasium is required to use MacrophageGymEnv") from exc

    return _gym_module, _spaces_module


def _load_torch_modules():
    global _torch_module
    global _nn_module
    global _base_features_extractor
    global _multi_input_actor_critic_policy_cls
    global _categorical_distribution_cls
    if (
        _torch_module is not None
        and _nn_module is not None
        and _base_features_extractor is not None
        and _multi_input_actor_critic_policy_cls is not None
        and _categorical_distribution_cls is not None
    ):
        return (
            _torch_module,
            _nn_module,
            _base_features_extractor,
            _multi_input_actor_critic_policy_cls,
            _categorical_distribution_cls,
        )

    try:
        _torch_module = importlib.import_module("torch")
        _nn_module = importlib.import_module("torch.nn")
        torch_layers = importlib.import_module("stable_baselines3.common.torch_layers")
        policies = importlib.import_module("stable_baselines3.common.policies")
        distributions = importlib.import_module("stable_baselines3.common.distributions")
        _base_features_extractor = torch_layers.BaseFeaturesExtractor
        _multi_input_actor_critic_policy_cls = policies.MultiInputActorCriticPolicy
        _categorical_distribution_cls = distributions.CategoricalDistribution
    except ImportError as exc:
        raise ImportError("torch and stable-baselines3 are required for custom RL features") from exc

    return (
        _torch_module,
        _nn_module,
        _base_features_extractor,
        _multi_input_actor_critic_policy_cls,
        _categorical_distribution_cls,
    )


try:
    _GYM_BASE = importlib.import_module("gymnasium").Env
except ImportError:
    _GYM_BASE = object


MACROPHAGE_ACTIONS = [
    ("move", (-1, 0)),
    ("move", (1, 0)),
    ("move", (0, -1)),
    ("move", (0, 1)),
    ("move", (0, 0)),
    ("attack", None),
    ("signal_low", None),
    ("signal_medium", None),
    ("signal_high", None),
]


@dataclass
class BeliefState:
    """Belief-style memory features for partial observability."""

    last_seen_bacteria_compartment: Optional[int] = None
    strongest_recent_chemokine_compartment: Optional[int] = None
    visited_compartments: Optional[np.ndarray] = None


try:
    (
        _torch,
        _nn,
        _BaseFeaturesExtractor,
        _MultiInputActorCriticPolicy,
        _CategoricalDistribution,
    ) = _load_torch_modules()
except ImportError:
    _torch = None
    _nn = None
    _MultiInputActorCriticPolicy = None
    _CategoricalDistribution = None

    class _MissingBaseFeaturesExtractor:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch and stable-baselines3 are required for custom RL features")

    _BaseFeaturesExtractor = _MissingBaseFeaturesExtractor

try:
    recurrent_policies = importlib.import_module("sb3_contrib.common.recurrent.policies")
    recurrent_type_aliases = importlib.import_module("sb3_contrib.common.recurrent.type_aliases")
    _RecurrentMultiInputActorCriticPolicy = recurrent_policies.RecurrentMultiInputActorCriticPolicy
    _RNNStates = recurrent_type_aliases.RNNStates
except ImportError:
    _RecurrentMultiInputActorCriticPolicy = None
    _RNNStates = None


def build_action_mask(env):
    mask = np.zeros(len(MACROPHAGE_ACTIONS), dtype=np.float32)
    valid_actions = set(env.get_macrophage_actions())
    for idx, action in enumerate(MACROPHAGE_ACTIONS):
        if action in valid_actions:
            mask[idx] = 1.0
    return mask


class MacrophageCombinedExtractor(_BaseFeaturesExtractor):
    """Compact CNN for the spatial grid plus a small MLP for scalar features."""

    def __init__(self, observation_space, grid_feature_dim=192, scalar_hidden_dim=64):
        super().__init__(observation_space, features_dim=1)

        grid_space = observation_space["grid"]
        scalar_space = observation_space["scalars"]
        input_channels = int(grid_space.shape[0])
        scalar_dim = int(scalar_space.shape[0])

        self.grid_extractor = _nn.Sequential(
            _nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            _nn.ReLU(),
            _nn.Conv2d(32, 64, kernel_size=3, padding=1),
            _nn.ReLU(),
            _nn.MaxPool2d(kernel_size=2),
            _nn.Conv2d(64, 64, kernel_size=3, padding=1),
            _nn.ReLU(),
            _nn.AdaptiveAvgPool2d((3, 3)),
            _nn.Flatten(),
            _nn.Linear(64 * 3 * 3, int(grid_feature_dim)),
            _nn.ReLU(),
        )
        self.scalar_extractor = _nn.Sequential(
            _nn.Linear(scalar_dim, int(scalar_hidden_dim)),
            _nn.ReLU(),
        )

        self._features_dim = int(grid_feature_dim) + int(scalar_hidden_dim)

    def forward(self, observations):
        grid_features = self.grid_extractor(observations["grid"])
        scalar_features = self.scalar_extractor(observations["scalars"])
        return _torch.cat((grid_features, scalar_features), dim=1)


if _MultiInputActorCriticPolicy is None:

    class MaskedMacrophagePolicy:
        def __init__(self, *args, **kwargs):
            raise ImportError("stable-baselines3 is required to use MaskedMacrophagePolicy")

else:

    class MaskedMacrophagePolicy(_MultiInputActorCriticPolicy):
        """SB3 multi-input policy that masks invalid discrete actions before sampling."""

        def _masked_distribution(self, latent_pi, obs):
            if not isinstance(self.action_dist, _CategoricalDistribution):
                return super()._get_action_dist_from_latent(latent_pi)

            action_logits = self.action_net(latent_pi)
            if isinstance(obs, dict) and "action_mask" in obs:
                action_mask = obs["action_mask"].float()
                if action_mask.ndim == 1:
                    action_mask = action_mask.unsqueeze(0)
                action_logits = action_logits.masked_fill(
                    action_mask <= 0.0,
                    _torch.finfo(action_logits.dtype).min,
                )
            return self.action_dist.proba_distribution(action_logits=action_logits)

        def forward(self, obs, deterministic=False):
            features = self.extract_features(obs)
            if self.share_features_extractor:
                latent_pi, latent_vf = self.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)

            values = self.value_net(latent_vf)
            distribution = self._masked_distribution(latent_pi, obs)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            actions = actions.reshape((-1, *self.action_space.shape))
            return actions, values, log_prob

        def evaluate_actions(self, obs, actions):
            features = self.extract_features(obs)
            if self.share_features_extractor:
                latent_pi, latent_vf = self.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)

            distribution = self._masked_distribution(latent_pi, obs)
            log_prob = distribution.log_prob(actions)
            values = self.value_net(latent_vf)
            entropy = distribution.entropy()
            return values, log_prob, entropy

        def get_distribution(self, obs):
            features = super().extract_features(obs, self.pi_features_extractor)
            latent_pi = self.mlp_extractor.forward_actor(features)
            return self._masked_distribution(latent_pi, obs)


if _RecurrentMultiInputActorCriticPolicy is None:

    class MaskedRecurrentMacrophagePolicy:
        def __init__(self, *args, **kwargs):
            raise ImportError("sb3-contrib is required to use MaskedRecurrentMacrophagePolicy")

else:

    class MaskedRecurrentMacrophagePolicy(_RecurrentMultiInputActorCriticPolicy):
        """Recurrent multi-input policy that masks invalid discrete actions before sampling."""

        def _masked_distribution(self, latent_pi, obs):
            if not isinstance(self.action_dist, _CategoricalDistribution):
                return self._get_action_dist_from_latent(latent_pi)

            action_logits = self.action_net(latent_pi)
            if isinstance(obs, dict) and "action_mask" in obs:
                action_mask = obs["action_mask"].float()
                if action_mask.ndim == 1:
                    action_mask = action_mask.unsqueeze(0)
                action_logits = action_logits.masked_fill(
                    action_mask <= 0.0,
                    _torch.finfo(action_logits.dtype).min,
                )
            return self.action_dist.proba_distribution(action_logits=action_logits)

        def forward(self, obs, lstm_states, episode_starts, deterministic=False):
            features = self.extract_features(obs)
            if self.share_features_extractor:
                pi_features = vf_features = features
            else:
                pi_features, vf_features = features

            latent_pi, lstm_states_pi = self._process_sequence(
                pi_features,
                lstm_states.pi,
                episode_starts,
                self.lstm_actor,
            )
            if self.lstm_critic is not None:
                latent_vf, lstm_states_vf = self._process_sequence(
                    vf_features,
                    lstm_states.vf,
                    episode_starts,
                    self.lstm_critic,
                )
            elif self.shared_lstm:
                latent_vf = latent_pi.detach()
                lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
            else:
                latent_vf = self.critic(vf_features)
                lstm_states_vf = lstm_states_pi

            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)

            values = self.value_net(latent_vf)
            distribution = self._masked_distribution(latent_pi, obs)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            return actions, values, log_prob, _RNNStates(lstm_states_pi, lstm_states_vf)

        def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
            features = self.extract_features(obs)
            if self.share_features_extractor:
                pi_features = vf_features = features
            else:
                pi_features, vf_features = features

            latent_pi, _ = self._process_sequence(
                pi_features,
                lstm_states.pi,
                episode_starts,
                self.lstm_actor,
            )
            if self.lstm_critic is not None:
                latent_vf, _ = self._process_sequence(
                    vf_features,
                    lstm_states.vf,
                    episode_starts,
                    self.lstm_critic,
                )
            elif self.shared_lstm:
                latent_vf = latent_pi.detach()
            else:
                latent_vf = self.critic(vf_features)

            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)

            distribution = self._masked_distribution(latent_pi, obs)
            log_prob = distribution.log_prob(actions)
            values = self.value_net(latent_vf)
            return values, log_prob, distribution.entropy()

        def get_distribution(self, obs, lstm_states, episode_starts):
            features = self.extract_features(obs)
            if self.share_features_extractor:
                pi_features = features
            else:
                pi_features, _ = features
            latent_pi, lstm_states = self._process_sequence(
                pi_features,
                lstm_states,
                episode_starts,
                self.lstm_actor,
            )
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            return self._masked_distribution(latent_pi, obs), lstm_states


class MacrophageObservationBuilder:
    """Build full-state or partial-state observations for macrophage policy learning."""

    def __init__(
        self,
        mode="partial_state",
        partial_radius=None,
        include_belief_features=None,
        include_action_mask=False,
    ):
        if mode not in {"full_state", "partial_state"}:
            raise ValueError("mode must be 'full_state' or 'partial_state'")
        self.mode = mode
        self.partial_radius = (
            config.RL_PARTIAL_OBS_RADIUS if partial_radius is None else int(partial_radius)
        )
        self.include_belief_features = (
            config.RL_INCLUDE_BELIEF_FEATURES
            if include_belief_features is None
            else bool(include_belief_features)
        )
        self.include_action_mask = bool(include_action_mask)
        self.belief = BeliefState()

    def reset(self, env):
        self.belief = BeliefState(
            last_seen_bacteria_compartment=None,
            strongest_recent_chemokine_compartment=None,
            visited_compartments=np.zeros(config.N_COMPARTMENTS, dtype=np.float32),
        )
        self._update_belief(env)

    def observation_space(self, env):
        _, spaces = _load_gym_modules()

        if self.mode == "full_state":
            grid_shape = (6, env.height, env.width)
        else:
            size = 2 * self.partial_radius + 1
            grid_shape = (6, size, size)

        scalar_len = 9
        if self.include_belief_features:
            scalar_len += config.N_COMPARTMENTS * 3

        obs_spaces = {
            "grid": spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32),
            "scalars": spaces.Box(low=0.0, high=1.0, shape=(scalar_len,), dtype=np.float32),
        }
        if self.include_action_mask:
            obs_spaces["action_mask"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(MACROPHAGE_ACTIONS),),
                dtype=np.float32,
            )
        return spaces.Dict(obs_spaces)

    def observe(self, env):
        self._update_belief(env)

        if self.mode == "full_state":
            grid = self._build_full_grid(env)
            visible_bacteria = len(env.bacteria)
            visible_neutrophils = len(env.neutrophils)
            local_peak_chem = float(env.chemokine.max())
            chem_scale = max(config.NEUTROPHIL_RECRUITMENT_THRESHOLD * 1.5, 1e-6)
            nearest_bacteria_dx, nearest_bacteria_dy = self._nearest_visible_bacteria_offset(
                env,
                bacteria=env.bacteria,
                radius=max(env.width, env.height),
            )
            chem_dx, chem_dy = self._strongest_local_chem_offset(
                env,
                radius=max(env.width, env.height),
                min_peak=0.0,
            )
        else:
            grid, _, _ = self._build_partial_grid(env)
            visible_bacteria = len(env.get_local_bacteria(env.macrophage.position, self.partial_radius))
            visible_neutrophils = len(env.get_local_neutrophils(env.macrophage.position, self.partial_radius))
            local_peak_chem = self._local_chemokine_peak(env)
            chem_scale = max(config.MACROPHAGE_CHEM_SENSE_SCALE, 1e-6)
            nearest_bacteria_dx, nearest_bacteria_dy = self._nearest_visible_bacteria_offset(env)
            chem_dx, chem_dy = self._strongest_local_chem_offset(
                env,
                radius=self.partial_radius,
                min_peak=max(0.02, config.BACTERIA_ALARM_CHEMOKINE_RELEASE * 0.25),
            )

        scalars = [
            float(env.macrophage.health / max(1, config.MACROPHAGE_MAX_HEALTH)),
            float(env.macrophage.signal_cooldown / max(1, config.MACROPHAGE_SIGNAL_COOLDOWN)),
            float(min(visible_bacteria, config.INITIAL_BACTERIA_COUNT * 4) / max(1, config.INITIAL_BACTERIA_COUNT * 4)),
            float(visible_neutrophils / max(1, config.MAX_NEUTROPHIL_POOL)),
            float(
                np.clip(
                    local_peak_chem / chem_scale,
                    0.0,
                    1.0,
                )
            ),
            self._offset_to_unit(nearest_bacteria_dx, max(1, self.partial_radius if self.mode == "partial_state" else env.width)),
            self._offset_to_unit(nearest_bacteria_dy, max(1, self.partial_radius if self.mode == "partial_state" else env.height)),
            self._offset_to_unit(chem_dx, max(1, self.partial_radius if self.mode == "partial_state" else env.width)),
            self._offset_to_unit(chem_dy, max(1, self.partial_radius if self.mode == "partial_state" else env.height)),
        ]

        if self.include_belief_features:
            scalars.extend(self._belief_vector())

        observation = {
            "grid": grid.astype(np.float32),
            "scalars": np.asarray(scalars, dtype=np.float32),
        }
        if self.include_action_mask:
            observation["action_mask"] = build_action_mask(env)
        return observation

    def _local_chemokine_peak(self, env):
        mx, my = env.macrophage.position
        peak = 0.0
        for y in range(max(0, my - self.partial_radius), min(env.height, my + self.partial_radius + 1)):
            for x in range(max(0, mx - self.partial_radius), min(env.width, mx + self.partial_radius + 1)):
                if abs(x - mx) + abs(y - my) > self.partial_radius:
                    continue
                peak = max(peak, float(env.chemokine[y, x]))
        return peak

    def _nearest_visible_bacteria_offset(self, env, bacteria=None, radius=None):
        if bacteria is None:
            bacteria = env.get_local_bacteria(env.macrophage.position, self.partial_radius)
        if not bacteria:
            return 0.0, 0.0
        mx, my = env.macrophage.position
        nearest = min(
            bacteria,
            key=lambda b: abs(b.position[0] - mx) + abs(b.position[1] - my),
        )
        dx = float(nearest.position[0] - mx)
        dy = float(nearest.position[1] - my)
        return dx, dy

    def _strongest_local_chem_offset(self, env, radius, min_peak):
        mx, my = env.macrophage.position
        peak_value = float(min_peak)
        best_dx = 0.0
        best_dy = 0.0
        for y in range(max(0, my - radius), min(env.height, my + radius + 1)):
            for x in range(max(0, mx - radius), min(env.width, mx + radius + 1)):
                dx = x - mx
                dy = y - my
                if abs(dx) + abs(dy) > radius:
                    continue
                value = float(env.chemokine[y, x])
                if value > peak_value:
                    peak_value = value
                    best_dx = float(dx)
                    best_dy = float(dy)
        return best_dx, best_dy

    def _offset_to_unit(self, value, scale):
        if scale <= 0:
            return 0.5
        normalized = np.clip(float(value) / float(scale), -1.0, 1.0)
        return float(0.5 + 0.5 * normalized)

    def _build_full_grid(self, env):
        blocked = np.zeros((env.height, env.width), dtype=np.float32)
        for x, y in env.blocked_tiles:
            blocked[y, x] = 1.0

        nutrient = np.clip(env.nutrients / max(config.PATCH_NUTRIENT_CAPACITY, 1e-6), 0.0, 1.0).astype(
            np.float32
        )
        chem = np.clip(
            env.chemokine / max(config.NEUTROPHIL_RECRUITMENT_THRESHOLD * 1.5, 1e-6),
            0.0,
            1.0,
        ).astype(np.float32)

        bacteria = np.zeros((env.height, env.width), dtype=np.float32)
        for b in env.bacteria:
            bacteria[b.position[1], b.position[0]] = 1.0

        neutrophils = np.zeros((env.height, env.width), dtype=np.float32)
        for n in env.neutrophils:
            neutrophils[n.position[1], n.position[0]] = 1.0

        macrophage = np.zeros((env.height, env.width), dtype=np.float32)
        mx, my = env.macrophage.position
        macrophage[my, mx] = 1.0

        return np.stack([blocked, nutrient, chem, bacteria, neutrophils, macrophage], axis=0)

    def _build_partial_grid(self, env):
        r = self.partial_radius
        size = 2 * r + 1
        mx, my = env.macrophage.position

        blocked = np.ones((size, size), dtype=np.float32)
        nutrient = np.zeros((size, size), dtype=np.float32)
        chem = np.zeros((size, size), dtype=np.float32)
        bacteria = np.zeros((size, size), dtype=np.float32)
        neutrophils = np.zeros((size, size), dtype=np.float32)
        macrophage = np.zeros((size, size), dtype=np.float32)
        chem_scale = max(config.MACROPHAGE_CHEM_SENSE_SCALE, 1e-6)

        for ly in range(size):
            for lx in range(size):
                gx = mx + (lx - r)
                gy = my + (ly - r)
                if 0 <= gx < env.width and 0 <= gy < env.height:
                    blocked[ly, lx] = 0.0 if env._is_walkable((gx, gy)) else 1.0
                    nutrient[ly, lx] = np.clip(
                        env.nutrients[gy, gx] / max(config.PATCH_NUTRIENT_CAPACITY, 1e-6),
                        0.0,
                        1.0,
                    )
                    chem[ly, lx] = np.clip(
                        env.chemokine[gy, gx] / chem_scale,
                        0.0,
                        1.0,
                    )

        for b in env.get_local_bacteria(env.macrophage.position, r):
            bx, by = b.position
            lx = bx - mx + r
            ly = by - my + r
            if 0 <= lx < size and 0 <= ly < size:
                bacteria[ly, lx] = 1.0

        for n in env.get_local_neutrophils(env.macrophage.position, r):
            nx, ny = n.position
            lx = nx - mx + r
            ly = ny - my + r
            if 0 <= lx < size and 0 <= ly < size:
                neutrophils[ly, lx] = 1.0

        macrophage[r, r] = 1.0
        macro_local_x = r / max(1, size - 1)
        macro_local_y = r / max(1, size - 1)

        grid = np.stack([blocked, nutrient, chem, bacteria, neutrophils, macrophage], axis=0)
        return grid, macro_local_x, macro_local_y

    def _update_belief(self, env):
        mx, my = env.macrophage.position
        current_compartment = int(env.compartment_map[my, mx])
        if self.belief.visited_compartments is not None and current_compartment >= 0:
            self.belief.visited_compartments[current_compartment] = 1.0

        if self.mode == "full_state":
            visible_bacteria = env.bacteria
        else:
            visible_bacteria = env.get_local_bacteria(env.macrophage.position, self.partial_radius)

        if visible_bacteria:
            nearest = min(visible_bacteria, key=lambda b: abs(b.position[0] - mx) + abs(b.position[1] - my))
            cid = int(env.compartment_map[nearest.position[1], nearest.position[0]])
            if cid >= 0:
                self.belief.last_seen_bacteria_compartment = cid

        if self.mode == "full_state":
            chem = env.chemokine
            peak_index = int(chem.argmax())
            py, px = divmod(peak_index, env.width)
        else:
            r = self.partial_radius
            x0, x1 = max(0, mx - r), min(env.width, mx + r + 1)
            y0, y1 = max(0, my - r), min(env.height, my + r + 1)
            local = env.chemokine[y0:y1, x0:x1]
            if local.size == 0:
                return
            local_peak = int(local.argmax())
            ly, lx = divmod(local_peak, local.shape[1])
            px, py = x0 + lx, y0 + ly

        chem_compartment = int(env.compartment_map[py, px])
        if chem_compartment >= 0:
            self.belief.strongest_recent_chemokine_compartment = chem_compartment

    def _belief_vector(self):
        bacteria_comp = np.zeros(config.N_COMPARTMENTS, dtype=np.float32)
        chem_comp = np.zeros(config.N_COMPARTMENTS, dtype=np.float32)
        visited = np.zeros(config.N_COMPARTMENTS, dtype=np.float32)

        if self.belief.last_seen_bacteria_compartment is not None:
            bacteria_comp[self.belief.last_seen_bacteria_compartment] = 1.0
        if self.belief.strongest_recent_chemokine_compartment is not None:
            chem_comp[self.belief.strongest_recent_chemokine_compartment] = 1.0
        if self.belief.visited_compartments is not None:
            visited = self.belief.visited_compartments.astype(np.float32)

        return np.concatenate([bacteria_comp, chem_comp, visited]).tolist()


def action_from_index(action_idx):
    idx = int(action_idx)
    idx = max(0, min(len(MACROPHAGE_ACTIONS) - 1, idx))
    return MACROPHAGE_ACTIONS[idx]


def sanitize_action(env, candidate_action):
    valid_actions = env.get_macrophage_actions()
    valid_set = set(valid_actions)
    if candidate_action in valid_set:
        return candidate_action, False

    stay_action = ("move", (0, 0))
    if stay_action in valid_set:
        return stay_action, True

    if valid_actions:
        return valid_actions[0], True

    return stay_action, True


class MacrophageGymEnv(_GYM_BASE):
    """Gymnasium-compatible RL wrapper that controls only the macrophage."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        observation_mode="partial_state",
        seed=None,
        include_belief_features=None,
        partial_radius=None,
        include_action_mask=False,
        guided_start_prob=0.0,
        env_id=0,
        deterministic_reset_stream=False,
    ):
        _, spaces = _load_gym_modules()
        self.base_seed = None if seed is None else int(seed)
        self.env_id = int(env_id)
        self.episode_index = 0
        self.deterministic_reset_stream = bool(deterministic_reset_stream)
        self.guided_start_prob = float(np.clip(guided_start_prob, 0.0, 1.0))
        self.env = None
        self.observer = MacrophageObservationBuilder(
            mode=observation_mode,
            partial_radius=partial_radius,
            include_belief_features=include_belief_features,
            include_action_mask=include_action_mask,
        )
        self.action_space = spaces.Discrete(len(MACROPHAGE_ACTIONS))

        bootstrap_seed = self._derive_episode_seed(explicit_seed=self.base_seed)
        bootstrap_env = self._build_env(bootstrap_seed)
        self.observer.reset(bootstrap_env)
        self.observation_space = self.observer.observation_space(bootstrap_env)
        self._last_utility = bootstrap_env.compute_host_utility()
        self.env = bootstrap_env
        self._last_macrophage_position = None
        self._visited_positions = {bootstrap_env.macrophage.position}
        self.episode_index += 1

    def _derive_episode_seed(self, explicit_seed=None):
        if explicit_seed is not None:
            return int(explicit_seed)

        if self.base_seed is None:
            if self.deterministic_reset_stream:
                return int(self.env_id * 1_000_003 + self.episode_index)
            # Unseeded training mode should vary episodes by default.
            return int(np.random.SeedSequence().generate_state(1)[0])

        return int(self.base_seed + self.env_id * 1_000_003 + self.episode_index)

    def _build_env(self, episode_seed):
        start_mode = "surface_random"
        if self.guided_start_prob > 0:
            curriculum_rng = np.random.default_rng(episode_seed + 97_531)
            if float(curriculum_rng.random()) < self.guided_start_prob:
                start_mode = "near_bacteria"
        return Environment(seed=episode_seed, macrophage_start_mode=start_mode)

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.base_seed = int(seed)
            self.episode_index = 0
            episode_seed = self._derive_episode_seed(explicit_seed=seed)
        else:
            episode_seed = self._derive_episode_seed()

        self.env = self._build_env(episode_seed)
        self.observer.reset(self.env)
        self._last_utility = self.env.compute_host_utility()
        self._last_macrophage_position = None
        self._visited_positions = {self.env.macrophage.position}
        observation = self.observer.observe(self.env)
        info = {"episode_seed": episode_seed, "episode_index": self.episode_index}
        self.episode_index += 1
        return observation, info

    def step(self, action):
        action_tuple = action_from_index(action)
        current_pos = self.env.macrophage.position
        killed_before = int(self.env.killed_bacteria)
        visible_bacteria_before = self.env.get_local_bacteria(
            current_pos, config.MACROPHAGE_SENSE_RADIUS
        )
        visible_neutrophils_before = self.env.get_local_neutrophils(
            current_pos, config.MACROPHAGE_SENSE_RADIUS
        )
        nearest_visible_target = None
        nearest_visible_distance = None
        if visible_bacteria_before:
            nearest_visible_target = min(
                visible_bacteria_before,
                key=lambda b: abs(b.position[0] - current_pos[0]) + abs(b.position[1] - current_pos[1]),
            ).position
            nearest_visible_distance = abs(nearest_visible_target[0] - current_pos[0]) + abs(
                nearest_visible_target[1] - current_pos[1]
            )
        adjacent_before = bool(self.env._adjacent_bacteria(current_pos))
        local_peak_before = (
            self.observer._local_chemokine_peak(self.env)
            if self.observer.mode == "partial_state"
            else float(self.env.chemokine.max())
        )
        safe_action, fallback_used = sanitize_action(self.env, action_tuple)

        # RL path always supplies explicit macrophage actions and never calls step(None).
        self.env.step(macrophage_action=safe_action)
        new_pos = self.env.macrophage.position
        killed_after = int(self.env.killed_bacteria)
        adjacent_after = bool(self.env._adjacent_bacteria(new_pos))

        observation = self.observer.observe(self.env)
        utility = self.env.compute_host_utility()
        reward = utility - self._last_utility
        if fallback_used:
            reward -= config.RL_INVALID_ACTION_PENALTY
        if safe_action[0] == "attack" and adjacent_before:
            reward += config.RL_VISIBLE_ATTACK_BONUS
            if killed_after > killed_before:
                reward += (killed_after - killed_before) * config.RL_KILL_BONUS
        else:
            if adjacent_before:
                reward -= config.RL_MISSED_ADJACENT_ATTACK_PENALTY

            if safe_action[0] == "move" and nearest_visible_target is not None and nearest_visible_distance is not None:
                dx, dy = safe_action[1]
                predicted_pos = (current_pos[0] + dx, current_pos[1] + dy)
                predicted_distance = abs(nearest_visible_target[0] - predicted_pos[0]) + abs(
                    nearest_visible_target[1] - predicted_pos[1]
                )
                distance_improvement = nearest_visible_distance - predicted_distance
                if distance_improvement > 0:
                    reward += distance_improvement * config.RL_APPROACH_VISIBLE_BACTERIA_BONUS
                    if adjacent_after or predicted_distance <= 1:
                        reward += config.RL_CLOSE_ENGAGEMENT_BONUS
                elif distance_improvement < 0:
                    reward -= abs(distance_improvement) * config.RL_RETREAT_FROM_VISIBLE_BACTERIA_PENALTY
                else:
                    reward -= 0.5 * config.RL_VISIBLE_BACTERIA_DEFER_PENALTY
            elif safe_action == ("move", (0, 0)):
                if visible_bacteria_before:
                    reward -= config.RL_VISIBLE_BACTERIA_DEFER_PENALTY
                elif local_peak_before < config.MACROPHAGE_SIGNAL_LOCAL_CHEMOKINE_THRESHOLD:
                    reward -= config.RL_IDLE_WITHOUT_CUE_PENALTY
            elif safe_action[0] in {"signal", "signal_low", "signal_medium", "signal_high"}:
                local_pressure = len(visible_bacteria_before) - len(visible_neutrophils_before)
                if adjacent_before:
                    reward -= config.RL_SIGNAL_WHILE_ADJACENT_BACTERIA_PENALTY
                elif visible_bacteria_before:
                    if (
                        local_pressure >= config.HEURISTIC_SIGNAL_PRESSURE_MEDIUM
                        and nearest_visible_distance is not None
                        and nearest_visible_distance <= config.HEURISTIC_SIGNAL_PROXIMITY_RADIUS
                    ):
                        reward += config.RL_CONTEXTUAL_SIGNAL_BONUS
                    else:
                        reward -= config.RL_SIGNAL_WHILE_VISIBLE_BACTERIA_PENALTY
                elif local_peak_before < config.NEUTROPHIL_RECRUITMENT_THRESHOLD * 0.4:
                    reward -= config.RL_BLIND_SIGNAL_PENALTY
                elif nearest_visible_distance is not None and nearest_visible_distance > 2:
                    reward -= config.RL_DISTANT_SIGNAL_PENALTY
            elif visible_bacteria_before:
                reward -= config.RL_VISIBLE_BACTERIA_DEFER_PENALTY

        if safe_action[0] == "move":
            if self._last_macrophage_position is not None and new_pos == self._last_macrophage_position:
                reward -= config.RL_BACKTRACK_PENALTY
            if new_pos not in self._visited_positions:
                reward += config.RL_NEW_TILE_BONUS
            self._visited_positions.add(new_pos)
            self._last_macrophage_position = current_pos

        if self.env.done:
            reward += 25.0 if self.env.winner == "Host" else -25.0

        self._last_utility = utility
        info = {
            "winner": self.env.winner,
            "host_utility": utility,
            "tissue_damage": self.env.tissue_damage,
            "episode_step": self.env.step_count,
            "fallback_action_used": fallback_used,
            "chosen_action": safe_action,
            "recruited_neutrophils": self.env.recruited_neutrophils_total,
        }
        terminated = bool(self.env.done)
        truncated = False
        return observation, float(reward), terminated, truncated, info

    def render(self):
        return self.env.get_grid() if self.env is not None else None
