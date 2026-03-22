"""Train PPO for macrophage-only control using the RL-safe Gym wrapper."""

import argparse
import importlib
from pathlib import Path
import shutil

import numpy as np

import config
from agents.heuristic import HeuristicAgent
from simulator.rl_interface import (
    MACROPHAGE_ACTIONS,
    MacrophageCombinedExtractor,
    MacrophageGymEnv,
    MaskedMacrophagePolicy,
    MaskedRecurrentMacrophagePolicy,
)


def _make_env(
    obs_mode,
    seed,
    env_id,
    include_belief_features,
    partial_radius,
    include_action_mask,
    guided_start_prob,
    deterministic_reset_stream,
):
    def _factory():
        env = MacrophageGymEnv(
            observation_mode=obs_mode,
            seed=seed,
            include_belief_features=include_belief_features,
            partial_radius=partial_radius,
            include_action_mask=include_action_mask,
            guided_start_prob=guided_start_prob,
            env_id=env_id,
            deterministic_reset_stream=deterministic_reset_stream,
        )
        return env

    return _factory


def _collect_expert_dataset(
    obs_mode,
    seed,
    include_belief_features,
    partial_radius,
    use_action_mask,
    guided_start_prob,
    expert_steps,
):
    expert_steps = max(0, int(expert_steps))
    if expert_steps <= 0:
        return [], np.asarray([], dtype=np.int64), []

    env = MacrophageGymEnv(
        observation_mode=obs_mode,
        seed=seed,
        include_belief_features=include_belief_features,
        partial_radius=partial_radius,
        include_action_mask=use_action_mask,
        guided_start_prob=guided_start_prob,
        env_id=99,
        deterministic_reset_stream=False,
    )
    expert = HeuristicAgent(use_memory=False, stochastic_patrol=False)

    observations = []
    actions = []
    metadata = []

    obs, _ = env.reset(seed=seed)
    expert.reset(env.env)
    while len(actions) < expert_steps:
        action = expert.choose_action(env.env)
        current_pos = env.env.macrophage.position
        visible_bacteria = env.env.get_local_bacteria(current_pos, config.MACROPHAGE_SENSE_RADIUS)
        visible_neutrophils = env.env.get_local_neutrophils(
            current_pos,
            config.MACROPHAGE_SENSE_RADIUS,
        )
        local_peak = env.env.get_local_chemokine_peak(
            current_pos,
            config.MACROPHAGE_SIGNAL_TRIGGER_RADIUS,
        )
        nearest_before = None
        nearest_after = None
        if visible_bacteria and action[0] == "move":
            nearest_before = min(
                abs(b.position[0] - current_pos[0]) + abs(b.position[1] - current_pos[1])
                for b in visible_bacteria
            )
            dx, dy = action[1]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            nearest_after = min(
                abs(b.position[0] - next_pos[0]) + abs(b.position[1] - next_pos[1])
                for b in visible_bacteria
            )

        observations.append({key: np.array(value, copy=True) for key, value in obs.items()})
        actions.append(MACROPHAGE_ACTIONS.index(action))
        metadata.append(
            {
                "visible_bacteria": len(visible_bacteria),
                "visible_neutrophils": len(visible_neutrophils),
                "adjacent_bacteria": int(bool(env.env._adjacent_bacteria(current_pos))),
                "local_peak": float(local_peak),
                "queued_reinforcements": int(len(env.env.recruitment_queue)),
                "local_pressure": float(
                    len(visible_bacteria) - 0.75 * len(visible_neutrophils)
                ),
                "approaches_visible_bacteria": int(
                    nearest_before is not None and nearest_after is not None and nearest_after < nearest_before
                ),
            }
        )
        obs, _, terminated, truncated, _ = env.step(MACROPHAGE_ACTIONS.index(action))
        if terminated or truncated:
            obs, _ = env.reset()
            expert.reset(env.env)

    return observations, np.asarray(actions, dtype=np.int64), metadata


def _behavior_clone_pretrain(model, observations, action_indices, metadata, bc_epochs, bc_batch_size):
    bc_epochs = max(0, int(bc_epochs))
    if bc_epochs <= 0 or len(action_indices) == 0:
        return

    torch = importlib.import_module("torch")
    functional = torch.nn.functional
    device = model.device

    stacked_obs = {
        key: np.stack([obs[key] for obs in observations], axis=0)
        for key in observations[0]
    }
    actions_tensor = torch.as_tensor(action_indices, dtype=torch.long, device=device)
    sample_weights = []
    for action_idx, meta in zip(action_indices, metadata):
        action = MACROPHAGE_ACTIONS[int(action_idx)]
        visible_bacteria = int(meta["visible_bacteria"])
        visible_neutrophils = int(meta["visible_neutrophils"])
        adjacent_bacteria = int(meta["adjacent_bacteria"])
        local_peak = float(meta["local_peak"])
        queued_reinforcements = int(meta["queued_reinforcements"])
        local_pressure = float(meta["local_pressure"])
        approaches_visible = bool(meta["approaches_visible_bacteria"])

        if action == ("attack", None):
            sample_weights.append(10.0 if adjacent_bacteria else 2.5)
        elif action[0] in {"signal", "signal_low", "signal_medium", "signal_high"}:
            if adjacent_bacteria:
                sample_weights.append(0.15)
            elif visible_bacteria > 0:
                if queued_reinforcements > 0 or visible_neutrophils > 0:
                    sample_weights.append(0.6)
                elif local_pressure >= config.HEURISTIC_SIGNAL_PRESSURE_MEDIUM:
                    sample_weights.append(5.5)
                elif (
                    local_pressure >= config.HEURISTIC_SIGNAL_PRESSURE_LOW
                    and local_peak >= config.MACROPHAGE_SIGNAL_LOCAL_CHEMOKINE_THRESHOLD * 0.75
                ):
                    sample_weights.append(3.5)
                else:
                    sample_weights.append(1.0)
            elif (
                queued_reinforcements == 0
                and local_peak >= config.MACROPHAGE_SIGNAL_LOCAL_CHEMOKINE_THRESHOLD
            ):
                sample_weights.append(1.8)
            else:
                sample_weights.append(0.3)
        elif action == ("move", (0, 0)):
            sample_weights.append(0.05 if visible_bacteria > 0 else 0.15)
        else:
            if visible_bacteria > 0:
                sample_weights.append(4.8 if approaches_visible else 0.35)
            elif local_peak > 0.12:
                sample_weights.append(1.6)
            else:
                sample_weights.append(0.55)
    weight_tensor = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    batch_size = max(32, int(bc_batch_size))
    n_samples = int(actions_tensor.shape[0])
    is_recurrent_model = bool(getattr(model.policy, "lstm_hidden_state_shape", None))

    model.policy.set_training_mode(True)
    optimizer = model.policy.optimizer

    for epoch in range(bc_epochs):
        permutation = np.random.permutation(n_samples)
        epoch_loss = 0.0
        epoch_correct = 0
        seen = 0
        for start in range(0, n_samples, batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_obs = {
                key: values[batch_idx]
                for key, values in stacked_obs.items()
            }
            obs_tensor, _ = model.policy.obs_to_tensor(batch_obs)
            if is_recurrent_model:
                lstm_shape = tuple(int(v) for v in model.policy.lstm_hidden_state_shape)
                recurrent_state = (
                    torch.zeros((lstm_shape[0], len(batch_idx), lstm_shape[2]), dtype=torch.float32, device=device),
                    torch.zeros((lstm_shape[0], len(batch_idx), lstm_shape[2]), dtype=torch.float32, device=device),
                )
                episode_starts = torch.ones((len(batch_idx),), dtype=torch.float32, device=device)
                distribution, _ = model.policy.get_distribution(
                    obs_tensor,
                    recurrent_state,
                    episode_starts,
                )
            else:
                distribution = model.policy.get_distribution(obs_tensor)
            logits = distribution.distribution.logits
            batch_actions = actions_tensor[batch_idx]
            batch_weights = weight_tensor[batch_idx]
            per_item_loss = functional.cross_entropy(logits, batch_actions, reduction="none")
            loss = (per_item_loss * batch_weights).sum() / torch.clamp(batch_weights.sum(), min=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item()) * len(batch_idx)
            epoch_correct += int((logits.argmax(dim=1) == batch_actions).sum().detach().cpu().item())
            seen += len(batch_idx)

        mean_loss = epoch_loss / max(1, seen)
        accuracy = epoch_correct / max(1, seen)
        print(f"[BC] epoch={epoch + 1}/{bc_epochs} loss={mean_loss:.4f} acc={accuracy:.3f}")

    model.policy.set_training_mode(False)


def train(
    obs_mode,
    timesteps,
    save_dir,
    seed,
    n_envs,
    eval_freq,
    include_belief_features,
    partial_radius,
    use_action_mask,
    guided_start_prob,
    bc_pretrain_steps,
    bc_epochs,
    n_steps,
    batch_size,
    recurrent,
    recurrent_hidden_size,
    recurrent_n_lstm_layers,
):
    try:
        sb3 = importlib.import_module("stable_baselines3")
        callbacks_mod = importlib.import_module("stable_baselines3.common.callbacks")
        vec_env_mod = importlib.import_module("stable_baselines3.common.vec_env")
        recurrent_mod = None
        recurrent_policies_mod = None
        if recurrent:
            recurrent_mod = importlib.import_module("sb3_contrib")
            recurrent_policies_mod = importlib.import_module("sb3_contrib.ppo_recurrent.policies")
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for PPO training. Install with: "
            "pip install stable-baselines3 gymnasium"
        ) from exc

    PPO = sb3.PPO
    RecurrentPPO = recurrent_mod.RecurrentPPO if recurrent_mod is not None else None
    CheckpointCallback = callbacks_mod.CheckpointCallback
    EvalCallback = callbacks_mod.EvalCallback
    DummyVecEnv = vec_env_mod.DummyVecEnv
    VecMonitor = vec_env_mod.VecMonitor

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_stem = f"ppo_macrophage_{obs_mode}_recurrent" if recurrent else f"ppo_macrophage_{obs_mode}"

    env_fns = [
        _make_env(
            obs_mode=obs_mode,
            seed=seed,
            env_id=i,
            include_belief_features=include_belief_features,
            partial_radius=partial_radius,
            include_action_mask=use_action_mask,
            guided_start_prob=guided_start_prob,
            deterministic_reset_stream=False,
        )
        for i in range(n_envs)
    ]
    train_env = VecMonitor(DummyVecEnv(env_fns))

    eval_seed = (seed + 10_000) if seed is not None else 10_000
    eval_env = VecMonitor(
        DummyVecEnv(
            [
                _make_env(
                    obs_mode=obs_mode,
                    seed=eval_seed,
                    env_id=0,
                    include_belief_features=include_belief_features,
                    partial_radius=partial_radius,
                    include_action_mask=use_action_mask,
                    guided_start_prob=0.0,
                    deterministic_reset_stream=True,
                )
            ]
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, eval_freq),
        save_path=str(save_dir / "checkpoints"),
        name_prefix=model_stem,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=eval_freq,
        deterministic=True,
        n_eval_episodes=10,
    )

    policy_kwargs = dict(
        features_extractor_class=MacrophageCombinedExtractor,
        features_extractor_kwargs=dict(grid_feature_dim=192, scalar_hidden_dim=64),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )
    if recurrent:
        policy_kwargs.update(
            dict(
                lstm_hidden_size=max(32, int(recurrent_hidden_size)),
                n_lstm_layers=max(1, int(recurrent_n_lstm_layers)),
                shared_lstm=False,
                enable_critic_lstm=True,
            )
        )
        algorithm_cls = RecurrentPPO
        policy_cls = (
            MaskedRecurrentMacrophagePolicy
            if use_action_mask
            else recurrent_policies_mod.MultiInputLstmPolicy
        )
    else:
        algorithm_cls = PPO
        policy_cls = MaskedMacrophagePolicy if use_action_mask else "MultiInputPolicy"

    model = algorithm_cls(
        policy_cls,
        train_env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
    )

    expert_observations, expert_actions, expert_metadata = _collect_expert_dataset(
        obs_mode=obs_mode,
        seed=(seed + 50_000) if seed is not None else 50_000,
        include_belief_features=include_belief_features,
        partial_radius=partial_radius,
        use_action_mask=use_action_mask,
        guided_start_prob=guided_start_prob,
        expert_steps=bc_pretrain_steps,
    )
    _behavior_clone_pretrain(
        model,
        expert_observations,
        expert_actions,
        expert_metadata,
        bc_epochs=bc_epochs,
        bc_batch_size=batch_size,
    )

    if timesteps > 0:
        model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback])
    else:
        print("Skipping PPO fine-tuning because timesteps <= 0")

    final_path = save_dir / f"{model_stem}_final"
    model.save(str(final_path))
    print(f"Saved final model to: {final_path}")

    best_model_path = save_dir / "best" / "best_model.zip"
    if best_model_path.exists():
        best_alias_path = save_dir / f"{model_stem}_best.zip"
        shutil.copy2(best_model_path, best_alias_path)
        print(f"Saved best-model alias to: {best_alias_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO macrophage controller")
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="partial_state",
        choices=["full_state", "partial_state"],
        help="Observation mode for PPO training",
    )
    parser.add_argument("--timesteps", type=int, default=150000, help="Total PPO training steps")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of vectorized environments")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout length per update")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size")
    parser.add_argument(
        "--disable-belief-features",
        action="store_true",
        help="Disable belief-style feature vector in observations",
    )
    parser.add_argument(
        "--partial-radius",
        type=int,
        default=None,
        help="Override local observation radius for partial_state mode",
    )
    parser.add_argument(
        "--disable-action-mask",
        action="store_true",
        help="Train with legacy unmasked discrete actions",
    )
    parser.add_argument(
        "--guided-start-prob",
        type=float,
        default=0.65,
        help="Training-only probability of starting the macrophage within sensing range of the infection hotspot",
    )
    parser.add_argument(
        "--bc-pretrain-steps",
        type=int,
        default=5000,
        help="Number of heuristic expert steps to use for behavior-cloning warm start before PPO",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=6,
        help="Number of behavior-cloning epochs to run before PPO",
    )
    parser.add_argument(
        "--recurrent",
        action="store_true",
        help="Train a recurrent PPO policy (LSTM) for the partial-observation task",
    )
    parser.add_argument(
        "--recurrent-hidden-size",
        type=int,
        default=128,
        help="Hidden size of the recurrent PPO LSTM when --recurrent is enabled",
    )
    parser.add_argument(
        "--recurrent-layers",
        type=int,
        default=1,
        help="Number of LSTM layers for recurrent PPO when --recurrent is enabled",
    )

    args = parser.parse_args()
    train(
        obs_mode=args.obs_mode,
        timesteps=args.timesteps,
        save_dir=args.save_dir,
        seed=args.seed,
        n_envs=max(1, args.n_envs),
        eval_freq=max(1000, args.eval_freq),
        include_belief_features=not args.disable_belief_features,
        partial_radius=args.partial_radius,
        use_action_mask=not args.disable_action_mask,
        guided_start_prob=max(0.0, min(1.0, args.guided_start_prob)),
        bc_pretrain_steps=max(0, args.bc_pretrain_steps),
        bc_epochs=max(0, args.bc_epochs),
        n_steps=max(32, args.n_steps),
        batch_size=max(32, args.batch_size),
        recurrent=bool(args.recurrent),
        recurrent_hidden_size=max(32, args.recurrent_hidden_size),
        recurrent_n_lstm_layers=max(1, args.recurrent_layers),
    )
