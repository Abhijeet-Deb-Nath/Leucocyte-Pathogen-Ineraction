"""Train the hierarchical recurrent learned controller via BC, DAgger, and RL fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from agents.heuristic import HeuristicAgent
from agents.hierarchical_policy import (
    ENGAGE_MODE_INDEX,
    FOLLOW_CHEM_MODE_INDEX,
    MICRO_ATTACK_INDEX,
    REQUEST_HELP_MODE_INDEX,
    HierarchicalRecurrentPolicy,
    build_macro_mask,
    build_micro_action_masks,
    compute_privileged_signals,
    macro_mode_to_index,
    micro_action_to_index,
    resolve_hierarchical_action,
    save_hierarchical_checkpoint,
)
from experiments.evaluate_policies import evaluate_all
from simulator.environment import Environment
from simulator.rl_interface import MacrophageObservationBuilder


def _build_env_for_seed(seed: int, guided_start_prob: float) -> Environment:
    start_mode = "surface_random"
    if guided_start_prob > 0.0:
        curriculum_rng = np.random.default_rng(int(seed) + 97_531)
        if float(curriculum_rng.random()) < guided_start_prob:
            start_mode = "near_bacteria"
    return Environment(seed=seed, macrophage_start_mode=start_mode)


def _copy_observation(observation):
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def _step_metadata(env, labeled_action):
    mx, my = env.macrophage.position
    visible_bacteria = env.get_local_bacteria((mx, my), config.MACROPHAGE_SENSE_RADIUS)
    visible_neutrophils = env.get_local_neutrophils((mx, my), config.MACROPHAGE_SENSE_RADIUS)
    local_peak = env.get_local_chemokine_peak((mx, my), config.MACROPHAGE_SIGNAL_TRIGGER_RADIUS)
    nearest_before = None
    nearest_after = None
    if visible_bacteria and labeled_action[0] == "move":
        nearest_before = min(
            abs(b.position[0] - mx) + abs(b.position[1] - my)
            for b in visible_bacteria
        )
        dx, dy = labeled_action[1]
        next_pos = (mx + dx, my + dy)
        nearest_after = min(
            abs(b.position[0] - next_pos[0]) + abs(b.position[1] - next_pos[1])
            for b in visible_bacteria
        )
    return {
        "visible_bacteria": int(len(visible_bacteria)),
        "visible_neutrophils": int(len(visible_neutrophils)),
        "adjacent_bacteria": int(bool(env._adjacent_bacteria((mx, my)))),
        "local_peak": float(local_peak),
        "local_pressure": float(len(visible_bacteria) - 0.75 * len(visible_neutrophils)),
        "approaches_visible_bacteria": int(
            nearest_before is not None and nearest_after is not None and nearest_after < nearest_before
        ),
    }


def _sample_weight(macro_label, micro_label, meta):
    visible_bacteria = int(meta["visible_bacteria"])
    adjacent_bacteria = int(meta["adjacent_bacteria"])
    local_peak = float(meta["local_peak"])
    local_pressure = float(meta["local_pressure"])
    approaches_visible = bool(meta["approaches_visible_bacteria"])

    if macro_label == REQUEST_HELP_MODE_INDEX:
        if adjacent_bacteria:
            return 0.4
        if local_pressure >= config.HEURISTIC_SIGNAL_PRESSURE_MEDIUM:
            return 4.5
        if local_pressure >= config.HEURISTIC_SIGNAL_PRESSURE_LOW:
            return 2.4
        return 1.0
    if micro_label == MICRO_ATTACK_INDEX:
        return 10.0 if adjacent_bacteria else 2.0
    if macro_label == ENGAGE_MODE_INDEX:
        if visible_bacteria > 0:
            return 5.5 if approaches_visible else 1.8
        return 0.8
    if macro_label == FOLLOW_CHEM_MODE_INDEX:
        return 2.0 if local_peak >= config.MACROPHAGE_SIGNAL_LOCAL_CHEMOKINE_THRESHOLD * 0.5 else 0.8
    return 0.75


def _new_episode(seed):
    return {
        "episode_seed": int(seed),
        "observations": [],
        "macro_labels": [],
        "micro_labels": [],
        "macro_masks": [],
        "micro_masks": [],
        "privileged_scalars": [],
        "aux_local_hidden_burden": [],
        "aux_escalation_needed": [],
        "aux_target_compartment": [],
        "metadata": [],
        "rewards": [],
        "values": [],
        "old_log_probs": [],
        "executed_macro": [],
        "executed_micro": [],
        "executed_actions": [],
    }


def _observation_config(include_belief_features, partial_radius):
    return {
        "include_belief_features": bool(include_belief_features),
        "partial_radius": partial_radius,
    }


def _collect_labeled_episodes(
    total_steps,
    base_seed,
    include_belief_features,
    partial_radius,
    guided_start_prob,
    policy=None,
    deterministic=False,
):
    observer = MacrophageObservationBuilder(
        mode="partial_state",
        partial_radius=partial_radius,
        include_belief_features=include_belief_features,
        include_action_mask=False,
    )
    teacher = HeuristicAgent(use_memory=False, stochastic_patrol=False)
    episodes = []
    steps_collected = 0
    episode_index = 0
    device = None if policy is None else next(policy.parameters()).device

    while steps_collected < total_steps:
        episode_seed = int(base_seed + episode_index)
        env = _build_env_for_seed(episode_seed, guided_start_prob)
        observer.reset(env)
        teacher.reset(env)
        hidden_state = None

        episode = _new_episode(episode_seed)
        while not env.done and steps_collected < total_steps:
            observation = _copy_observation(observer.observe(env))
            teacher_mode, teacher_action = teacher.choose_labeled_action(env)
            teacher_macro = macro_mode_to_index(teacher_mode)
            teacher_micro = micro_action_to_index(teacher_action)
            macro_mask = build_macro_mask(env)
            micro_masks = build_micro_action_masks(env)
            privileged = compute_privileged_signals(env)
            metadata = _step_metadata(env, teacher_action)

            if policy is None:
                executed_action = teacher_action
                executed_macro = teacher_macro
                executed_micro = teacher_micro
                step_value = 0.0
                step_log_prob = 0.0
            else:
                policy_output = policy.act(
                    observation=observation,
                    macro_mask=macro_mask,
                    micro_masks=micro_masks,
                    hidden_state=hidden_state,
                    privileged_scalars=privileged["privileged_scalars"],
                    deterministic=deterministic,
                    device=device,
                )
                hidden_state = policy_output["hidden_state"]
                executed_macro = int(policy_output["macro_index"])
                executed_micro = int(policy_output["micro_index"])
                executed_action = resolve_hierarchical_action(
                    env,
                    macro_index=executed_macro,
                    micro_index=executed_micro,
                )
                step_value = float(policy_output["value"])
                step_log_prob = float(policy_output["log_prob"])

            utility_before = env.compute_host_utility()
            env.step(macrophage_action=executed_action)
            utility_after = env.compute_host_utility()
            reward = float(utility_after - utility_before)

            episode["observations"].append(observation)
            episode["macro_labels"].append(int(teacher_macro))
            episode["micro_labels"].append(int(teacher_micro))
            episode["macro_masks"].append(np.array(macro_mask, copy=True))
            episode["micro_masks"].append(np.array(micro_masks, copy=True))
            episode["privileged_scalars"].append(np.array(privileged["privileged_scalars"], copy=True))
            episode["aux_local_hidden_burden"].append(float(privileged["local_hidden_burden"]))
            episode["aux_escalation_needed"].append(float(privileged["escalation_needed"]))
            episode["aux_target_compartment"].append(int(privileged["target_compartment"]))
            episode["metadata"].append(metadata)
            episode["rewards"].append(reward)
            episode["values"].append(step_value)
            episode["old_log_probs"].append(step_log_prob)
            episode["executed_macro"].append(int(executed_macro))
            episode["executed_micro"].append(int(executed_micro))
            episode["executed_actions"].append(executed_action)

            steps_collected += 1

        episodes.append(episode)
        episode_index += 1

    return episodes


def _episode_chunks(episodes, sequence_length):
    sequence_length = max(4, int(sequence_length))
    for episode_idx, episode in enumerate(episodes):
        episode_length = len(episode["macro_labels"])
        for start in range(0, episode_length, sequence_length):
            end = min(episode_length, start + sequence_length)
            yield episode_idx, start, end


def _stack_chunk(episode, start, end, device):
    observations = episode["observations"][start:end]
    grid = np.stack([step["grid"] for step in observations], axis=0)
    scalars = np.stack([step["scalars"] for step in observations], axis=0)
    privileged = np.stack(episode["privileged_scalars"][start:end], axis=0)
    macro_masks = np.stack(episode["macro_masks"][start:end], axis=0)
    micro_masks = np.stack(episode["micro_masks"][start:end], axis=0)
    macro_labels = np.asarray(episode["macro_labels"][start:end], dtype=np.int64)
    micro_labels = np.asarray(episode["micro_labels"][start:end], dtype=np.int64)
    aux_burden = np.asarray(episode["aux_local_hidden_burden"][start:end], dtype=np.float32)
    aux_escalation = np.asarray(episode["aux_escalation_needed"][start:end], dtype=np.float32)
    aux_target = np.asarray(episode["aux_target_compartment"][start:end], dtype=np.int64)
    weights = np.asarray(
        [
            _sample_weight(macro_label, micro_label, meta)
            for macro_label, micro_label, meta in zip(
                macro_labels,
                micro_labels,
                episode["metadata"][start:end],
            )
        ],
        dtype=np.float32,
    )

    return {
        "grid": torch.as_tensor(grid, dtype=torch.float32, device=device).unsqueeze(1),
        "scalars": torch.as_tensor(scalars, dtype=torch.float32, device=device).unsqueeze(1),
        "privileged_scalars": torch.as_tensor(privileged, dtype=torch.float32, device=device).unsqueeze(1),
        "macro_masks": torch.as_tensor(macro_masks, dtype=torch.float32, device=device),
        "micro_masks": torch.as_tensor(micro_masks, dtype=torch.float32, device=device),
        "macro_labels": torch.as_tensor(macro_labels, dtype=torch.long, device=device),
        "micro_labels": torch.as_tensor(micro_labels, dtype=torch.long, device=device),
        "aux_burden": torch.as_tensor(aux_burden, dtype=torch.float32, device=device),
        "aux_escalation": torch.as_tensor(aux_escalation, dtype=torch.float32, device=device),
        "aux_target": torch.as_tensor(aux_target, dtype=torch.long, device=device),
        "weights": torch.as_tensor(weights, dtype=torch.float32, device=device),
    }


def _supervised_epoch(model, episodes, sequence_length, optimizer, device):
    model.train()
    chunks = list(_episode_chunks(episodes, sequence_length))
    np.random.shuffle(chunks)

    totals = {
        "macro_correct": 0,
        "micro_correct": 0,
        "samples": 0,
        "loss": 0.0,
    }

    for episode_idx, start, end in chunks:
        batch = _stack_chunk(episodes[episode_idx], start, end, device)
        outputs = model.forward_sequence(
            batch["grid"],
            batch["scalars"],
            privileged_scalars=batch["privileged_scalars"],
        )

        macro_logits = outputs["macro_logits"].squeeze(1)
        macro_logits = torch.where(
            batch["macro_masks"] > 0.0,
            macro_logits,
            torch.full_like(macro_logits, torch.finfo(macro_logits.dtype).min),
        )
        macro_loss_all = F.cross_entropy(macro_logits, batch["macro_labels"], reduction="none")
        macro_predictions = macro_logits.argmax(dim=-1)

        selected_micro_logits = outputs["micro_logits"].squeeze(1)[
            torch.arange(batch["macro_labels"].shape[0], device=device),
            batch["macro_labels"],
        ]
        selected_micro_masks = batch["micro_masks"][
            torch.arange(batch["macro_labels"].shape[0], device=device),
            batch["macro_labels"],
        ]
        selected_micro_logits = torch.where(
            selected_micro_masks > 0.0,
            selected_micro_logits,
            torch.full_like(selected_micro_logits, torch.finfo(selected_micro_logits.dtype).min),
        )
        micro_loss_all = F.cross_entropy(selected_micro_logits, batch["micro_labels"], reduction="none")
        micro_predictions = selected_micro_logits.argmax(dim=-1)

        burden_loss_all = F.mse_loss(outputs["burden"].squeeze(1), batch["aux_burden"], reduction="none")
        escalation_loss_all = F.binary_cross_entropy_with_logits(
            outputs["escalation"].squeeze(1),
            batch["aux_escalation"],
            reduction="none",
        )
        target_loss_all = F.cross_entropy(
            outputs["target_logits"].squeeze(1),
            batch["aux_target"],
            reduction="none",
        )

        weights = batch["weights"]
        denom = torch.clamp(weights.sum(), min=1.0)
        total_loss = (
            ((macro_loss_all + micro_loss_all) * weights).sum() / denom
            + 0.25 * (burden_loss_all.mean() + escalation_loss_all.mean() + target_loss_all.mean())
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        totals["macro_correct"] += int((macro_predictions == batch["macro_labels"]).sum().item())
        totals["micro_correct"] += int((micro_predictions == batch["micro_labels"]).sum().item())
        totals["samples"] += int(batch["macro_labels"].shape[0])
        totals["loss"] += float(total_loss.detach().cpu().item()) * int(batch["macro_labels"].shape[0])

    return {
        "loss": totals["loss"] / max(1, totals["samples"]),
        "macro_accuracy": totals["macro_correct"] / max(1, totals["samples"]),
        "micro_accuracy": totals["micro_correct"] / max(1, totals["samples"]),
    }


def _train_supervised_stage(
    model,
    episodes,
    epochs,
    sequence_length,
    learning_rate,
    device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_metrics = None
    for epoch in range(max(1, int(epochs))):
        final_metrics = _supervised_epoch(
            model=model,
            episodes=episodes,
            sequence_length=sequence_length,
            optimizer=optimizer,
            device=device,
        )
        print(
            "[hier-bc] "
            f"epoch={epoch + 1}/{max(1, int(epochs))} "
            f"loss={final_metrics['loss']:.4f} "
            f"macro_acc={final_metrics['macro_accuracy']:.3f} "
            f"micro_acc={final_metrics['micro_accuracy']:.3f}"
        )
    return final_metrics if final_metrics is not None else {}


def _compute_advantages(episode, gamma, gae_lambda):
    rewards = np.asarray(episode["rewards"], dtype=np.float32)
    values = np.asarray(episode["values"], dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    next_value = 0.0
    for idx in reversed(range(len(rewards))):
        delta = rewards[idx] + gamma * next_value - values[idx]
        last_adv = delta + gamma * gae_lambda * last_adv
        advantages[idx] = last_adv
        next_value = values[idx]
    returns = advantages + values
    episode["advantages"] = advantages.tolist()
    episode["returns"] = returns.tolist()


def _collect_rl_episodes(
    model,
    rollout_episodes,
    base_seed,
    include_belief_features,
    partial_radius,
    guided_start_prob,
    device,
):
    observer = MacrophageObservationBuilder(
        mode="partial_state",
        partial_radius=partial_radius,
        include_belief_features=include_belief_features,
        include_action_mask=False,
    )
    episodes = []
    for episode_index in range(max(1, int(rollout_episodes))):
        episode_seed = int(base_seed + episode_index)
        env = _build_env_for_seed(episode_seed, guided_start_prob)
        observer.reset(env)
        hidden_state = None
        episode = _new_episode(episode_seed)
        while not env.done:
            observation = _copy_observation(observer.observe(env))
            macro_mask = build_macro_mask(env)
            micro_masks = build_micro_action_masks(env)
            privileged = compute_privileged_signals(env)
            output = model.act(
                observation=observation,
                macro_mask=macro_mask,
                micro_masks=micro_masks,
                hidden_state=hidden_state,
                privileged_scalars=privileged["privileged_scalars"],
                deterministic=False,
                device=device,
            )
            hidden_state = output["hidden_state"]
            executed_action = resolve_hierarchical_action(
                env,
                macro_index=output["macro_index"],
                micro_index=output["micro_index"],
            )
            utility_before = env.compute_host_utility()
            env.step(macrophage_action=executed_action)
            utility_after = env.compute_host_utility()
            reward = float(utility_after - utility_before)

            episode["observations"].append(observation)
            episode["macro_labels"].append(int(output["macro_index"]))
            episode["micro_labels"].append(int(output["micro_index"]))
            episode["macro_masks"].append(np.array(macro_mask, copy=True))
            episode["micro_masks"].append(np.array(micro_masks, copy=True))
            episode["privileged_scalars"].append(np.array(privileged["privileged_scalars"], copy=True))
            episode["aux_local_hidden_burden"].append(float(privileged["local_hidden_burden"]))
            episode["aux_escalation_needed"].append(float(privileged["escalation_needed"]))
            episode["aux_target_compartment"].append(int(privileged["target_compartment"]))
            episode["metadata"].append(_step_metadata(env, executed_action))
            episode["rewards"].append(reward)
            episode["values"].append(float(output["value"]))
            episode["old_log_probs"].append(float(output["log_prob"]))
            episode["executed_macro"].append(int(output["macro_index"]))
            episode["executed_micro"].append(int(output["micro_index"]))
            episode["executed_actions"].append(executed_action)

        episodes.append(episode)
    return episodes


def _ppo_update(
    model,
    episodes,
    ppo_epochs,
    learning_rate,
    gamma,
    gae_lambda,
    clip_range,
    value_coef,
    entropy_coef,
    device,
):
    for episode in episodes:
        _compute_advantages(episode, gamma=gamma, gae_lambda=gae_lambda)

    all_advantages = np.concatenate([np.asarray(ep["advantages"], dtype=np.float32) for ep in episodes], axis=0)
    adv_mean = float(all_advantages.mean()) if all_advantages.size else 0.0
    adv_std = max(float(all_advantages.std()), 1e-6) if all_advantages.size else 1.0
    for episode in episodes:
        normalized_adv = (np.asarray(episode["advantages"], dtype=np.float32) - adv_mean) / adv_std
        episode["advantages"] = normalized_adv.tolist()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metrics = None

    for epoch in range(max(1, int(ppo_epochs))):
        np.random.shuffle(episodes)
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_steps = 0

        for episode in episodes:
            batch = _stack_chunk(episode, 0, len(episode["macro_labels"]), device)
            old_log_probs = torch.as_tensor(
                np.asarray(episode["old_log_probs"], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            returns = torch.as_tensor(
                np.asarray(episode["returns"], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            advantages = torch.as_tensor(
                np.asarray(episode["advantages"], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )

            outputs = model.forward_sequence(
                batch["grid"],
                batch["scalars"],
                privileged_scalars=batch["privileged_scalars"],
            )

            macro_logits = outputs["macro_logits"].squeeze(1)
            macro_logits = torch.where(
                batch["macro_masks"] > 0.0,
                macro_logits,
                torch.full_like(macro_logits, torch.finfo(macro_logits.dtype).min),
            )
            macro_dist = torch.distributions.Categorical(logits=macro_logits)
            macro_log_prob = macro_dist.log_prob(batch["macro_labels"])
            macro_entropy = macro_dist.entropy()

            micro_logits = outputs["micro_logits"].squeeze(1)[
                torch.arange(batch["macro_labels"].shape[0], device=device),
                batch["macro_labels"],
            ]
            micro_masks = batch["micro_masks"][
                torch.arange(batch["macro_labels"].shape[0], device=device),
                batch["macro_labels"],
            ]
            micro_logits = torch.where(
                micro_masks > 0.0,
                micro_logits,
                torch.full_like(micro_logits, torch.finfo(micro_logits.dtype).min),
            )
            micro_dist = torch.distributions.Categorical(logits=micro_logits)
            micro_log_prob = micro_dist.log_prob(batch["micro_labels"])
            micro_entropy = micro_dist.entropy()

            joint_log_prob = macro_log_prob + micro_log_prob
            ratio = torch.exp(joint_log_prob - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(outputs["values"].squeeze(1), returns)
            entropy = (macro_entropy + micro_entropy).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_policy += float(policy_loss.detach().cpu().item()) * len(episode["macro_labels"])
            total_value += float(value_loss.detach().cpu().item()) * len(episode["macro_labels"])
            total_entropy += float(entropy.detach().cpu().item()) * len(episode["macro_labels"])
            total_steps += len(episode["macro_labels"])

        metrics = {
            "policy_loss": total_policy / max(1, total_steps),
            "value_loss": total_value / max(1, total_steps),
            "entropy": total_entropy / max(1, total_steps),
        }
        print(
            "[hier-rl] "
            f"epoch={epoch + 1}/{max(1, int(ppo_epochs))} "
            f"policy_loss={metrics['policy_loss']:.4f} "
            f"value_loss={metrics['value_loss']:.4f} "
            f"entropy={metrics['entropy']:.4f}"
        )

    return metrics if metrics is not None else {}


def _evaluate_checkpoint(checkpoint_path, save_dir, stage_name, eval_episodes, eval_seed):
    output_csv = Path(save_dir) / f"{stage_name}_eval.csv"
    evaluate_all(
        episodes=max(1, int(eval_episodes)),
        base_seed=int(eval_seed),
        rl_model_path=str(checkpoint_path),
        obs_mode="partial_state",
        output_csv=str(output_csv),
        max_steps=None,
    )
    return output_csv


def train_hierarchical(
    stages,
    save_dir,
    seed,
    include_belief_features,
    partial_radius,
    guided_start_prob,
    hidden_size,
    recurrent_layers,
    bc_pretrain_steps,
    bc_epochs,
    bc_sequence_length,
    dagger_iterations,
    dagger_steps_per_iter,
    dagger_epochs_per_iter,
    rl_rollout_episodes,
    rl_iterations,
    rl_ppo_epochs,
    bc_learning_rate,
    rl_learning_rate,
    gamma,
    gae_lambda,
    clip_range,
    value_coef,
    entropy_coef,
    eval_episodes,
    eval_seed,
    device_name,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)

    observation_builder = MacrophageObservationBuilder(
        mode="partial_state",
        partial_radius=partial_radius,
        include_belief_features=include_belief_features,
        include_action_mask=False,
    )
    bootstrap_env = _build_env_for_seed(int(seed), guided_start_prob)
    observation_builder.reset(bootstrap_env)
    bootstrap_obs = observation_builder.observe(bootstrap_env)
    privileged_dim = int(len(compute_privileged_signals(bootstrap_env)["privileged_scalars"]))

    model = HierarchicalRecurrentPolicy(
        scalar_dim=int(bootstrap_obs["scalars"].shape[0]),
        privileged_dim=privileged_dim,
        grid_channels=int(bootstrap_obs["grid"].shape[0]),
        hidden_size=hidden_size,
        recurrent_layers=recurrent_layers,
    ).to(device)

    observation_config = _observation_config(
        include_belief_features=include_belief_features,
        partial_radius=partial_radius,
    )
    stage_records = []
    dataset_episodes = []
    stages = [str(stage).strip().lower() for stage in stages]

    if "bc" in stages:
        dataset_episodes = _collect_labeled_episodes(
            total_steps=max(1, int(bc_pretrain_steps)),
            base_seed=int(seed) + 50_000,
            include_belief_features=include_belief_features,
            partial_radius=partial_radius,
            guided_start_prob=guided_start_prob,
            policy=None,
            deterministic=True,
        )
        bc_metrics = _train_supervised_stage(
            model=model,
            episodes=dataset_episodes,
            epochs=bc_epochs,
            sequence_length=bc_sequence_length,
            learning_rate=bc_learning_rate,
            device=device,
        )
        bc_path = save_hierarchical_checkpoint(
            save_dir / "hierarchical_macrophage_bc.pt",
            model=model,
            observation_config=observation_config,
            stage="bc",
            metadata={"metrics": bc_metrics},
        )
        if eval_episodes > 0:
            _evaluate_checkpoint(bc_path, save_dir, "bc", eval_episodes, eval_seed)
        stage_records.append({"stage": "bc", **bc_metrics})

    if "dagger" in stages:
        if not dataset_episodes:
            dataset_episodes = _collect_labeled_episodes(
                total_steps=max(1, int(bc_pretrain_steps)),
                base_seed=int(seed) + 50_000,
                include_belief_features=include_belief_features,
                partial_radius=partial_radius,
                guided_start_prob=guided_start_prob,
                policy=None,
                deterministic=True,
            )
        for iteration in range(max(1, int(dagger_iterations))):
            dagger_episodes = _collect_labeled_episodes(
                total_steps=max(1, int(dagger_steps_per_iter)),
                base_seed=int(seed) + 60_000 + iteration * 10_000,
                include_belief_features=include_belief_features,
                partial_radius=partial_radius,
                guided_start_prob=guided_start_prob,
                policy=model,
                deterministic=False,
            )
            dataset_episodes.extend(dagger_episodes)
            dagger_metrics = _train_supervised_stage(
                model=model,
                episodes=dataset_episodes,
                epochs=dagger_epochs_per_iter,
                sequence_length=bc_sequence_length,
                learning_rate=bc_learning_rate,
                device=device,
            )
            save_hierarchical_checkpoint(
                save_dir / f"hierarchical_macrophage_dagger_iter{iteration + 1}.pt",
                model=model,
                observation_config=observation_config,
                stage="dagger",
                metadata={"iteration": iteration + 1, "metrics": dagger_metrics},
            )
            stage_records.append({"stage": f"dagger_iter_{iteration + 1}", **dagger_metrics})
        dagger_path = save_hierarchical_checkpoint(
            save_dir / "hierarchical_macrophage_dagger.pt",
            model=model,
            observation_config=observation_config,
            stage="dagger",
            metadata={"metrics": stage_records[-1] if stage_records else {}},
        )
        if eval_episodes > 0:
            _evaluate_checkpoint(dagger_path, save_dir, "dagger", eval_episodes, eval_seed)

    if "rl_finetune" in stages:
        for iteration in range(max(1, int(rl_iterations))):
            rollout_episodes = _collect_rl_episodes(
                model=model,
                rollout_episodes=max(1, int(rl_rollout_episodes)),
                base_seed=int(seed) + 80_000 + iteration * 10_000,
                include_belief_features=include_belief_features,
                partial_radius=partial_radius,
                guided_start_prob=guided_start_prob,
                device=device,
            )
            rl_metrics = _ppo_update(
                model=model,
                episodes=rollout_episodes,
                ppo_epochs=rl_ppo_epochs,
                learning_rate=rl_learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                device=device,
            )
            save_hierarchical_checkpoint(
                save_dir / f"hierarchical_macrophage_rl_iter{iteration + 1}.pt",
                model=model,
                observation_config=observation_config,
                stage="rl_finetune",
                metadata={"iteration": iteration + 1, "metrics": rl_metrics},
            )
            stage_records.append({"stage": f"rl_iter_{iteration + 1}", **rl_metrics})

    final_path = save_hierarchical_checkpoint(
        save_dir / "hierarchical_macrophage_final.pt",
        model=model,
        observation_config=observation_config,
        stage="final",
        metadata={"stages": stage_records},
    )
    if eval_episodes > 0:
        _evaluate_checkpoint(final_path, save_dir, "final", eval_episodes, eval_seed)

    summary_path = save_dir / "hierarchical_training_summary.json"
    summary_path.write_text(json.dumps(stage_records, indent=2), encoding="utf-8")
    print(f"Saved training summary to: {summary_path}")
    print(f"Saved final checkpoint to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the hierarchical recurrent macrophage controller"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["bc", "dagger", "rl_finetune"],
        choices=["bc", "dagger", "rl_finetune"],
    )
    parser.add_argument("--save-dir", type=str, default="models/hierarchical_mainline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--recurrent-layers", type=int, default=1)
    parser.add_argument("--bc-pretrain-steps", type=int, default=4096)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-sequence-length", type=int, default=32)
    parser.add_argument("--dagger-iterations", type=int, default=2)
    parser.add_argument("--dagger-steps-per-iter", type=int, default=2048)
    parser.add_argument("--dagger-epochs-per-iter", type=int, default=4)
    parser.add_argument("--rl-rollout-episodes", type=int, default=6)
    parser.add_argument("--rl-iterations", type=int, default=2)
    parser.add_argument("--rl-ppo-epochs", type=int, default=3)
    parser.add_argument("--bc-learning-rate", type=float, default=3e-4)
    parser.add_argument("--rl-learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--guided-start-prob", type=float, default=0.65)
    parser.add_argument("--partial-radius", type=int, default=None)
    parser.add_argument("--disable-belief-features", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    train_hierarchical(
        stages=args.stages,
        save_dir=args.save_dir,
        seed=args.seed,
        include_belief_features=not args.disable_belief_features,
        partial_radius=args.partial_radius,
        guided_start_prob=max(0.0, min(1.0, args.guided_start_prob)),
        hidden_size=max(32, int(args.hidden_size)),
        recurrent_layers=max(1, int(args.recurrent_layers)),
        bc_pretrain_steps=max(1, int(args.bc_pretrain_steps)),
        bc_epochs=max(1, int(args.bc_epochs)),
        bc_sequence_length=max(4, int(args.bc_sequence_length)),
        dagger_iterations=max(1, int(args.dagger_iterations)),
        dagger_steps_per_iter=max(1, int(args.dagger_steps_per_iter)),
        dagger_epochs_per_iter=max(1, int(args.dagger_epochs_per_iter)),
        rl_rollout_episodes=max(1, int(args.rl_rollout_episodes)),
        rl_iterations=max(1, int(args.rl_iterations)),
        rl_ppo_epochs=max(1, int(args.rl_ppo_epochs)),
        bc_learning_rate=float(args.bc_learning_rate),
        rl_learning_rate=float(args.rl_learning_rate),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_range=float(args.clip_range),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        eval_episodes=max(0, int(args.eval_episodes)),
        eval_seed=int(args.eval_seed),
        device_name=args.device,
    )
