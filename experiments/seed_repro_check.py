"""Small reproducibility checks for environment and RL wrapper seeding behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

from simulator.environment import Environment
from simulator.rl_interface import MacrophageGymEnv


def _snapshot(env):
    bacteria = sorted((b.position[0], b.position[1], b.health, b.age, b.state) for b in env.bacteria)
    neutrophils = sorted((n.position[0], n.position[1], n.health, n.age) for n in env.neutrophils)
    return {
        "step": env.step_count,
        "done": env.done,
        "winner": env.winner,
        "win_reason": env.win_reason,
        "macrophage": (env.macrophage.position, env.macrophage.health, env.macrophage.signal_cooldown),
        "bacteria": bacteria,
        "neutrophils": neutrophils,
        "tissue_damage": float(env.tissue_damage),
        "immune_usage": float(env.immune_usage),
        "chemokine": np.array(env.chemokine, copy=True),
        "nutrients": np.array(env.nutrients, copy=True),
        "queue": list(env.recruitment_queue),
        "recent_burden": float(env.recent_chemokine_burden),
    }


def _assert_snapshots_equal(a, b):
    assert a["step"] == b["step"]
    assert a["done"] == b["done"]
    assert a["winner"] == b["winner"]
    assert a["win_reason"] == b["win_reason"]
    assert a["macrophage"] == b["macrophage"]
    assert a["bacteria"] == b["bacteria"]
    assert a["neutrophils"] == b["neutrophils"]
    assert abs(a["tissue_damage"] - b["tissue_damage"]) < 1e-12
    assert abs(a["immune_usage"] - b["immune_usage"]) < 1e-12
    assert a["queue"] == b["queue"]
    assert abs(a["recent_burden"] - b["recent_burden"]) < 1e-12
    assert np.array_equal(a["chemokine"], b["chemokine"])
    assert np.array_equal(a["nutrients"], b["nutrients"])


def check_env_reproducibility():
    actions = [
        ("move", (0, 0)),
        ("move", (1, 0)),
        ("move", (0, 1)),
        ("attack", None),
        ("signal_low", None),
        ("move", (-1, 0)),
        ("move", (0, -1)),
        ("attack", None),
    ]

    env_a = Environment(seed=1234)
    env_b = Environment(seed=1234)

    for action in actions:
        env_a.step(macrophage_action=action)
        env_b.step(macrophage_action=action)

    snap_a = _snapshot(env_a)
    snap_b = _snapshot(env_b)
    _assert_snapshots_equal(snap_a, snap_b)


def check_clone_equivalence():
    warmup_actions = [
        ("move", (1, 0)),
        ("move", (0, 1)),
        ("attack", None),
        ("signal_medium", None),
        ("move", (-1, 0)),
    ]
    future_actions = [
        ("move", (0, -1)),
        ("move", (1, 0)),
        ("attack", None),
        ("move", (0, 1)),
        ("signal_low", None),
        ("move", (-1, 0)),
        ("attack", None),
    ]

    env = Environment(seed=2468)
    for action in warmup_actions:
        env.step(macrophage_action=action)

    env_clone = env.clone()

    for action in future_actions:
        env.step(macrophage_action=action)
        env_clone.step(macrophage_action=action)

    _assert_snapshots_equal(_snapshot(env), _snapshot(env_clone))


def check_training_reset_diversity():
    gym_env = MacrophageGymEnv(observation_mode="partial_state", seed=77, env_id=2)

    seeds = []
    for _ in range(5):
        _, info = gym_env.reset()
        seeds.append(int(info["episode_seed"]))

    unique_count = len(set(seeds))
    assert unique_count > 1, f"Expected diverse reset seeds, got {seeds}"


def check_eval_reset_progression_reproducibility():
    # Eval mode should produce deterministic, reproducible per-reset seed progression.
    env_a = MacrophageGymEnv(
        observation_mode="partial_state",
        seed=77,
        env_id=2,
        deterministic_reset_stream=True,
    )
    env_b = MacrophageGymEnv(
        observation_mode="partial_state",
        seed=77,
        env_id=2,
        deterministic_reset_stream=True,
    )

    seq_a = []
    seq_b = []
    for _ in range(6):
        _, info_a = env_a.reset()
        _, info_b = env_b.reset()
        seq_a.append(int(info_a["episode_seed"]))
        seq_b.append(int(info_b["episode_seed"]))

    assert seq_a == seq_b, f"Expected reproducible eval seed progression, got {seq_a} vs {seq_b}"
    assert len(set(seq_a)) > 1, f"Expected progressing eval seeds, got constant sequence: {seq_a}"


def _run_check(name, fn):
    try:
        fn()
        return True, name, "PASS"
    except Exception as exc:  # pragma: no cover - intentional for CLI diagnostics
        return False, name, f"FAIL: {type(exc).__name__}: {exc}"


def main():
    checks = [
        ("same_seed_same_actions_full_stochasticity", check_env_reproducibility),
        ("clone_equivalence_under_future_actions", check_clone_equivalence),
        ("rl_training_reset_diversity", check_training_reset_diversity),
        ("rl_eval_reset_progression_reproducibility", check_eval_reset_progression_reproducibility),
    ]

    results = [_run_check(name, fn) for name, fn in checks]
    all_passed = all(ok for ok, _, _ in results)

    print("=== Seed/RNG Reproducibility Summary ===")
    for _, name, status in results:
        print(f"{name}: {status}")

    if all_passed:
        print("OVERALL: PASS")
        return

    print("OVERALL: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
