"""Run paper-track robustness evaluations for the frozen learned controller."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import config
from experiments.evaluate_policies import evaluate_all


ROBUSTNESS_SUITES = {
    "baseline_reference": {
        "description": "Frozen benchmark with no config overrides.",
        "overrides": {},
    },
    "high_bacteria_stochasticity": {
        "description": "Increased bacteria movement stochasticity under the same host policy budget.",
        "overrides": {
            "BACTERIA_STOCHASTICITY": 0.24,
        },
    },
    "hotspot_overload": {
        "description": "Higher initial bacterial burden in the primary hotspot compartment.",
        "overrides": {
            "INITIAL_BACTERIA_COUNT": 5,
        },
    },
    "recruitment_delay_stress": {
        "description": "Slower and noisier neutrophil arrival after signaling.",
        "overrides": {
            "NEUTROPHIL_ARRIVAL_DELAY": 11,
            "NEUTROPHIL_ARRIVAL_DELAY_JITTER": 3,
        },
    },
    "fragile_tissue": {
        "description": "Host becomes more damage-sensitive under inflammation and immune usage.",
        "overrides": {
            "TISSUE_DAMAGE_FROM_NEUTROPHILS": 0.60,
            "TISSUE_DAMAGE_FROM_BACTERIA": 0.14,
            "DAMAGE_FROM_PROLONGED_INFLAMMATION": 0.08,
        },
    },
    "narrow_bottleneck": {
        "description": "Doorway bottlenecks are tightened to stress movement and coordination.",
        "overrides": {
            "PASSAGE_BOTTLENECK_WIDTH": 1,
        },
    },
}


@contextmanager
def _temporary_config(overrides):
    originals = {name: getattr(config, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(config, name, value)


def _run_single_suite(
    suite_name,
    suite_spec,
    rl_model_path,
    episodes,
    base_seed,
    obs_mode,
    output_dir,
    max_steps,
):
    suite_dir = output_dir / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    output_csv = suite_dir / f"{suite_name}.csv"
    summary_csv = suite_dir / f"{suite_name}_summary.csv"

    print(f"\n=== Robustness Suite: {suite_name} ===")
    print(suite_spec["description"])
    print(f"Overrides: {suite_spec['overrides']}")

    with _temporary_config(suite_spec["overrides"]):
        evaluate_all(
            episodes=episodes,
            base_seed=base_seed,
            rl_model_path=rl_model_path,
            obs_mode=obs_mode,
            output_csv=str(output_csv),
            max_steps=max_steps,
        )

    summary = pd.read_csv(summary_csv)
    summary.insert(0, "suite", suite_name)
    summary.insert(1, "description", suite_spec["description"])
    summary["overrides"] = json.dumps(suite_spec["overrides"], sort_keys=True)
    return summary


def run_robustness_suite(
    rl_model_path,
    output_dir,
    episodes=100,
    base_seed=123,
    obs_mode="partial_state",
    suites=None,
    max_steps=None,
    include_baseline=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_names = list(suites) if suites else list(ROBUSTNESS_SUITES.keys())
    if not include_baseline and "baseline_reference" in selected_names:
        selected_names = [name for name in selected_names if name != "baseline_reference"]

    unknown = [name for name in selected_names if name not in ROBUSTNESS_SUITES]
    if unknown:
        raise ValueError(f"Unknown robustness suite(s): {unknown}")

    suite_summaries = []
    for suite_name in selected_names:
        suite_spec = ROBUSTNESS_SUITES[suite_name]
        suite_summary = _run_single_suite(
            suite_name=suite_name,
            suite_spec=suite_spec,
            rl_model_path=rl_model_path,
            episodes=episodes,
            base_seed=base_seed,
            obs_mode=obs_mode,
            output_dir=output_dir,
            max_steps=max_steps,
        )
        suite_summaries.append(suite_summary)

    aggregate_summary = pd.concat(suite_summaries, ignore_index=True)
    aggregate_path = output_dir / "robustness_suite_summary.csv"
    aggregate_summary.to_csv(aggregate_path, index=False)

    manifest = {
        "episodes": int(episodes),
        "base_seed": None if base_seed is None else int(base_seed),
        "obs_mode": obs_mode,
        "rl_model_path": str(rl_model_path),
        "suites": {name: ROBUSTNESS_SUITES[name] for name in selected_names},
    }
    manifest_path = output_dir / "robustness_suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n=== Aggregate Robustness Summary ===")
    print(aggregate_summary.to_string(index=False))
    print(f"\nSaved aggregate summary to: {aggregate_path}")
    print(f"Saved suite manifest to: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run robustness suites for heuristic vs frozen learned-controller policies."
    )
    parser.add_argument(
        "--rl-model",
        type=str,
        default=None,
        help="Path to the learned-controller checkpoint (.pt or .zip).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/robustness",
        help="Directory where suite CSVs and aggregate summaries will be written.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of held-out episodes per suite.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base seed for held-out evaluation.",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="partial_state",
        choices=["partial_state"],
        help="Observation mode expected by the learned controller.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        dest="suites",
        help="Optional suite name to run. Repeat to run multiple named suites.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional evaluation truncation horizon.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not run the baseline_reference suite.",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="Print the available robustness suites and exit.",
    )
    args = parser.parse_args()

    if args.list_suites:
        for suite_name, suite_spec in ROBUSTNESS_SUITES.items():
            print(f"{suite_name}: {suite_spec['description']}")
        raise SystemExit(0)

    if not args.rl_model:
        parser.error("--rl-model is required unless --list-suites is used.")

    run_robustness_suite(
        rl_model_path=args.rl_model,
        output_dir=args.output_dir,
        episodes=max(1, args.episodes),
        base_seed=args.seed,
        obs_mode=args.obs_mode,
        suites=args.suites,
        max_steps=args.max_steps,
        include_baseline=not args.skip_baseline,
    )
