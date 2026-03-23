"""Build paper-ready tables and figures from curated result artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_SPECS = [
    {
        "name": "heuristic",
        "label": "Heuristic",
        "source": "hier_dagger_100ep_summary.csv",
        "policy": "heuristic",
    },
    {
        "name": "bc_only",
        "label": "Hierarchical BC-only",
        "source": "hier_bc_100ep_summary.csv",
        "policy": "rl",
    },
    {
        "name": "bc_dagger",
        "label": "Hierarchical BC+DAgger",
        "source": "hier_dagger_100ep_summary.csv",
        "policy": "rl",
    },
    {
        "name": "bc_dagger_rl",
        "label": "Hierarchical BC+DAgger+RL",
        "source": "hier_rl_100ep_summary.csv",
        "policy": "rl",
    },
]

MAIN_METRICS = [
    "win_rate",
    "tissue_damage",
    "host_utility",
    "episode_length",
    "recruited_neutrophils",
]

BEHAVIOR_METRICS = [
    "attack_actions",
    "signal_actions",
    "stay_actions",
    "coverage_ratio",
    "visible_bacteria_ratio",
]


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, separator, *rows, ""])


def _load_method_rows(final_eval_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    missing: list[str] = []

    for spec in METHOD_SPECS:
        path = final_eval_dir / spec["source"]
        if not path.exists():
            missing.append(spec["source"])
            continue

        df = pd.read_csv(path)
        match = df[df["policy"] == spec["policy"]]
        if match.empty:
            missing.append(f"{spec['source']}:{spec['policy']}")
            continue

        row = match.iloc[0].to_dict()
        row["method"] = spec["name"]
        row["method_label"] = spec["label"]
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        ordered_labels = [spec["label"] for spec in METHOD_SPECS if spec["label"] in set(result["method_label"])]
        result["method_label"] = pd.Categorical(result["method_label"], categories=ordered_labels, ordered=True)
        result = result.sort_values("method_label").reset_index(drop=True)
    return result, missing


def _load_reproducibility_table(final_eval_dir: Path) -> pd.DataFrame:
    path = final_eval_dir / "reproducibility_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing reproducibility table: {path}")
    return pd.read_csv(path)


def _load_robustness_table(robustness_dir: Path) -> pd.DataFrame:
    path = robustness_dir / "robustness_suite_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing robustness summary: {path}")
    return pd.read_csv(path)


def _write_missing_report(output_dir: Path, missing: list[str]) -> None:
    report_path = output_dir / "missing_artifacts.txt"
    if missing:
        report_path.write_text(
            "Missing curated artifacts required for a complete main comparison table:\n"
            + "\n".join(f"- {item}" for item in sorted(set(missing)))
            + "\n",
            encoding="utf-8",
        )
    else:
        report_path.write_text("All expected curated artifacts are present.\n", encoding="utf-8")


def _plot_main_comparison(main_df: pd.DataFrame, output_dir: Path) -> None:
    if main_df.empty:
        return

    plot_df = main_df.set_index("method_label")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    plot_df["win_rate"].plot(kind="bar", ax=axes[0], color="#2f6db3")
    axes[0].set_title("Win Rate")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0.0, 1.0)

    plot_df["tissue_damage"].plot(kind="bar", ax=axes[1], color="#c95d3a")
    axes[1].set_title("Tissue Damage")
    axes[1].set_ylabel("Damage")

    plot_df["host_utility"].plot(kind="bar", ax=axes[2], color="#4f8a4b")
    axes[2].set_title("Host Utility")
    axes[2].set_ylabel("Utility")

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_robustness_summary(robustness_df: pd.DataFrame, output_dir: Path) -> None:
    pivot = robustness_df.pivot(index="suite", columns="policy", values="win_rate")
    pivot = pivot.rename(columns={"heuristic": "Heuristic", "rl": "BC+DAgger"})

    fig, ax = plt.subplots(figsize=(10, 4.5))
    pivot.plot(kind="bar", ax=ax, color=["#6e6e6e", "#2f6db3"])
    ax.set_title("Robustness Suite Win Rate")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_behavior_tradeoffs(main_df: pd.DataFrame, output_dir: Path) -> None:
    if main_df.empty:
        return

    plot_df = main_df.set_index("method_label")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    plot_df["recruited_neutrophils"].plot(kind="bar", ax=axes[0], color="#7b5ea7")
    axes[0].set_title("Recruited Neutrophils")
    axes[0].set_ylabel("Count")

    plot_df["signal_actions"].plot(kind="bar", ax=axes[1], color="#bf7f2f")
    axes[1].set_title("Signal Actions")
    axes[1].set_ylabel("Count")

    plot_df["episode_length"].plot(kind="bar", ax=axes[2], color="#3c8f8f")
    axes[2].set_title("Episode Length")
    axes[2].set_ylabel("Steps")

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_dir / "behavior_tradeoffs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_artifacts(final_eval_dir: Path, robustness_dir: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    main_df, missing = _load_method_rows(final_eval_dir)
    repro_df = _load_reproducibility_table(final_eval_dir)
    robustness_df = _load_robustness_table(robustness_dir)

    if not main_df.empty:
        main_df.to_csv(output_dir / "main_comparison.csv", index=False)
        (output_dir / "main_comparison.md").write_text(
            _dataframe_to_markdown(main_df[["method_label", *MAIN_METRICS, *BEHAVIOR_METRICS]]),
            encoding="utf-8",
        )

    repro_df.to_csv(output_dir / "reproducibility_table.csv", index=False)
    (output_dir / "reproducibility_table.md").write_text(
        _dataframe_to_markdown(repro_df),
        encoding="utf-8",
    )

    robustness_df.to_csv(output_dir / "robustness_table.csv", index=False)
    (output_dir / "robustness_table.md").write_text(
        _dataframe_to_markdown(robustness_df),
        encoding="utf-8",
    )

    _write_missing_report(output_dir, missing)
    _plot_main_comparison(main_df, output_dir)
    _plot_robustness_summary(robustness_df, output_dir)
    _plot_behavior_tradeoffs(main_df, output_dir)

    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready tables and figures from curated result artifacts.")
    parser.add_argument("--final-eval-dir", default="results/final_eval")
    parser.add_argument("--robustness-dir", default="results/robustness/hier_dagger_main")
    parser.add_argument("--output-dir", default="results/paper_ready")
    args = parser.parse_args()

    missing = build_artifacts(
        final_eval_dir=Path(args.final_eval_dir),
        robustness_dir=Path(args.robustness_dir),
        output_dir=Path(args.output_dir),
    )
    if missing:
        print("Built paper artifacts with missing comparison rows:")
        for item in sorted(set(missing)):
            print(f"  - {item}")
    else:
        print("Built paper artifacts with all expected comparison rows present.")


if __name__ == "__main__":
    main()
