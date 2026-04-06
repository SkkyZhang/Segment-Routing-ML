#!/usr/bin/env python3
"""Batch replay SR-TE decisions and summarize chosen path counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import srte_decider as sd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay decider choices over a dataset and export stats")
    parser.add_argument("--dataset", required=True, help="Path to dataset_full.csv")
    parser.add_argument("--model", required=True, help="Path to trained model joblib")
    parser.add_argument("--paths-json", required=True, help="Candidate paths JSON")
    parser.add_argument("--state-file", default="data/controller_state.json", help="Runtime state JSON")
    parser.add_argument("--lab", default="lab.clab.yml", help="lab.clab.yml used for physical link mapping")
    parser.add_argument("--output-dir", default="data/path_choice_analysis", help="Directory for CSV and plots")
    parser.add_argument("--sample-step", type=int, default=30, help="Evaluate every Nth row per run")
    parser.add_argument("--target-col", default="network_mlu_pct", help="Forecast target column")
    parser.add_argument("--threshold-pct", type=float, default=70.0)
    parser.add_argument("--elephant-rate-mbps", type=float, default=900.0)
    parser.add_argument("--switch-penalty-pct", type=float, default=2.0)
    parser.add_argument("--min-improvement-pct", type=float, default=1.0)
    parser.add_argument("--max-path-util-pct", type=float, default=85.0)
    parser.add_argument("--force-evaluate", action="store_true", help="Ignore gating and always take best score")
    return parser.parse_args()


def build_sample_rows(df: pd.DataFrame, sample_step: int) -> pd.DataFrame:
    groups: List[pd.DataFrame] = []
    for _, group in df.groupby("run_key", sort=True):
        sampled = group.iloc[:: max(1, sample_step)].copy()
        if sampled.empty:
            continue
        if pd.Timestamp(sampled.iloc[-1]["timestamp"]) != pd.Timestamp(group.iloc[-1]["timestamp"]):
            sampled = pd.concat([sampled, group.iloc[[-1]]], ignore_index=True)
        groups.append(sampled)
    if not groups:
        return pd.DataFrame()
    return pd.concat(groups, ignore_index=True)


def save_count_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    pivot = summary_df.pivot_table(
        index="run_key",
        columns="chosen_candidate_path_id",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Run")
    ax.set_ylabel("Chosen Count")
    ax.set_title("Chosen path counts by run")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Path", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_reason_plot(reason_df: pd.DataFrame, output_path: Path) -> None:
    if reason_df.empty:
        return
    pivot = reason_df.pivot_table(
        index="run_key",
        columns="decision_reason",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_xlabel("Run")
    ax.set_ylabel("Count")
    ax.set_title("Decision reasons by run")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Reason", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["run_key", "timestamp"]).reset_index(drop=True)

    model = joblib.load(args.model)
    peer_map = sd.parse_lab_peer_map(args.lab)
    paths_map = sd.load_paths_map(args.paths_json, peer_map=peer_map)
    state = sd.load_json(args.state_file)

    sampled = build_sample_rows(df, args.sample_step)
    if sampled.empty:
        raise SystemExit("No rows available for analysis")

    decisions: List[Dict[str, Any]] = []
    for _, row in sampled.iterrows():
        decision = sd.build_decision(
            row=row,
            model=model,
            paths_map=paths_map,
            state=state,
            target_col=args.target_col,
            threshold_pct=args.threshold_pct,
            elephant_rate_mbps=args.elephant_rate_mbps,
            switch_penalty_pct=args.switch_penalty_pct,
            min_improvement_pct=args.min_improvement_pct,
            max_path_util_pct=args.max_path_util_pct,
            force_evaluate=args.force_evaluate,
            peer_map=peer_map,
        )
        decisions.append(
            {
                "run_key": decision["run_key"],
                "timestamp": decision["timestamp_scored"],
                "mode": decision["mode"],
                "active_policy": decision["active_policy"],
                "current_candidate_path_id": decision["current_candidate_path_id"],
                "chosen_candidate_path_id": decision["chosen_candidate_path_id"],
                "decision_reason": decision["decision_reason"],
                "predicted_future_mlu_pct": decision["predicted_future_mlu_pct"],
                "current_mlu_pct": decision["current_mlu_pct"],
                "scoring_basis": decision.get("scoring_basis", ""),
            }
        )

    decisions_df = pd.DataFrame(decisions)
    decisions_csv = output_dir / "path_choice_replay.csv"
    decisions_df.to_csv(decisions_csv, index=False)

    counts_df = (
        decisions_df.groupby(["run_key", "mode", "chosen_candidate_path_id"])
        .size()
        .reset_index(name="count")
        .sort_values(["run_key", "chosen_candidate_path_id"])
    )
    counts_csv = output_dir / "path_choice_counts.csv"
    counts_df.to_csv(counts_csv, index=False)

    reason_df = (
        decisions_df.groupby(["run_key", "mode", "decision_reason"])
        .size()
        .reset_index(name="count")
        .sort_values(["run_key", "decision_reason"])
    )
    reason_csv = output_dir / "path_choice_reasons.csv"
    reason_df.to_csv(reason_csv, index=False)

    save_count_plot(counts_df, output_dir / "path_choice_counts.png")
    save_reason_plot(reason_df, output_dir / "path_choice_reasons.png")

    print(f"[INFO] Wrote replay rows to {decisions_csv}")
    print(f"[INFO] Wrote path counts to {counts_csv}")
    print(f"[INFO] Wrote reason counts to {reason_csv}")
    print("[INFO] Path choice counts summary:")
    print(counts_df.to_string(index=False))


if __name__ == "__main__":
    main()
