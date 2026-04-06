#!/usr/bin/env python3
"""Summarize SR-TE baseline experiment results for 60-second workflow.

This script reads collected raw experiment outputs from data/ and optionally
merges ML evaluation results from an ml_results directory. It produces:
- per-run baseline summary CSV
- per-mode aggregate summary CSV
- optional merged ML metrics by mode
- a few comparison plots for MLU, RTT, loss, and control overhead
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SR-TE baseline experiment results.")
    parser.add_argument("--data-dir", default="data", help="Directory containing raw experiment CSV files")
    parser.add_argument(
        "--ml-output-dir",
        default="data/ml_results",
        help="Optional ML output directory containing metrics.csv and metrics_by_group.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/experiment_summary",
        help="Directory for summary CSV files and plots",
    )
    parser.add_argument(
        "--hot-threshold-pct",
        type=float,
        default=70.0,
        help="Threshold used to define hot/overloaded samples",
    )
    return parser.parse_args()


def extract_run_key(path: Path) -> Optional[str]:
    match = TIMESTAMP_RE.search(path.name)
    return match.group(1) if match else None


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None


def resolve_ml_output_dir(path: Path) -> Path:
    if path.exists():
        return path
    parent = path.parent if path.parent != Path("") else Path(".")
    candidates = []
    latest_dir = parent / "ml_results_latest"
    if latest_dir.exists():
        candidates.append(latest_dir)
    candidates.extend(sorted(parent.glob("ml_results_run_*")))
    if candidates:
        chosen = candidates[-1]
        print(f"[INFO] Requested ML output dir {path} was not found; using {chosen} instead")
        return chosen
    return path


def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def summarize_run(run_key: str, data_dir: Path, hot_threshold_pct: float) -> Optional[Dict[str, object]]:
    telemetry_path = data_dir / f"telemetry_wide_{run_key}.csv"
    probe_path = data_dir / f"probe_rtt_{run_key}.csv"
    control_path = data_dir / f"control_overhead_{run_key}.csv"

    telemetry = read_csv_if_exists(telemetry_path)
    if telemetry is None or telemetry.empty:
        return None

    probe = read_csv_if_exists(probe_path)
    control = read_csv_if_exists(control_path)

    telemetry = telemetry.copy()
    telemetry["network_mlu_pct"] = safe_float(telemetry.get("network_mlu_pct", pd.Series(dtype=float)))
    telemetry["elapsed_s"] = safe_float(telemetry.get("elapsed_s", pd.Series(dtype=float)))

    mode = str(telemetry.get("mode", pd.Series(["unknown"])).iloc[0])
    active_policy = str(telemetry.get("active_policy", pd.Series(["unknown"])).iloc[0])
    experiment_id = str(telemetry.get("experiment_id", pd.Series(["unknown"])).iloc[0])

    summary: Dict[str, object] = {
        "run_key": run_key,
        "experiment_id": experiment_id,
        "mode": mode,
        "active_policy": active_policy,
        "samples": int(len(telemetry)),
        "duration_s": float(telemetry["elapsed_s"].dropna().max()) if telemetry["elapsed_s"].notna().any() else None,
        "mlu_mean_pct": float(telemetry["network_mlu_pct"].dropna().mean()) if telemetry["network_mlu_pct"].notna().any() else None,
        "mlu_p95_pct": float(telemetry["network_mlu_pct"].dropna().quantile(0.95)) if telemetry["network_mlu_pct"].notna().any() else None,
        "mlu_max_pct": float(telemetry["network_mlu_pct"].dropna().max()) if telemetry["network_mlu_pct"].notna().any() else None,
        "hot_sample_ratio": float((telemetry["network_mlu_pct"] >= hot_threshold_pct).mean()) if telemetry["network_mlu_pct"].notna().any() else None,
    }

    if probe is not None and not probe.empty:
        probe = probe.copy()
        for col in ["ping_rtt_avg_ms", "ping_rtt_p95_ms", "ping_rtt_p99_ms", "ping_loss_pct"]:
            if col in probe.columns:
                probe[col] = safe_float(probe[col])
        summary.update(
            {
                "probe_rows": int(len(probe)),
                "rtt_avg_mean_ms": float(probe["ping_rtt_avg_ms"].dropna().mean()) if "ping_rtt_avg_ms" in probe and probe["ping_rtt_avg_ms"].notna().any() else None,
                "rtt_p95_mean_ms": float(probe["ping_rtt_p95_ms"].dropna().mean()) if "ping_rtt_p95_ms" in probe and probe["ping_rtt_p95_ms"].notna().any() else None,
                "rtt_p99_mean_ms": float(probe["ping_rtt_p99_ms"].dropna().mean()) if "ping_rtt_p99_ms" in probe and probe["ping_rtt_p99_ms"].notna().any() else None,
                "rtt_p99_max_ms": float(probe["ping_rtt_p99_ms"].dropna().max()) if "ping_rtt_p99_ms" in probe and probe["ping_rtt_p99_ms"].notna().any() else None,
                "loss_mean_pct": float(probe["ping_loss_pct"].dropna().mean()) if "ping_loss_pct" in probe and probe["ping_loss_pct"].notna().any() else None,
                "loss_max_pct": float(probe["ping_loss_pct"].dropna().max()) if "ping_loss_pct" in probe and probe["ping_loss_pct"].notna().any() else None,
            }
        )

    if control is not None and not control.empty:
        control = control.copy()
        for col in ["policy_changed", "decision_changed", "path_changed", "cooldown_active", "reroute_event"]:
            if col in control.columns:
                control[col] = safe_float(control[col]).fillna(0.0)
        summary.update(
            {
                "control_rows": int(len(control)),
                "policy_updates": int(control["policy_changed"].sum()) if "policy_changed" in control else 0,
                "decision_updates": int(control["decision_changed"].sum()) if "decision_changed" in control else 0,
                "path_changes": int(control["path_changed"].sum()) if "path_changed" in control else 0,
                "reroute_events": int(control["reroute_event"].sum()) if "reroute_event" in control else 0,
                "cooldown_active_ratio": float(control["cooldown_active"].mean()) if "cooldown_active" in control else 0.0,
            }
        )

    return summary


def plot_mode_bar(df: pd.DataFrame, value_col: str, output_path: Path, title: str, ylabel: str) -> None:
    plot_df = df.dropna(subset=[value_col]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("mode")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(plot_df["mode"], plot_df[value_col], color=["#3b82f6", "#10b981", "#f59e0b"][: len(plot_df)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Mode")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    ml_output_dir = resolve_ml_output_dir(Path(args.ml_output_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_keys = sorted(
        {
            key
            for key in (extract_run_key(p) for p in data_dir.glob("telemetry_wide_*.csv"))
            if key is not None
        }
    )
    if not run_keys:
        raise SystemExit(f"No telemetry_wide_*.csv files found in {data_dir}")

    rows: List[Dict[str, object]] = []
    for run_key in run_keys:
        row = summarize_run(run_key, data_dir, args.hot_threshold_pct)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("No valid runs could be summarized")

    run_df = pd.DataFrame(rows).sort_values(["mode", "run_key"]).reset_index(drop=True)
    run_df.to_csv(output_dir / "baseline_run_summary.csv", index=False)

    numeric_cols = [
        col
        for col in run_df.columns
        if col not in {"run_key", "experiment_id", "mode", "active_policy"}
        and pd.api.types.is_numeric_dtype(run_df[col])
    ]
    mode_df = run_df.groupby("mode", dropna=False)[numeric_cols].mean(numeric_only=True).reset_index()
    mode_df.to_csv(output_dir / "baseline_mode_summary.csv", index=False)

    ml_group_path = ml_output_dir / "metrics_by_group.csv"
    ml_metrics_path = ml_output_dir / "metrics.csv"
    if ml_group_path.exists():
        ml_group_df = pd.read_csv(ml_group_path)
        group_col = "group_by" if "group_by" in ml_group_df.columns else "group_name" if "group_name" in ml_group_df.columns else None
        if group_col is not None:
            ml_group_df = ml_group_df[ml_group_df[group_col].eq("mode")].copy()
        if not ml_group_df.empty:
            ml_group_df.to_csv(output_dir / "ml_metrics_by_mode.csv", index=False)
    if ml_metrics_path.exists():
        ml_metrics_df = pd.read_csv(ml_metrics_path)
        ml_metrics_df.to_csv(output_dir / "ml_metrics_overall.csv", index=False)

    plot_mode_bar(mode_df, "mlu_max_pct", output_dir / "baseline_mlu_max_by_mode.png", "Peak MLU by Mode", "Peak MLU (%)")
    plot_mode_bar(mode_df, "rtt_p99_mean_ms", output_dir / "baseline_rtt_p99_by_mode.png", "Mean Probe P99 RTT by Mode", "RTT (ms)")
    plot_mode_bar(mode_df, "loss_mean_pct", output_dir / "baseline_loss_by_mode.png", "Mean Probe Loss by Mode", "Loss (%)")
    plot_mode_bar(mode_df, "path_changes", output_dir / "baseline_path_changes_by_mode.png", "Mean Path Changes by Mode", "Count")

    summary_payload = {
        "run_count": int(len(run_df)),
        "modes": sorted(run_df["mode"].dropna().unique().tolist()),
        "data_dir": str(data_dir),
        "ml_output_dir": str(ml_output_dir),
        "output_dir": str(output_dir),
        "workflow_focus": "60-second SR-TE baseline comparison",
    }
    (output_dir / "summary_meta.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"[INFO] Wrote per-run summary to {output_dir / 'baseline_run_summary.csv'}")
    print(f"[INFO] Wrote per-mode summary to {output_dir / 'baseline_mode_summary.csv'}")
    if (output_dir / "ml_metrics_by_mode.csv").exists():
        print(f"[INFO] Wrote ML by-mode summary to {output_dir / 'ml_metrics_by_mode.csv'}")
    print(f"[INFO] Plots saved under {output_dir}")


if __name__ == "__main__":
    main()
