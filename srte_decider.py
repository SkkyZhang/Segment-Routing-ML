#!/usr/bin/env python3
"""Offline SR-TE decider: forecast future MLU, then pick candidate_path_id.

This script is meant for the next step after ML.py:
1) load the trained future-MLU model (for example model_network_mlu_pct_60s.joblib)
2) load the engineered dataset produced by ML.py (dataset_full.csv)
3) score candidate paths with a simple reroute simulator
4) output a decision JSON with candidate_path_id

Important: this is a *simple* decision layer, not a full online controller.
It assumes you already have a candidate path map that tells the script which
router:iface links belong to each candidate path.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


LINK_TX_SUFFIX = "__tx_mbps"
LINK_CAP_SUFFIX = "__link_capacity_mbps"
LINK_UTIL_SUFFIX = "__util_pct"


def sanitize_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast future MLU and choose candidate_path_id")
    parser.add_argument("--dataset", default="data/ml_results/dataset_full.csv", help="Path to dataset_full.csv")
    parser.add_argument(
        "--model",
        default="data/ml_results/model_network_mlu_pct_60s.joblib",
        help="Path to trained model joblib",
    )
    parser.add_argument(
        "--paths-json",
        required=True,
        help="JSON file describing candidate paths and their router:iface memberships",
    )
    parser.add_argument(
        "--state-file",
        default="data/controller_state.json",
        help="Optional runtime state JSON; used to read current candidate_path_id / cooldown",
    )
    parser.add_argument(
        "--run-key",
        default="",
        help="Optional run_key to score. Default: latest row across all runs",
    )
    parser.add_argument(
        "--timestamp",
        default="",
        help="Optional timestamp to score; picks latest row at or before this timestamp",
    )
    parser.add_argument(
        "--target-col",
        default="network_mlu_pct",
        help="Forecast target column used during training",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=70.0,
        help="If predicted future MLU stays below this and cooldown is off, keep current path",
    )
    parser.add_argument(
        "--elephant-rate-mbps",
        type=float,
        default=900.0,
        help="Estimated elephant flow rate moved by reroute",
    )
    parser.add_argument(
        "--switch-penalty-pct",
        type=float,
        default=2.0,
        help="Penalty added to candidates that switch away from current path",
    )
    parser.add_argument(
        "--min-improvement-pct",
        type=float,
        default=1.0,
        help="Only switch if the best candidate improves estimated future MLU by at least this amount",
    )
    parser.add_argument(
        "--force-evaluate",
        action="store_true",
        help="Ignore cooldown and threshold gating; always evaluate all candidates",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional output path for decision JSON",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not path or not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_paths_map(path: str) -> Dict[str, Dict[str, Any]]:
    raw = load_json(path)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid or empty paths JSON: {path}")

    out: Dict[str, Dict[str, Any]] = {}
    for candidate_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Path entry {candidate_id!r} must be a JSON object")
        interfaces = spec.get("interfaces", [])
        if not isinstance(interfaces, list) or not interfaces:
            raise ValueError(f"Path entry {candidate_id!r} needs a non-empty interfaces list")
        link_keys = [router_iface_to_link_key(x) for x in interfaces]
        out[str(candidate_id)] = {
            "interfaces": interfaces,
            "link_keys": link_keys,
            "segment_list": spec.get("segment_list", ""),
            "description": spec.get("description", ""),
        }
    return out


def router_iface_to_link_key(router_iface: str) -> str:
    if ":" not in router_iface:
        raise ValueError(f"Expected router:iface format, got {router_iface!r}")
    router, iface = router_iface.split(":", 1)
    return f"{sanitize_key(router)}__{sanitize_key(iface)}"


def choose_row(df: pd.DataFrame, run_key: str, timestamp: str) -> pd.Series:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values(["timestamp", "run_key"]).reset_index(drop=True)
    if run_key:
        work = work[work["run_key"].astype(str) == str(run_key)].copy()
    if work.empty:
        raise ValueError("No rows left after filtering by run_key")
    if timestamp:
        ts = pd.to_datetime(timestamp, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Invalid --timestamp: {timestamp}")
        work = work[work["timestamp"] <= ts].copy()
        if work.empty:
            raise ValueError("No rows at or before the requested timestamp")
    return work.iloc[-1]


def clean_feature_row(row_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    drop_cols = {
        f"target_{target_col}_60s",
        f"target_{target_col}_300s",
        f"target_{target_col}_600s",
        f"target_{target_col}_900s",
        "timestamp",
        "wall_time_epoch_ms",
        "sample_id",
    }
    return row_df.drop(columns=[c for c in drop_cols if c in row_df.columns], errors="ignore")


def extract_link_state(row: pd.Series) -> Dict[str, Dict[str, float]]:
    link_state: Dict[str, Dict[str, float]] = {}
    for col in row.index:
        if col.endswith(LINK_TX_SUFFIX):
            link_key = col[: -len(LINK_TX_SUFFIX)]
            tx = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            cap_col = f"{link_key}{LINK_CAP_SUFFIX}"
            util_col = f"{link_key}{LINK_UTIL_SUFFIX}"
            cap = pd.to_numeric(pd.Series([row.get(cap_col, np.nan)]), errors="coerce").iloc[0]
            util = pd.to_numeric(pd.Series([row.get(util_col, np.nan)]), errors="coerce").iloc[0]
            if pd.isna(tx):
                tx = np.nan
            if pd.isna(cap):
                cap = np.nan
            if pd.isna(util):
                util = np.nan
            link_state[link_key] = {
                "tx_mbps": float(tx) if np.isfinite(tx) else np.nan,
                "capacity_mbps": float(cap) if np.isfinite(cap) else np.nan,
                "util_pct": float(util) if np.isfinite(util) else np.nan,
            }
    if not link_state:
        raise ValueError("No per-link __tx_mbps columns found in dataset row")
    return link_state


def simulate_candidate(
    link_state: Dict[str, Dict[str, float]],
    current_path_links: List[str],
    candidate_links: List[str],
    elephant_rate_mbps: float,
    future_delta_pct: float,
) -> Tuple[float, Dict[str, float]]:
    current_set = set(current_path_links)
    candidate_set = set(candidate_links)
    adjusted_utils: Dict[str, float] = {}

    for link_key, vals in link_state.items():
        tx = vals.get("tx_mbps", np.nan)
        cap = vals.get("capacity_mbps", np.nan)
        if not np.isfinite(tx) or not np.isfinite(cap) or cap <= 0:
            continue

        adjusted_tx = float(tx)
        if link_key in current_set and link_key not in candidate_set:
            adjusted_tx = max(0.0, adjusted_tx - elephant_rate_mbps)
        if link_key in candidate_set and link_key not in current_set:
            adjusted_tx = adjusted_tx + elephant_rate_mbps

        adjusted_util = (adjusted_tx / cap) * 100.0
        estimated_future_util = max(0.0, adjusted_util + future_delta_pct)
        adjusted_utils[link_key] = estimated_future_util

    est_mlu = max(adjusted_utils.values()) if adjusted_utils else math.inf
    return float(est_mlu), adjusted_utils


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    df = pd.read_csv(dataset_path)
    if "timestamp" not in df.columns:
        raise SystemExit("dataset_full.csv is missing timestamp")
    if args.target_col not in df.columns:
        raise SystemExit(f"dataset_full.csv is missing target column {args.target_col!r}")

    model = joblib.load(model_path)
    paths_map = load_paths_map(args.paths_json)
    state = load_json(args.state_file)

    row = choose_row(df, run_key=args.run_key, timestamp=args.timestamp)
    row_df = pd.DataFrame([row])
    X_row = clean_feature_row(row_df, target_col=args.target_col)
    pred_future_mlu = float(model.predict(X_row)[0])

    current_mlu = pd.to_numeric(pd.Series([row.get(args.target_col, np.nan)]), errors="coerce").iloc[0]
    current_mlu = float(current_mlu) if np.isfinite(current_mlu) else pred_future_mlu
    future_delta_pct = pred_future_mlu - current_mlu

    current_candidate = str(
        state.get("candidate_path_id")
        or row.get("candidate_path_id")
        or ""
    ).strip()
    cooldown_active = int(pd.to_numeric(pd.Series([state.get("cooldown_active", row.get("cooldown_active", 0))]), errors="coerce").fillna(0).iloc[0])

    link_state = extract_link_state(row)
    if current_candidate and current_candidate in paths_map:
        current_links = paths_map[current_candidate]["link_keys"]
    else:
        current_links = []

    candidate_scores: List[Dict[str, Any]] = []
    for candidate_id, spec in paths_map.items():
        est_mlu, est_utils = simulate_candidate(
            link_state=link_state,
            current_path_links=current_links,
            candidate_links=spec["link_keys"],
            elephant_rate_mbps=args.elephant_rate_mbps,
            future_delta_pct=future_delta_pct,
        )
        score = est_mlu
        if current_candidate and candidate_id != current_candidate:
            score += args.switch_penalty_pct
        hottest = sorted(est_utils.items(), key=lambda kv: kv[1], reverse=True)[:5]
        candidate_scores.append(
            {
                "candidate_path_id": candidate_id,
                "score": round(float(score), 3),
                "estimated_future_mlu_pct": round(float(est_mlu), 3),
                "segment_list": spec.get("segment_list", ""),
                "interfaces": spec.get("interfaces", []),
                "top_hot_links": [
                    {"link_key": lk, "estimated_future_util_pct": round(float(util), 3)}
                    for lk, util in hottest
                ],
            }
        )

    candidate_scores = sorted(candidate_scores, key=lambda x: x["score"])
    if not candidate_scores:
        raise SystemExit("No candidate paths to score")

    best = candidate_scores[0]
    reason = "best_score"
    chosen = best

    current_score = None
    for item in candidate_scores:
        if item["candidate_path_id"] == current_candidate:
            current_score = item
            break

    if not args.force_evaluate:
        if cooldown_active == 1 and current_score is not None:
            chosen = current_score
            reason = "cooldown_active_keep_current"
        elif pred_future_mlu < args.threshold_pct and current_score is not None:
            chosen = current_score
            reason = "below_threshold_keep_current"
        elif current_score is not None:
            improvement = current_score["estimated_future_mlu_pct"] - best["estimated_future_mlu_pct"]
            if best["candidate_path_id"] != current_candidate and improvement < args.min_improvement_pct:
                chosen = current_score
                reason = "improvement_too_small_keep_current"

    decision = {
        "timestamp_scored": str(row["timestamp"]),
        "run_key": str(row.get("run_key", "")),
        "experiment_id": str(row.get("experiment_id", "")),
        "mode": str(row.get("mode", "")),
        "active_policy": str(row.get("active_policy", "")),
        "current_candidate_path_id": current_candidate,
        "predicted_future_mlu_pct": round(pred_future_mlu, 3),
        "current_mlu_pct": round(current_mlu, 3),
        "future_delta_pct": round(future_delta_pct, 3),
        "threshold_pct": args.threshold_pct,
        "cooldown_active": cooldown_active,
        "decision_reason": reason,
        "chosen_candidate_path_id": chosen["candidate_path_id"],
        "chosen_segment_list": chosen.get("segment_list", ""),
        "chosen_estimated_future_mlu_pct": chosen["estimated_future_mlu_pct"],
        "switch_penalty_pct": args.switch_penalty_pct,
        "min_improvement_pct": args.min_improvement_pct,
        "elephant_rate_mbps": args.elephant_rate_mbps,
        "candidate_scores": candidate_scores,
    }

    text = json.dumps(decision, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[INFO] Wrote decision to {out_path}")


if __name__ == "__main__":
    main()