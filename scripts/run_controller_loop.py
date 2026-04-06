#!/usr/bin/env python3
"""Minimal 60s controller loop for proposal-aligned SR-TE experiments.

This script does not push policy into FRR/MPLS directly. Instead, it closes the
loop at the project-control layer by:
1. loading the latest engineered dataset row,
2. identifying the current Top-K elephant window,
3. running the decider,
4. updating data/controller_state.json with the new decision and cooldown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import srte_decider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one 60s SR-TE controller iteration.")
    parser.add_argument("--dataset", default="data/ml_results/dataset_full.csv")
    parser.add_argument("--model", default="data/ml_results/model_network_mlu_pct_60s.joblib")
    parser.add_argument("--paths-json", default="candidate_paths_example.json")
    parser.add_argument("--lab", default="lab.clab.yml")
    parser.add_argument("--state-file", default="data/controller_state.json")
    parser.add_argument("--out-json", default="data/controller_decision_latest.json")
    parser.add_argument("--run-key", default="", help="Optional run_key filter")
    parser.add_argument("--timestamp", default="", help="Optional timestamp to score")
    parser.add_argument("--target-col", default="network_mlu_pct")
    parser.add_argument("--threshold-pct", type=float, default=70.0)
    parser.add_argument("--elephant-rate-mbps", type=float, default=900.0)
    parser.add_argument("--switch-penalty-pct", type=float, default=2.0)
    parser.add_argument("--min-improvement-pct", type=float, default=1.0)
    parser.add_argument("--max-path-util-pct", type=float, default=85.0)
    parser.add_argument("--cooldown-seconds", type=int, default=60)
    parser.add_argument("--topk-csv", default="", help="Optional topk_elephants CSV")
    parser.add_argument("--force-evaluate", action="store_true")
    return parser.parse_args()


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def auto_topk_csv(data_dir: Path, run_key: str) -> str:
    candidates = sorted(data_dir.glob(f"topk_elephants_{run_key}.csv"))
    if candidates:
        return str(candidates[-1])
    all_candidates = sorted(data_dir.glob("topk_elephants_*.csv"))
    return str(all_candidates[-1]) if all_candidates else ""


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    state_path = Path(args.state_file)
    out_path = Path(args.out_json)

    df = pd.read_csv(dataset_path, low_memory=False)
    row = srte_decider.choose_row(df, run_key=args.run_key, timestamp=args.timestamp)
    model = joblib.load(model_path)
    peer_map = srte_decider.parse_lab_peer_map(args.lab)
    paths_map = srte_decider.load_paths_map(args.paths_json, peer_map=peer_map)
    state = load_state(state_path)

    topk_csv = args.topk_csv or auto_topk_csv(dataset_path.parent.parent if dataset_path.parent.name.startswith("ml_results") else Path("data"), str(row.get("run_key", "")))
    topk_context = srte_decider.read_topk_context(
        path=topk_csv,
        run_key=str(row.get("run_key", "")),
        timestamp=pd.to_datetime(row["timestamp"], errors="coerce"),
    ) if topk_csv else {}

    decision = srte_decider.build_decision(
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
        topk_context=topk_context,
    )

    prev_policy_seq = int(pd.to_numeric(pd.Series([state.get("policy_seq", 0)]), errors="coerce").fillna(0).iloc[0])
    path_changed = str(decision.get("current_candidate_path_id", "")) != str(decision.get("chosen_candidate_path_id", ""))
    now_ts = pd.to_datetime(decision["timestamp_scored"], errors="coerce")
    cooldown_until = (now_ts + pd.Timedelta(seconds=args.cooldown_seconds)).isoformat() if path_changed and not pd.isna(now_ts) else state.get("cooldown_until", "")

    current_score = next(
        (x for x in decision["candidate_scores"] if x["candidate_path_id"] == decision.get("current_candidate_path_id")),
        None,
    )
    gain_mlu_pct = None
    if current_score is not None:
        gain_mlu_pct = round(float(current_score["estimated_future_mlu_pct"]) - float(decision["chosen_estimated_future_mlu_pct"]), 3)

    new_state = dict(state)
    new_state.update(
        {
            "mode": decision.get("mode") or "ml_dynamic",
            "traffic_profile": state.get("traffic_profile", row.get("traffic_profile", "medium")),
            "active_policy": "srte_ml_dynamic",
            "policy_id": f"ml_policy_{decision['chosen_candidate_path_id']}",
            "candidate_path_id": decision["chosen_candidate_path_id"],
            "policy_seq": prev_policy_seq + (1 if path_changed else 0),
            "decision_id": f"ml_decision_{str(decision['timestamp_scored']).replace(':', '').replace('-', '').replace('T', '_')}",
            "reroute_event": 1 if path_changed else 0,
            "cooldown_active": 1 if path_changed else 0,
            "event": "reroute" if path_changed else "hold",
            "event_ts": decision["timestamp_scored"],
            "cooldown_until": cooldown_until,
            "elephant_path_hint": decision["chosen_candidate_path_id"],
            "elephant_flow_id": decision.get("target_elephant_event_id", ""),
            "elephant_topk_rank": decision.get("target_elephant_rank", ""),
            "ingress_node": state.get("ingress_node", row.get("ingress_node", "r1")),
            "segment_list": next((x.get("segment_list", "") for x in decision["candidate_scores"] if x["candidate_path_id"] == decision["chosen_candidate_path_id"]), ""),
            "controller_epoch_s": int(now_ts.timestamp()) if not pd.isna(now_ts) else state.get("controller_epoch_s", 0),
            "gain_mlu_pct": gain_mlu_pct,
            "gain_rtt_ms": state.get("gain_rtt_ms", ""),
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    state_path.write_text(json.dumps(new_state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"[INFO] Wrote decision to {out_path}")
    print(f"[INFO] Updated controller state at {state_path}")


if __name__ == "__main__":
    main()
