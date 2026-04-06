#!/usr/bin/env python3
"""Validate SR-TE decider outputs for logical consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


KEEP_CURRENT_REASONS = {
    "below_threshold_keep_current",
    "cooldown_active_keep_current",
    "improvement_too_small_keep_current",
    "best_candidate_path_still_too_hot_keep_current",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a decider JSON output")
    parser.add_argument("--decision-json", required=True, help="Path to decision JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any candidate path has obviously suspicious metrics",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_candidate(scores: List[Dict[str, Any]], candidate_id: str) -> Dict[str, Any] | None:
    for item in scores:
        if item.get("candidate_path_id") == candidate_id:
            return item
    return None


def main() -> None:
    args = parse_args()
    decision_path = Path(args.decision_json)
    if not decision_path.exists():
        raise SystemExit(f"decision json not found: {decision_path}")

    obj = load_json(decision_path)
    scores = obj.get("candidate_scores", [])
    if not isinstance(scores, list) or not scores:
        raise SystemExit("candidate_scores missing or empty")

    errors: List[str] = []
    warnings: List[str] = []

    chosen_id = obj.get("chosen_candidate_path_id", "")
    current_id = obj.get("current_candidate_path_id", "")
    reason = obj.get("decision_reason", "")
    scoring_basis = obj.get("scoring_basis", "")
    threshold_pct = float(obj.get("threshold_pct", 0.0) or 0.0)
    predicted_future_mlu = float(obj.get("predicted_future_mlu_pct", 0.0) or 0.0)

    chosen = find_candidate(scores, chosen_id)
    current = find_candidate(scores, current_id) if current_id else None
    best = min(scores, key=lambda item: float(item.get("score", float("inf"))))

    if chosen is None:
        errors.append(f"chosen_candidate_path_id {chosen_id!r} not found in candidate_scores")
    if scoring_basis not in {"physical_edges", "interface_links"}:
        errors.append(f"unexpected scoring_basis: {scoring_basis!r}")

    if reason == "best_score":
        if chosen is None or chosen.get("candidate_path_id") != best.get("candidate_path_id"):
            errors.append(
                f"reason=best_score but chosen={chosen_id!r} is not best={best.get('candidate_path_id')!r}"
            )

    if reason in KEEP_CURRENT_REASONS:
        if not current_id:
            errors.append(f"reason={reason} but current_candidate_path_id is empty")
        elif chosen_id != current_id:
            errors.append(
                f"reason={reason} but chosen={chosen_id!r} does not match current={current_id!r}"
            )

    if reason == "below_threshold_keep_current" and predicted_future_mlu >= threshold_pct:
        errors.append(
            f"reason=below_threshold_keep_current but predicted_future_mlu_pct={predicted_future_mlu:.3f} "
            f">= threshold_pct={threshold_pct:.3f}"
        )

    if chosen is not None:
        chosen_mlu = float(chosen.get("estimated_future_mlu_pct", 0.0) or 0.0)
        recorded = float(obj.get("chosen_estimated_future_mlu_pct", 0.0) or 0.0)
        if abs(chosen_mlu - recorded) > 1e-6:
            errors.append(
                f"chosen_estimated_future_mlu_pct={recorded:.3f} "
                f"does not match chosen candidate metric={chosen_mlu:.3f}"
            )

    for item in scores:
        candidate_id = str(item.get("candidate_path_id", ""))
        score = float(item.get("score", 0.0) or 0.0)
        est_mlu = float(item.get("estimated_future_mlu_pct", 0.0) or 0.0)
        path_hot = item.get("estimated_path_hot_util_pct")
        if score < est_mlu - 1e-6:
            errors.append(
                f"{candidate_id}: score={score:.3f} is lower than estimated_future_mlu_pct={est_mlu:.3f}"
            )
        if path_hot is None:
            warnings.append(f"{candidate_id}: estimated_path_hot_util_pct is missing")
        else:
            path_hot = float(path_hot)
            if path_hot > est_mlu + 1e-6:
                warnings.append(
                    f"{candidate_id}: estimated_path_hot_util_pct={path_hot:.3f} "
                    f"is higher than estimated_future_mlu_pct={est_mlu:.3f}"
                )

        loads = item.get("path_edge_loads") or item.get("path_link_loads") or []
        if not loads:
            warnings.append(f"{candidate_id}: no per-path load details found")
        elif args.strict:
            all_zero = all(float((x.get("estimated_util_pct") or 0.0)) == 0.0 for x in loads)
            if all_zero and est_mlu > 1.0:
                warnings.append(
                    f"{candidate_id}: all path load details are zero while estimated_future_mlu_pct={est_mlu:.3f}"
                )

    print(f"[INFO] decision={decision_path}")
    print(f"[INFO] chosen={chosen_id} current={current_id} reason={reason} scoring_basis={scoring_basis}")
    print(f"[INFO] best_by_score={best.get('candidate_path_id')} best_score={float(best.get('score', 0.0)):.3f}")

    if warnings:
        print("[WARN] validation warnings:")
        for item in warnings:
            print(f"  - {item}")

    if errors:
        print("[FAIL] validation errors:")
        for item in errors:
            print(f"  - {item}")
        raise SystemExit(1)

    print("[PASS] decision output is logically consistent")


if __name__ == "__main__":
    main()
