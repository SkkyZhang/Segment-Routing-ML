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


def normalize_node_name(text: str) -> str:
    return re.sub(r"^clab-[^-]+-", "", str(text).strip())


def normalize_iface_spec(text: str) -> str:
    node, iface = str(text).split(":", 1)
    return f"{normalize_node_name(node)}:{iface}"


def edge_id_from_nodes(left: str, right: str) -> str:
    return "__".join(sorted([normalize_node_name(left), normalize_node_name(right)]))


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
        "--lab",
        default="lab.clab.yml",
        help="Optional lab.clab.yml used to map path interfaces to physical links",
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
    parser.add_argument(
        "--max-path-util-pct",
        type=float,
        default=85.0,
        help="Soft limit for the hottest link on the chosen candidate path",
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


def parse_lab_peer_map(path: str) -> Dict[str, str]:
    p = Path(path)
    if not path or not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    mapping: Dict[str, str] = {}
    pattern = re.compile(r'endpoints:\s*\["([^"]+)",\s*"([^"]+)"\]')
    for left, right in pattern.findall(text):
        left_norm = normalize_iface_spec(left)
        right_norm = normalize_iface_spec(right)
        mapping[left_norm] = right_norm
        mapping[right_norm] = left_norm
    return mapping


def load_paths_map(path: str, peer_map: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
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
        edge_ids: List[str] = []
        if peer_map:
            for iface_spec in interfaces:
                iface_norm = normalize_iface_spec(str(iface_spec))
                peer = peer_map.get(iface_norm)
                if not peer:
                    continue
                edge_id = edge_id_from_nodes(iface_norm.split(":", 1)[0], peer.split(":", 1)[0])
                if edge_id not in edge_ids:
                    edge_ids.append(edge_id)
        out[str(candidate_id)] = {
            "interfaces": interfaces,
            "link_keys": link_keys,
            "edge_ids": edge_ids,
            "segment_list": spec.get("segment_list", ""),
            "description": spec.get("description", ""),
        }
    return out


def router_iface_to_link_key(router_iface: str) -> str:
    if ":" not in router_iface:
        raise ValueError(f"Expected router:iface format, got {router_iface!r}")
    router, iface = router_iface.split(":", 1)
    return f"{sanitize_key(router)}__{sanitize_key(iface)}"


def resolve_link_key_for_iface(
    router_iface: str,
    link_state: Dict[str, Dict[str, float]],
) -> Optional[str]:
    direct = router_iface_to_link_key(router_iface)
    if direct in link_state:
        return direct

    suffix = direct
    matches = [key for key in link_state if key == suffix or key.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]

    node, iface = router_iface.split(":", 1)
    node_token = sanitize_key(node)
    iface_token = sanitize_key(iface)
    for key in link_state:
        if key.endswith(f"{node_token}__{iface_token}"):
            return key
    return None


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
    out = row_df.drop(columns=[c for c in drop_cols if c in row_df.columns], errors="ignore").copy()
    leak_prefixes = ("target_", "gain_")
    leak_exact = {
        "run_key",
        "experiment_id",
        "policy_id",
        "decision_id",
        "event",
        "event_ts",
        "cooldown_until",
        "elephant_flow_id",
        "segment_list",
        "ping_rtts_ms",
    }
    leak_cols = [
        c for c in out.columns
        if c in leak_exact
        or c.startswith("source_")
        or any(c.startswith(prefix) for prefix in leak_prefixes)
    ]
    if leak_cols:
        out = out.drop(columns=leak_cols, errors="ignore")
    return out


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


def extract_edge_state(
    link_state: Dict[str, Dict[str, float]],
    peer_map: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    edge_state: Dict[str, Dict[str, Any]] = {}
    for iface_norm, peer_norm in peer_map.items():
        left_link_key = resolve_link_key_for_iface(iface_norm, link_state)
        if not left_link_key:
            continue
        vals = link_state.get(left_link_key)
        if vals is None:
            continue
        edge_id = edge_id_from_nodes(iface_norm.split(":", 1)[0], peer_norm.split(":", 1)[0])
        item = edge_state.setdefault(
            edge_id,
            {
                "tx_mbps": np.nan,
                "capacity_mbps": np.nan,
                "util_pct": np.nan,
                "member_link_keys": [],
            },
        )
        if left_link_key not in item["member_link_keys"]:
            item["member_link_keys"].append(left_link_key)
        tx_val = vals.get("tx_mbps", np.nan)
        prev_tx = item.get("tx_mbps", np.nan)
        if np.isfinite(tx_val) and (not np.isfinite(prev_tx) or float(tx_val) > float(prev_tx)):
            item["tx_mbps"] = float(tx_val)

        cap_val = vals.get("capacity_mbps", np.nan)
        prev_cap = item.get("capacity_mbps", np.nan)
        if np.isfinite(cap_val) and cap_val > 0:
            if not np.isfinite(prev_cap) or float(cap_val) < float(prev_cap):
                item["capacity_mbps"] = float(cap_val)

        util_val = vals.get("util_pct", np.nan)
        prev_util = item.get("util_pct", np.nan)
        if np.isfinite(util_val) and (not np.isfinite(prev_util) or float(util_val) > float(prev_util)):
            item["util_pct"] = float(util_val)
    return edge_state


def infer_current_candidate_from_load(
    link_state: Dict[str, Dict[str, float]],
    paths_map: Dict[str, Dict[str, Any]],
    edge_state: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    best_candidate = ""
    best_metrics: Dict[str, Any] = {}
    best_score = -math.inf

    use_edges = bool(edge_state and any(spec.get("edge_ids") for spec in paths_map.values()))
    for candidate_id, spec in paths_map.items():
        units = spec.get("edge_ids", []) if use_edges and spec.get("edge_ids") else spec.get("link_keys", [])
        state_map = edge_state if use_edges and spec.get("edge_ids") else link_state
        if not units:
            continue

        utils: List[float] = []
        positive_units = 0
        for unit in units:
            util = state_map.get(unit, {}).get("util_pct", np.nan)
            if np.isfinite(util):
                util_f = float(util)
                utils.append(util_f)
                if util_f >= 5.0:
                    positive_units += 1
            else:
                utils.append(0.0)

        coverage_ratio = positive_units / len(units)
        avg_util = float(np.mean(utils)) if utils else 0.0
        min_util = float(np.min(utils)) if utils else 0.0
        max_util = float(np.max(utils)) if utils else 0.0

        # Prefer paths whose load is both strong and continuous across the whole path,
        # which helps distinguish a true carried path from partial overlap.
        score = (coverage_ratio * 1000.0) + (min_util * 10.0) + avg_util + (max_util * 0.1)
        metrics = {
            "candidate_path_id": candidate_id,
            "coverage_ratio": round(coverage_ratio, 3),
            "avg_util_pct": round(avg_util, 3),
            "min_util_pct": round(min_util, 3),
            "max_util_pct": round(max_util, 3),
            "units": units,
            "basis": "physical_edges" if use_edges and spec.get("edge_ids") else "interface_links",
            "score": round(score, 3),
        }
        if score > best_score:
            best_score = score
            best_candidate = candidate_id
            best_metrics = metrics

    return best_candidate, best_metrics


def infer_current_candidate(
    row: pd.Series,
    state: Dict[str, Any],
    paths_map: Dict[str, Dict[str, Any]],
    link_state: Optional[Dict[str, Dict[str, float]]] = None,
    edge_state: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    if link_state:
        load_candidate, load_metrics = infer_current_candidate_from_load(
            link_state=link_state,
            paths_map=paths_map,
            edge_state=edge_state,
        )
        if load_candidate and load_metrics.get("coverage_ratio", 0.0) >= 0.75:
            return load_candidate, "observed_load", load_metrics

    alias_map = {
        "igp_shortest": "upper_corridor",
        "path_a": "upper_corridor",
        "path_b": "lower_corridor",
        "path_c": "cross_path",
        "path_d": "cross_path_reverse",
    }

    for value in [
        row.get("candidate_path_id"),
        row.get("elephant_path_hint"),
        state.get("candidate_path_id"),
        state.get("elephant_path_hint"),
    ]:
        candidate = str(value or "").strip()
        candidate = alias_map.get(candidate, candidate)
        if candidate in paths_map:
            return candidate, "state_or_row", {"raw_value": str(value or "")}

    active_policy = str(row.get("active_policy") or state.get("active_policy") or "").strip().lower()
    if active_policy == "igp" and "upper_corridor" in paths_map:
        return "upper_corridor", "policy_default", {"active_policy": active_policy}
    if active_policy == "srte_static" and "lower_corridor" in paths_map:
        return "lower_corridor", "policy_default", {"active_policy": active_policy}
    return "", "unknown", {}


def estimate_elephant_rate(row: pd.Series, fallback_rate_mbps: float) -> float:
    candidates = [
        row.get("topk_detail__top1_throughput_mbps_mean"),
        row.get("topk_detail__top1_throughput_mbps_peak"),
        row.get("intervals__active_throughput_sum"),
        row.get("events__active_throughput_sum"),
    ]
    for value in candidates:
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if np.isfinite(num) and float(num) > 0:
            return float(num)
    return float(fallback_rate_mbps)


def current_path_util_pct(path_links: List[str], link_state: Dict[str, Dict[str, float]]) -> float:
    vals: List[float] = []
    for link_key in path_links:
        util = link_state.get(link_key, {}).get("util_pct", np.nan)
        if np.isfinite(util):
            vals.append(float(util))
    return max(vals) if vals else math.nan


def simulate_candidate(
    link_state: Dict[str, Dict[str, float]],
    current_path_links: List[str],
    candidate_links: List[str],
    elephant_rate_mbps: float,
    future_delta_pct: float,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    current_set = set(current_path_links)
    candidate_set = set(candidate_links)
    adjusted_utils: Dict[str, float] = {}
    adjusted_txs: Dict[str, float] = {}

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
        adjusted_txs[link_key] = adjusted_tx

    est_mlu = max(adjusted_utils.values()) if adjusted_utils else math.inf
    return float(est_mlu), adjusted_utils, adjusted_txs


def simulate_candidate_edges(
    edge_state: Dict[str, Dict[str, Any]],
    current_edge_ids: List[str],
    candidate_edge_ids: List[str],
    elephant_rate_mbps: float,
    future_delta_pct: float,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    current_set = set(current_edge_ids)
    candidate_set = set(candidate_edge_ids)
    adjusted_utils: Dict[str, float] = {}
    adjusted_txs: Dict[str, float] = {}

    for edge_id, vals in edge_state.items():
        tx = vals.get("tx_mbps", np.nan)
        cap = vals.get("capacity_mbps", np.nan)
        if not np.isfinite(tx) or not np.isfinite(cap) or float(cap) <= 0:
            continue
        adjusted_tx = float(tx)
        if edge_id in current_set and edge_id not in candidate_set:
            adjusted_tx = max(0.0, adjusted_tx - elephant_rate_mbps)
        if edge_id in candidate_set and edge_id not in current_set:
            adjusted_tx = adjusted_tx + elephant_rate_mbps
        adjusted_util = (adjusted_tx / float(cap)) * 100.0
        estimated_future_util = max(0.0, adjusted_util + future_delta_pct)
        adjusted_txs[edge_id] = adjusted_tx
        adjusted_utils[edge_id] = estimated_future_util

    est_mlu = max(adjusted_utils.values()) if adjusted_utils else math.inf
    return float(est_mlu), adjusted_utils, adjusted_txs


def build_decision(
    row: pd.Series,
    model: Any,
    paths_map: Dict[str, Dict[str, Any]],
    state: Dict[str, Any],
    target_col: str,
    threshold_pct: float,
    elephant_rate_mbps: float,
    switch_penalty_pct: float,
    min_improvement_pct: float,
    max_path_util_pct: float,
    force_evaluate: bool,
    peer_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    row_df = pd.DataFrame([row])
    X_row = clean_feature_row(row_df, target_col=target_col)
    pred_future_mlu = float(model.predict(X_row)[0])

    current_mlu = pd.to_numeric(pd.Series([row.get(target_col, np.nan)]), errors="coerce").iloc[0]
    current_mlu = float(current_mlu) if np.isfinite(current_mlu) else pred_future_mlu
    future_delta_pct = pred_future_mlu - current_mlu

    link_state = extract_link_state(row)
    edge_state = extract_edge_state(link_state, peer_map or {}) if peer_map else {}
    current_candidate, current_candidate_basis, current_candidate_metrics = infer_current_candidate(
        row,
        state,
        paths_map,
        link_state=link_state,
        edge_state=edge_state,
    )
    cooldown_active = int(
        pd.to_numeric(
            pd.Series([state.get("cooldown_active", row.get("cooldown_active", 0))]),
            errors="coerce",
        ).fillna(0).iloc[0]
    )
    estimated_elephant_rate_mbps = estimate_elephant_rate(row, elephant_rate_mbps)
    use_edge_scoring = bool(peer_map and edge_state and all(spec.get("edge_ids") for spec in paths_map.values()))

    if current_candidate and current_candidate in paths_map:
        current_links = paths_map[current_candidate]["link_keys"]
        current_edges = paths_map[current_candidate].get("edge_ids", [])
    else:
        current_links = []
        current_edges = []
    if use_edge_scoring and current_edges:
        current_path_hot_vals = [edge_state.get(edge_id, {}).get("util_pct", np.nan) for edge_id in current_edges]
        current_path_hot_vals = [float(v) for v in current_path_hot_vals if np.isfinite(v)]
        current_path_hot_util = max(current_path_hot_vals) if current_path_hot_vals else math.nan
    else:
        current_path_hot_util = current_path_util_pct(current_links, link_state)

    candidate_scores: List[Dict[str, Any]] = []
    for candidate_id, spec in paths_map.items():
        if use_edge_scoring:
            est_mlu, est_utils, adjusted_txs = simulate_candidate_edges(
                edge_state=edge_state,
                current_edge_ids=current_edges,
                candidate_edge_ids=spec.get("edge_ids", []),
                elephant_rate_mbps=estimated_elephant_rate_mbps,
                future_delta_pct=future_delta_pct,
            )
            path_units = spec.get("edge_ids", [])
            load_label = "path_edge_loads"
        else:
            est_mlu, est_utils, adjusted_txs = simulate_candidate(
                link_state=link_state,
                current_path_links=current_links,
                candidate_links=spec["link_keys"],
                elephant_rate_mbps=estimated_elephant_rate_mbps,
                future_delta_pct=future_delta_pct,
            )
            path_units = spec["link_keys"]
            load_label = "path_link_loads"

        path_hottest = max(
            [est_utils.get(unit, -math.inf) for unit in path_units if unit in est_utils],
            default=math.inf,
        )
        score = est_mlu
        if np.isfinite(path_hottest) and path_hottest > max_path_util_pct:
            score += (path_hottest - max_path_util_pct) * 1.5
        if current_candidate and candidate_id != current_candidate:
            score += switch_penalty_pct

        hottest = sorted(est_utils.items(), key=lambda kv: kv[1], reverse=True)[:5]
        bottleneck_key = hottest[0][0] if hottest else ""
        bottleneck_util = hottest[0][1] if hottest else math.nan
        item = {
            "candidate_path_id": candidate_id,
            "score": round(float(score), 3),
            "estimated_future_mlu_pct": round(float(est_mlu), 3),
            "estimated_path_hot_util_pct": round(float(path_hottest), 3) if np.isfinite(path_hottest) else None,
            "current_path_hot_util_pct": round(float(current_path_hot_util), 3) if np.isfinite(current_path_hot_util) else None,
            "bottleneck_link_key": bottleneck_key,
            "bottleneck_estimated_util_pct": round(float(bottleneck_util), 3) if np.isfinite(bottleneck_util) else None,
            "moved_elephant_rate_mbps": round(float(estimated_elephant_rate_mbps), 3),
            "segment_list": spec.get("segment_list", ""),
            "interfaces": spec.get("interfaces", []),
            "top_hot_links": [
                {"link_key": lk, "estimated_future_util_pct": round(float(util), 3)}
                for lk, util in hottest
            ],
            load_label: [
                {
                    ("edge_id" if use_edge_scoring else "link_key"): unit,
                    "estimated_tx_mbps": round(float(adjusted_txs.get(unit, np.nan)), 3)
                    if np.isfinite(adjusted_txs.get(unit, np.nan))
                    else None,
                    "estimated_util_pct": round(float(est_utils.get(unit, np.nan)), 3)
                    if np.isfinite(est_utils.get(unit, np.nan))
                    else None,
                }
                for unit in path_units
            ],
        }
        candidate_scores.append(item)

    candidate_scores = sorted(candidate_scores, key=lambda x: x["score"])
    if not candidate_scores:
        raise ValueError("No candidate paths to score")

    best = candidate_scores[0]
    chosen = best
    reason = "best_score"
    current_score = next((item for item in candidate_scores if item["candidate_path_id"] == current_candidate), None)

    if not force_evaluate:
        if cooldown_active == 1 and current_score is not None:
            chosen = current_score
            reason = "cooldown_active_keep_current"
        elif pred_future_mlu < threshold_pct and current_score is not None:
            chosen = current_score
            reason = "below_threshold_keep_current"
        elif current_score is not None:
            improvement = current_score["estimated_future_mlu_pct"] - best["estimated_future_mlu_pct"]
            if best["candidate_path_id"] != current_candidate and improvement < min_improvement_pct:
                chosen = current_score
                reason = "improvement_too_small_keep_current"
            elif (
                best["candidate_path_id"] != current_candidate
                and best.get("estimated_path_hot_util_pct") is not None
                and float(best["estimated_path_hot_util_pct"]) > max_path_util_pct
            ):
                chosen = current_score
                reason = "best_candidate_path_still_too_hot_keep_current"

    return {
        "timestamp_scored": str(row["timestamp"]),
        "run_key": str(row.get("run_key", "")),
        "experiment_id": str(row.get("experiment_id", "")),
        "mode": str(row.get("mode", "")),
        "active_policy": str(row.get("active_policy", "")),
        "current_candidate_path_id": current_candidate,
        "current_candidate_inference_basis": current_candidate_basis,
        "current_candidate_inference_metrics": current_candidate_metrics,
        "predicted_future_mlu_pct": round(pred_future_mlu, 3),
        "current_mlu_pct": round(current_mlu, 3),
        "future_delta_pct": round(future_delta_pct, 3),
        "estimated_elephant_rate_mbps": round(float(estimated_elephant_rate_mbps), 3),
        "threshold_pct": threshold_pct,
        "max_path_util_pct": max_path_util_pct,
        "cooldown_active": cooldown_active,
        "decision_reason": reason,
        "chosen_candidate_path_id": chosen["candidate_path_id"],
        "chosen_segment_list": chosen.get("segment_list", ""),
        "chosen_estimated_future_mlu_pct": chosen["estimated_future_mlu_pct"],
        "switch_penalty_pct": switch_penalty_pct,
        "min_improvement_pct": min_improvement_pct,
        "elephant_rate_mbps": elephant_rate_mbps,
        "scoring_basis": "physical_edges" if use_edge_scoring else "interface_links",
        "candidate_scores": candidate_scores,
    }


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    df = pd.read_csv(dataset_path, low_memory=False)
    if "timestamp" not in df.columns:
        raise SystemExit("dataset_full.csv is missing timestamp")
    if args.target_col not in df.columns:
        raise SystemExit(f"dataset_full.csv is missing target column {args.target_col!r}")

    model = joblib.load(model_path)
    peer_map = parse_lab_peer_map(args.lab)
    paths_map = load_paths_map(args.paths_json, peer_map=peer_map)
    state = load_json(args.state_file)

    row = choose_row(df, run_key=args.run_key, timestamp=args.timestamp)
    decision = build_decision(
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

    text = json.dumps(decision, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[INFO] Wrote decision to {out_path}")


if __name__ == "__main__":
    main()
