#!/usr/bin/env python3
"""Build per-window Top-K elephant-flow datasets from traffic interval logs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Top-K elephant flows per decision window.")
    parser.add_argument("--flow-intervals-csv", required=True, help="traffic_flow_intervals_*.csv path")
    parser.add_argument("--output-csv", required=True, help="Output CSV path for Top-K elephant rows")
    parser.add_argument(
        "--window-summary-csv",
        default="",
        help="Optional output CSV path for per-window summary rows",
    )
    parser.add_argument(
        "--window-secs",
        type=int,
        default=60,
        help="Decision window size in seconds (default: 60)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many elephant flows to keep per window (default: 5)",
    )
    return parser.parse_args()


def extract_run_key(path: Path) -> str:
    match = TIMESTAMP_RE.search(path.name)
    return match.group(1) if match else ""


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def epoch_ms_to_iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def init_flow_acc(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "event_id": row.get("event_id", ""),
        "flow_type": row.get("flow_type", ""),
        "traffic_profile": row.get("traffic_profile", ""),
        "flow_proto": row.get("flow_proto", ""),
        "seed": row.get("seed", ""),
        "port": row.get("port", ""),
        "duration_s": row.get("duration_s", ""),
        "parallel": row.get("parallel", ""),
        "target_bitrate_mbps": row.get("target_bitrate_mbps", ""),
        "start_active_policy": row.get("start_active_policy", ""),
        "start_policy_id": row.get("start_policy_id", ""),
        "start_candidate_path_id": row.get("start_candidate_path_id", ""),
        "start_elephant_path_hint": row.get("start_elephant_path_hint", ""),
        "interval_count": 0,
        "active_seconds": 0.0,
        "transferred_mbits": 0.0,
        "throughput_peak_mbps": 0.0,
        "retransmits_sum": 0.0,
        "loss_weighted_sum": 0.0,
        "jitter_weighted_sum": 0.0,
    }


def update_flow_acc(acc: Dict[str, Any], row: Dict[str, str]) -> None:
    seconds = max(to_float(row.get("interval_seconds"), 0.0), 0.0)
    throughput = max(to_float(row.get("throughput_mbps"), 0.0), 0.0)
    retransmits = max(to_float(row.get("retransmits"), 0.0), 0.0)
    lost_percent = to_float(row.get("lost_percent"), 0.0)
    jitter_ms = to_float(row.get("jitter_ms"), 0.0)

    acc["interval_count"] += 1
    acc["active_seconds"] += seconds
    acc["transferred_mbits"] += throughput * seconds
    acc["throughput_peak_mbps"] = max(acc["throughput_peak_mbps"], throughput)
    acc["retransmits_sum"] += retransmits
    acc["loss_weighted_sum"] += lost_percent * seconds
    acc["jitter_weighted_sum"] += jitter_ms * seconds


def build_window_rows(
    rows: Iterable[Dict[str, str]],
    run_key: str,
    window_secs: int,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    work = [row for row in rows if str(row.get("flow_type", "")).strip().lower() == "elephant"]
    if not work:
        return [], []

    min_epoch_ms = min(to_int(row.get("interval_start_epoch_ms")) for row in work)
    window_ms = max(window_secs, 1) * 1000

    by_window: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    meta_by_window: Dict[int, Dict[str, Any]] = {}

    for row in work:
        start_epoch_ms = to_int(row.get("interval_start_epoch_ms"))
        window_index = max(0, (start_epoch_ms - min_epoch_ms) // window_ms)
        event_id = row.get("event_id", "")

        if event_id not in by_window[window_index]:
            by_window[window_index][event_id] = init_flow_acc(row)

        update_flow_acc(by_window[window_index][event_id], row)

        if window_index not in meta_by_window:
            window_start_epoch_ms = min_epoch_ms + window_index * window_ms
            meta_by_window[window_index] = {
                "run_key": run_key,
                "window_index": window_index,
                "window_start_epoch_ms": window_start_epoch_ms,
                "window_end_epoch_ms": window_start_epoch_ms + window_ms,
                "window_start_ts": epoch_ms_to_iso(window_start_epoch_ms),
                "window_end_ts": epoch_ms_to_iso(window_start_epoch_ms + window_ms),
                "decision_window_s": window_secs,
            }

    topk_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for window_index in sorted(by_window.keys()):
        meta = meta_by_window[window_index]
        candidates = list(by_window[window_index].values())
        candidates.sort(key=lambda x: (x["transferred_mbits"], x["throughput_peak_mbps"]), reverse=True)

        total_elephant_mbits = sum(x["transferred_mbits"] for x in candidates)
        total_retransmits = sum(x["retransmits_sum"] for x in candidates)
        topk_total_mbits = sum(x["transferred_mbits"] for x in candidates[:top_k])
        top1_mbits = candidates[0]["transferred_mbits"] if candidates else 0.0

        summary_rows.append(
            {
                **meta,
                "elephant_flow_count": len(candidates),
                "elephant_active_flow_count": sum(1 for x in candidates if x["active_seconds"] > 0),
                "total_elephant_mbits": round(total_elephant_mbits, 3),
                "top1_elephant_mbits": round(top1_mbits, 3),
                "topk_elephant_mbits": round(topk_total_mbits, 3),
                "top1_share_pct": round((top1_mbits / total_elephant_mbits) * 100.0, 3) if total_elephant_mbits > 0 else 0.0,
                "topk_share_pct": round((topk_total_mbits / total_elephant_mbits) * 100.0, 3) if total_elephant_mbits > 0 else 0.0,
                "total_retransmits": round(total_retransmits, 3),
            }
        )

        cumulative_mbits = 0.0
        for rank, item in enumerate(candidates[:top_k], start=1):
            active_seconds = item["active_seconds"]
            transferred_mbits = item["transferred_mbits"]
            cumulative_mbits += transferred_mbits
            throughput_mean = transferred_mbits / active_seconds if active_seconds > 0 else 0.0
            loss_mean = item["loss_weighted_sum"] / active_seconds if active_seconds > 0 else 0.0
            jitter_mean = item["jitter_weighted_sum"] / active_seconds if active_seconds > 0 else 0.0

            topk_rows.append(
                {
                    **meta,
                    "event_id": item["event_id"],
                    "rank": rank,
                    "flow_type": item["flow_type"],
                    "traffic_profile": item["traffic_profile"],
                    "flow_proto": item["flow_proto"],
                    "seed": item["seed"],
                    "port": item["port"],
                    "duration_s": item["duration_s"],
                    "parallel": item["parallel"],
                    "target_bitrate_mbps": item["target_bitrate_mbps"],
                    "start_active_policy": item["start_active_policy"],
                    "start_policy_id": item["start_policy_id"],
                    "start_candidate_path_id": item["start_candidate_path_id"],
                    "start_elephant_path_hint": item["start_elephant_path_hint"],
                    "interval_count": item["interval_count"],
                    "active_seconds": round(active_seconds, 3),
                    "transferred_mbits": round(transferred_mbits, 3),
                    "throughput_mbps_mean": round(throughput_mean, 3),
                    "throughput_mbps_peak": round(item["throughput_peak_mbps"], 3),
                    "retransmits_sum": round(item["retransmits_sum"], 3),
                    "loss_pct_mean": round(loss_mean, 3),
                    "jitter_ms_mean": round(jitter_mean, 3),
                    "flow_share_pct": round((transferred_mbits / total_elephant_mbits) * 100.0, 3) if total_elephant_mbits > 0 else 0.0,
                    "cumulative_share_pct": round((cumulative_mbits / total_elephant_mbits) * 100.0, 3) if total_elephant_mbits > 0 else 0.0,
                    "window_elephant_flow_count": len(candidates),
                    "window_total_elephant_mbits": round(total_elephant_mbits, 3),
                    "window_topk_elephant_mbits": round(topk_total_mbits, 3),
                    "window_topk_share_pct": round((topk_total_mbits / total_elephant_mbits) * 100.0, 3) if total_elephant_mbits > 0 else 0.0,
                }
            )

    return topk_rows, summary_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    flow_intervals_path = Path(args.flow_intervals_csv)
    if not flow_intervals_path.exists():
        raise SystemExit(f"Input CSV not found: {flow_intervals_path}")
    if args.window_secs <= 0:
        raise SystemExit("--window-secs must be > 0")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")

    rows = read_rows(flow_intervals_path)
    run_key = extract_run_key(flow_intervals_path)
    topk_rows, summary_rows = build_window_rows(
        rows=rows,
        run_key=run_key,
        window_secs=args.window_secs,
        top_k=args.top_k,
    )

    output_csv = Path(args.output_csv)
    write_csv(output_csv, topk_rows)
    print(f"[INFO] Wrote {len(topk_rows)} Top-K elephant rows to {output_csv}")

    if args.window_summary_csv:
        summary_csv = Path(args.window_summary_csv)
        write_csv(summary_csv, summary_rows)
        print(f"[INFO] Wrote {len(summary_rows)} window summary rows to {summary_csv}")


if __name__ == "__main__":
    main()
