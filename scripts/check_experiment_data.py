#!/usr/bin/env python3
"""Check whether collected experiment runs have complete artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


TS_RE = re.compile(r"(\d{8}_\d{6})")

REQUIRED_FILE_TYPES = [
    "telemetry_wide",
    "telemetry_long",
    "probe_rtt",
    "control_overhead",
    "traffic_events",
    "traffic_flow_intervals",
    "traffic_manifest",
    "topk_elephants",
    "topk_elephant_windows",
]

OPTIONAL_DIR_TYPES = [
    "iperf_json_dir",
    "state_snapshots_dir",
]


@dataclass
class RunCheck:
    anchor_key: str
    anchor_dt: datetime
    telemetry_wide: Path
    mode: str = "unknown"
    experiment_id: str = ""
    active_policy: str = ""
    files: Dict[str, Optional[Path]] = field(default_factory=dict)
    directories: Dict[str, Optional[Path]] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check completeness of IGP/static/ML experiment data.")
    parser.add_argument("--data-dir", default="data", help="Directory with collected CSV/JSON artifacts")
    parser.add_argument(
        "--expected-modes",
        nargs="+",
        default=["igp", "static", "ml_dynamic"],
        help="Expected run modes to find",
    )
    parser.add_argument(
        "--max-skew-seconds",
        type=int,
        default=10,
        help="Maximum timestamp skew allowed when matching traffic artifacts to telemetry artifacts",
    )
    return parser.parse_args()


def extract_ts(path: Path) -> Optional[str]:
    match = TS_RE.search(path.name)
    return match.group(1) if match else None


def parse_ts(ts_text: str) -> datetime:
    return datetime.strptime(ts_text, "%Y%m%d_%H%M%S")


def read_first_csv_row(path: Path) -> Dict[str, str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                return {k: (v or "") for k, v in row.items()}
    except Exception:
        return {}
    return {}


def count_csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return -1


def load_manifest(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_candidates(data_dir: Path) -> Dict[str, List[Tuple[datetime, Path]]]:
    mapping = {
        "telemetry_wide": sorted(data_dir.glob("telemetry_wide_*.csv")),
        "telemetry_long": sorted(data_dir.glob("telemetry_long_*.csv")),
        "probe_rtt": sorted(data_dir.glob("probe_rtt_*.csv")),
        "control_overhead": sorted(data_dir.glob("control_overhead_*.csv")),
        "traffic_events": sorted(data_dir.glob("traffic_events_*.csv")),
        "traffic_flow_intervals": sorted(data_dir.glob("traffic_flow_intervals_*.csv")),
        "traffic_manifest": sorted(data_dir.glob("traffic_manifest_*.json")),
        "topk_elephants": sorted(data_dir.glob("topk_elephants_*.csv")),
        "topk_elephant_windows": sorted(data_dir.glob("topk_elephant_windows_*.csv")),
        "iperf_json_dir": sorted([p for p in data_dir.glob("iperf_json_*") if p.is_dir()]),
        "state_snapshots_dir": sorted([p for p in data_dir.glob("state_snapshots_*") if p.is_dir()]),
    }
    out: Dict[str, List[Tuple[datetime, Path]]] = {}
    for key, paths in mapping.items():
        items: List[Tuple[datetime, Path]] = []
        for path in paths:
            ts_text = extract_ts(path)
            if ts_text is None:
                continue
            items.append((parse_ts(ts_text), path))
        out[key] = items
    return out


def pick_nearest(
    anchor_dt: datetime,
    candidates: Sequence[Tuple[datetime, Path]],
    max_skew_seconds: int,
    used_paths: set[Path],
) -> Optional[Path]:
    best: Optional[Tuple[float, Path]] = None
    for cand_dt, cand_path in candidates:
        if cand_path in used_paths:
            continue
        skew = abs((cand_dt - anchor_dt).total_seconds())
        if skew > max_skew_seconds:
            continue
        if best is None or skew < best[0]:
            best = (skew, cand_path)
    if best is None:
        return None
    used_paths.add(best[1])
    return best[1]


def build_run_checks(data_dir: Path, max_skew_seconds: int) -> List[RunCheck]:
    candidates = list_candidates(data_dir)
    telemetry_items = candidates["telemetry_wide"]
    used: Dict[str, set[Path]] = {key: set() for key in candidates.keys()}
    runs: List[RunCheck] = []

    for anchor_dt, telemetry_path in telemetry_items:
        anchor_key = extract_ts(telemetry_path)
        if anchor_key is None:
            continue
        first_row = read_first_csv_row(telemetry_path)
        run = RunCheck(
            anchor_key=anchor_key,
            anchor_dt=anchor_dt,
            telemetry_wide=telemetry_path,
            mode=first_row.get("mode", "") or "unknown",
            experiment_id=first_row.get("experiment_id", "") or "",
            active_policy=first_row.get("active_policy", "") or "",
            files={"telemetry_wide": telemetry_path},
        )
        used["telemetry_wide"].add(telemetry_path)

        for file_type in REQUIRED_FILE_TYPES:
            if file_type == "telemetry_wide":
                continue
            matched = pick_nearest(anchor_dt, candidates[file_type], max_skew_seconds, used[file_type])
            run.files[file_type] = matched

        for dir_type in OPTIONAL_DIR_TYPES:
            matched = pick_nearest(anchor_dt, candidates[dir_type], max_skew_seconds, used[dir_type])
            run.directories[dir_type] = matched

        runs.append(run)
    return runs


def validate_run(run: RunCheck) -> None:
    for file_type in REQUIRED_FILE_TYPES:
        path = run.files.get(file_type)
        if path is None:
            run.problems.append(f"missing {file_type}")
            continue
        if not path.exists():
            run.problems.append(f"{file_type} path missing on disk")
            continue
        if path.suffix.lower() == ".csv":
            row_count = count_csv_rows(path)
            if row_count <= 0:
                run.problems.append(f"{file_type} is empty")
        elif path.suffix.lower() == ".json":
            manifest = load_manifest(path)
            if not manifest:
                run.problems.append(f"{file_type} is empty or unreadable")

    for dir_type in OPTIONAL_DIR_TYPES:
        path = run.directories.get(dir_type)
        if path is None:
            run.notes.append(f"missing {dir_type}")
            continue
        entries = list(path.iterdir())
        if not entries:
            run.problems.append(f"{dir_type} directory is empty")

    control_path = run.files.get("control_overhead")
    if control_path is not None:
        first_control = read_first_csv_row(control_path)
        control_mode = first_control.get("mode", "") or "unknown"
        if control_mode != run.mode:
            run.problems.append(
                f"mode mismatch: telemetry_wide={run.mode} control_overhead={control_mode}"
            )

    manifest_path = run.files.get("traffic_manifest")
    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
        manifest_mode = str(manifest.get("active_policy", "")).strip()
        if manifest_mode and run.active_policy and manifest_mode != run.active_policy:
            run.notes.append(
                f"manifest active_policy={manifest_mode} telemetry active_policy={run.active_policy}"
            )


def summarize_modes(runs: Sequence[RunCheck]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for run in runs:
        counts[run.mode] = counts.get(run.mode, 0) + 1
    return counts


def print_run_summary(run: RunCheck) -> None:
    status = "PASS" if not run.problems else "FAIL"
    print(
        f"[{status}] run={run.anchor_key} mode={run.mode} "
        f"experiment_id={run.experiment_id or '-'} policy={run.active_policy or '-'}"
    )
    for file_type in REQUIRED_FILE_TYPES:
        path = run.files.get(file_type)
        if path is None:
            print(f"  - missing: {file_type}")
            continue
        if path.suffix.lower() == ".csv":
            row_count = count_csv_rows(path)
            print(f"  - {file_type}: {path.name} rows={row_count}")
        else:
            print(f"  - {file_type}: {path.name}")
    for dir_type in OPTIONAL_DIR_TYPES:
        path = run.directories.get(dir_type)
        if path is None:
            print(f"  - optional missing: {dir_type}")
        else:
            entry_count = len(list(path.iterdir()))
            print(f"  - {dir_type}: {path.name} entries={entry_count}")
    for problem in run.problems:
        print(f"  - problem: {problem}")
    for note in run.notes:
        print(f"  - note: {note}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    runs = build_run_checks(data_dir, args.max_skew_seconds)
    if not runs:
        raise SystemExit(f"no telemetry_wide_*.csv found in {data_dir}")

    for run in runs:
        validate_run(run)

    print(f"[INFO] Found {len(runs)} collected run(s) in {data_dir}")
    print()
    for run in runs:
        print_run_summary(run)
        print()

    mode_counts = summarize_modes(runs)
    missing_modes = [mode for mode in args.expected_modes if mode not in mode_counts]

    complete_runs = sum(1 for run in runs if not run.problems)
    failed_runs = len(runs) - complete_runs
    print("[SUMMARY]")
    print(f"complete_runs={complete_runs}")
    print(f"failed_runs={failed_runs}")
    print(f"modes_found={mode_counts}")
    if missing_modes:
        print(f"missing_expected_modes={missing_modes}")
        raise SystemExit(1)
    if failed_runs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
