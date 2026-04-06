#!/usr/bin/env python3
"""Build dataset_full.csv from collected telemetry without training a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ML


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset_full.csv only")
    parser.add_argument("--data-dir", default="data", help="Directory with collected CSVs")
    parser.add_argument("--output-dir", default="dataset_only_out", help="Output directory")
    parser.add_argument("--target-col", default="network_mlu_pct")
    parser.add_argument("--horizons", nargs="+", type=int, default=[60])
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 5, 10, 30, 60])
    parser.add_argument("--rolling-windows", nargs="+", type=int, default=[5, 10, 30, 60])
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--min-run-rows", type=int, default=1)
    parser.add_argument("--no-traffic-merge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = ML.discover_runs(data_dir, max_runs=args.max_runs)
    if not runs:
        raise SystemExit(f"No telemetry_wide_*.csv found in {data_dir}")

    run_frames: List[pd.DataFrame] = []
    run_meta: List[Dict[str, Any]] = []
    for bundle in runs:
        df_run, meta = ML.build_run_dataset(
            bundle=bundle,
            target_col=args.target_col,
            horizons=args.horizons,
            lags=args.lags,
            rolling_windows=args.rolling_windows,
            merge_traffic=not args.no_traffic_merge,
        )
        if len(df_run) < args.min_run_rows:
            continue
        df_run["split"] = ML.split_run_timewise(df_run, test_ratio=0.2, val_ratio=0.1)
        run_frames.append(df_run)
        run_meta.append(meta)

    if not run_frames:
        raise SystemExit("No usable runs after building datasets")

    full_df = pd.concat(run_frames, ignore_index=True)
    full_df = full_df.sort_values(["run_key", "timestamp"]).reset_index(drop=True)
    dataset_path = output_dir / "dataset_full.csv"
    full_df.to_csv(dataset_path, index=False)

    meta_path = output_dir / "dataset_runs.json"
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"[INFO] Wrote dataset to {dataset_path}")
    print(f"[INFO] Wrote run metadata to {meta_path}")


if __name__ == "__main__":
    main()
