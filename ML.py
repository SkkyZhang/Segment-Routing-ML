#!/usr/bin/env python3
"""Train forecasting models for SR-TE telemetry and save visualizations.

This script is designed for the data layout produced by the collector / traffic
scripts in this conversation:
- telemetry_wide_*.csv
- traffic_events_*.csv
- traffic_flow_intervals_*.csv (optional)
- traffic_manifest_*.json (optional)

What it does:
1. Auto-discovers runs from a data directory.
2. Builds one aligned time-series dataset per run.
3. Creates future targets such as network_mlu_pct at +60 seconds by default.
4. Trains time-aware regression models (default: XGBoost if available).
5. Writes cleaned datasets, metrics, predictions, fitted models, and plots.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBRegressor = None  # type: ignore


TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


@dataclass
class RunBundle:
    run_key: str
    telemetry_path: Path
    traffic_events_path: Optional[Path] = None
    traffic_flow_intervals_path: Optional[Path] = None
    manifest_path: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SR-TE forecasting model with plots.")
    parser.add_argument("--data-dir", default="data", help="Directory with telemetry / traffic CSVs")
    parser.add_argument("--output-dir", default="ml_results", help="Directory for ML outputs")
    parser.add_argument(
        "--target-col",
        default="network_mlu_pct",
        help="Target column in telemetry_wide CSV (default: network_mlu_pct)",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[60],
        help="Forecast horizons in seconds (default: 60)",
    )
    parser.add_argument(
        "--model",
        choices=["auto", "xgboost", "random_forest", "hist_gbm"],
        default="auto",
        help="Model type",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Per-run trailing fraction for test split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Per-run trailing fraction before test split for validation split",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[1, 5, 10, 30, 60],
        help="Lag steps to create for key time-series signals",
    )
    parser.add_argument(
        "--rolling-windows",
        nargs="+",
        type=int,
        default=[5, 10, 30, 60],
        help="Rolling windows in steps for key time-series signals",
    )
    parser.add_argument(
        "--min-run-rows",
        type=int,
        default=120,
        help="Skip runs shorter than this many rows after cleaning",
    )
    parser.add_argument(
        "--no-traffic-merge",
        action="store_true",
        help="Ignore traffic events / flow intervals and train from telemetry only",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap on number of runs to load (0 = all)",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=3000,
        help="Maximum number of test points to draw in scatter/time-series plots",
    )
    return parser.parse_args()


def extract_run_key(path: Path) -> Optional[str]:
    match = TIMESTAMP_RE.search(path.name)
    return match.group(1) if match else None


def discover_runs(data_dir: Path, max_runs: int = 0) -> List[RunBundle]:
    telemetry_paths = sorted(data_dir.glob("telemetry_wide_*.csv"))
    bundles: Dict[str, RunBundle] = {}

    for tp in telemetry_paths:
        run_key = extract_run_key(tp)
        if run_key is None:
            continue
        bundles[run_key] = RunBundle(run_key=run_key, telemetry_path=tp)

    for pattern, attr in [
        ("traffic_events_*.csv", "traffic_events_path"),
        ("traffic_flow_intervals_*.csv", "traffic_flow_intervals_path"),
        ("traffic_manifest_*.json", "manifest_path"),
    ]:
        for p in sorted(data_dir.glob(pattern)):
            key = extract_run_key(p)
            if key in bundles:
                setattr(bundles[key], attr, p)

    runs = [bundles[k] for k in sorted(bundles.keys())]
    if max_runs > 0:
        runs = runs[-max_runs:]
    return runs


def load_manifest(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv_safe(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None


def find_timestamp_col(df: pd.DataFrame, preferred: Sequence[str]) -> Optional[str]:
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        lower = col.lower()
        if "timestamp" in lower or lower.endswith("_ts") or lower.endswith("_time"):
            return col
    return None


def to_datetime_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() > 0:
                non_empty = out[col].astype(str).str.strip().ne("").sum()
                if converted.notna().sum() >= max(1, int(non_empty * 0.5)):
                    out[col] = converted
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    out = df.copy()
    out["ts_second"] = ts.dt.second
    out["ts_minute"] = ts.dt.minute
    out["ts_hour"] = ts.dt.hour
    out["ts_elapsed_seconds"] = (ts - ts.min()).dt.total_seconds()
    out["ts_second_sin"] = np.sin(2 * np.pi * out["ts_second"] / 60.0)
    out["ts_second_cos"] = np.cos(2 * np.pi * out["ts_second"] / 60.0)
    out["ts_minute_sin"] = np.sin(2 * np.pi * out["ts_minute"] / 60.0)
    out["ts_minute_cos"] = np.cos(2 * np.pi * out["ts_minute"] / 60.0)
    return out


def choose_key_signal_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    preferred = [
        target_col,
        "ping_rtt_avg_ms",
        "ping_rtt_p95_ms",
        "ping_rtt_p99_ms",
        "ping_loss_pct",
        "reroute_event",
        "cooldown_active",
        "gain_mlu_pct",
        "gain_rtt_ms",
    ]
    existing = [c for c in preferred if c in df.columns]

    util_cols = [c for c in df.columns if c.endswith("__util_pct")]
    tx_cols = [c for c in df.columns if c.endswith("__tx_mbps")]
    backlog_cols = [c for c in df.columns if c.endswith("__qdisc_backlog_bytes")]
    drop_cols = [c for c in df.columns if c.endswith("__qdisc_drop_delta")]

    existing.extend(util_cols[:8])
    existing.extend(tx_cols[:8])
    existing.extend(backlog_cols[:4])
    existing.extend(drop_cols[:4])

    deduped: List[str] = []
    seen = set()
    for col in existing:
        if col not in seen:
            deduped.append(col)
            seen.add(col)
    return deduped


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    key_cols: Sequence[str],
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> pd.DataFrame:
    out = df.copy()
    for col in key_cols:
        if col not in out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        for lag in lags:
            out[f"{col}__lag_{lag}"] = out[col].shift(lag)
        for win in rolling_windows:
            roll = out[col].rolling(win, min_periods=max(1, win // 2))
            out[f"{col}__roll_mean_{win}"] = roll.mean()
            out[f"{col}__roll_std_{win}"] = roll.std()
            out[f"{col}__roll_min_{win}"] = roll.min()
            out[f"{col}__roll_max_{win}"] = roll.max()
    return out


def estimate_step_seconds(df: pd.DataFrame) -> float:
    if "dt_s" in df.columns:
        dt = pd.to_numeric(df["dt_s"], errors="coerce")
        median_dt = float(dt.dropna().median()) if dt.dropna().size else np.nan
        if np.isfinite(median_dt) and median_dt > 0:
            return median_dt
    delta = df["timestamp"].diff().dt.total_seconds().dropna()
    median_delta = float(delta.median()) if len(delta) else 1.0
    if not np.isfinite(median_delta) or median_delta <= 0:
        median_delta = 1.0
    return median_delta


def add_future_targets(df: pd.DataFrame, target_col: str, horizons: Sequence[int]) -> Tuple[pd.DataFrame, Dict[int, int]]:
    out = df.copy()
    step_seconds = estimate_step_seconds(out)
    horizon_steps: Dict[int, int] = {}
    for horizon in horizons:
        steps = max(1, int(round(horizon / step_seconds)))
        horizon_steps[horizon] = steps
        out[f"target_{target_col}_{horizon}s"] = pd.to_numeric(out[target_col], errors="coerce").shift(-steps)
    return out, horizon_steps


def add_manifest_fields(df: pd.DataFrame, manifest: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if not manifest:
        return out
    for key, value in manifest.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[f"manifest__{key}"] = value
    return out


def aggregate_traffic_events(
    telemetry_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    if traffic_df.empty:
        return pd.DataFrame(index=telemetry_df.index)

    time_index = telemetry_df["timestamp"]
    n = len(time_index)
    if n == 0:
        return pd.DataFrame(index=telemetry_df.index)

    start_col = find_timestamp_col(traffic_df, ["actual_start_ts", "interval_start_ts", "start_ts", "start_time"])
    end_col = find_timestamp_col(traffic_df, ["actual_end_ts", "interval_end_ts", "end_ts", "end_time"])
    if start_col is None or end_col is None:
        return pd.DataFrame(index=telemetry_df.index)

    work = traffic_df.copy()
    work[start_col] = to_datetime_series(work[start_col])
    work[end_col] = to_datetime_series(work[end_col])
    work = work.dropna(subset=[start_col, end_col])
    if work.empty:
        return pd.DataFrame(index=telemetry_df.index)

    for maybe_num in [
        "throughput_mbps",
        "target_bitrate_mbps",
        "parallel",
        "retransmits",
        "lost_percent",
        "jitter_ms",
        "duration_s",
    ]:
        if maybe_num in work.columns:
            work[maybe_num] = pd.to_numeric(work[maybe_num], errors="coerce").fillna(0.0)

    metrics = {
        f"{prefix}__active_count": np.zeros(n + 1, dtype=float),
        f"{prefix}__elephant_count": np.zeros(n + 1, dtype=float),
        f"{prefix}__mice_count": np.zeros(n + 1, dtype=float),
        f"{prefix}__active_parallel_sum": np.zeros(n + 1, dtype=float),
        f"{prefix}__active_target_bitrate_sum": np.zeros(n + 1, dtype=float),
        f"{prefix}__active_throughput_sum": np.zeros(n + 1, dtype=float),
        f"{prefix}__active_retransmits_sum": np.zeros(n + 1, dtype=float),
    }

    time_ns = time_index.view("int64").to_numpy()

    for _, row in work.iterrows():
        start_ns = int(pd.Timestamp(row[start_col]).value)
        end_ns = int(pd.Timestamp(row[end_col]).value)
        left = int(np.searchsorted(time_ns, start_ns, side="left"))
        right = int(np.searchsorted(time_ns, end_ns, side="right"))
        if left >= n:
            continue
        right = min(right, n)
        if right <= left:
            right = min(left + 1, n)

        metrics[f"{prefix}__active_count"][left] += 1
        metrics[f"{prefix}__active_count"][right] -= 1

        flow_type = str(row.get("flow_type", "")).strip().lower()
        if flow_type == "elephant":
            metrics[f"{prefix}__elephant_count"][left] += 1
            metrics[f"{prefix}__elephant_count"][right] -= 1
        elif flow_type == "mice":
            metrics[f"{prefix}__mice_count"][left] += 1
            metrics[f"{prefix}__mice_count"][right] -= 1

        metrics[f"{prefix}__active_parallel_sum"][left] += float(row.get("parallel", 0.0))
        metrics[f"{prefix}__active_parallel_sum"][right] -= float(row.get("parallel", 0.0))
        metrics[f"{prefix}__active_target_bitrate_sum"][left] += float(row.get("target_bitrate_mbps", 0.0))
        metrics[f"{prefix}__active_target_bitrate_sum"][right] -= float(row.get("target_bitrate_mbps", 0.0))
        metrics[f"{prefix}__active_throughput_sum"][left] += float(row.get("throughput_mbps", 0.0))
        metrics[f"{prefix}__active_throughput_sum"][right] -= float(row.get("throughput_mbps", 0.0))
        metrics[f"{prefix}__active_retransmits_sum"][left] += float(row.get("retransmits", 0.0))
        metrics[f"{prefix}__active_retransmits_sum"][right] -= float(row.get("retransmits", 0.0))

    out = pd.DataFrame(index=telemetry_df.index)
    for name, arr in metrics.items():
        out[name] = np.cumsum(arr[:-1])
    return out


def standardize_traffic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for col in out.columns:
        lower = col.lower()
        if lower == "target_bitrate_mbps":
            rename_map[col] = "target_bitrate_mbps"
        elif lower == "bitrate_mbps":
            rename_map[col] = "target_bitrate_mbps"
        elif lower == "retrans":
            rename_map[col] = "retransmits"
        elif lower == "retransmits":
            rename_map[col] = "retransmits"
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def build_run_dataset(
    bundle: RunBundle,
    target_col: str,
    horizons: Sequence[int],
    lags: Sequence[int],
    rolling_windows: Sequence[int],
    merge_traffic: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    telemetry = pd.read_csv(bundle.telemetry_path)
    if "timestamp" not in telemetry.columns:
        raise ValueError(f"telemetry file missing timestamp: {bundle.telemetry_path}")

    telemetry["timestamp"] = to_datetime_series(telemetry["timestamp"])
    telemetry = telemetry.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    telemetry = safe_numeric(telemetry)
    telemetry = telemetry.loc[:, ~telemetry.columns.duplicated()].copy()

    if target_col not in telemetry.columns:
        raise ValueError(f"target column {target_col!r} not found in {bundle.telemetry_path.name}")

    manifest = load_manifest(bundle.manifest_path)
    telemetry = add_manifest_fields(telemetry, manifest)
    telemetry = add_time_features(telemetry)

    if merge_traffic:
        traffic_events = read_csv_safe(bundle.traffic_events_path)
        if traffic_events is not None and not traffic_events.empty:
            traffic_events = standardize_traffic_columns(traffic_events)
            agg_events = aggregate_traffic_events(telemetry, traffic_events, prefix="events")
            telemetry = pd.concat([telemetry, agg_events], axis=1)

        flow_intervals = read_csv_safe(bundle.traffic_flow_intervals_path)
        if flow_intervals is not None and not flow_intervals.empty:
            flow_intervals = standardize_traffic_columns(flow_intervals)
            agg_intervals = aggregate_traffic_events(telemetry, flow_intervals, prefix="intervals")
            telemetry = pd.concat([telemetry, agg_intervals], axis=1)

    key_signal_cols = choose_key_signal_columns(telemetry, target_col=target_col)
    telemetry = add_lag_and_rolling_features(
        telemetry,
        key_cols=key_signal_cols,
        lags=lags,
        rolling_windows=rolling_windows,
    )
    telemetry, horizon_steps = add_future_targets(telemetry, target_col, horizons)

    telemetry["run_key"] = bundle.run_key
    telemetry["source_telemetry_file"] = bundle.telemetry_path.name
    telemetry["source_events_file"] = bundle.traffic_events_path.name if bundle.traffic_events_path else ""
    telemetry["source_intervals_file"] = (
        bundle.traffic_flow_intervals_path.name if bundle.traffic_flow_intervals_path else ""
    )

    meta = {
        "run_key": bundle.run_key,
        "telemetry_path": str(bundle.telemetry_path),
        "traffic_events_path": str(bundle.traffic_events_path) if bundle.traffic_events_path else "",
        "traffic_flow_intervals_path": str(bundle.traffic_flow_intervals_path) if bundle.traffic_flow_intervals_path else "",
        "manifest_path": str(bundle.manifest_path) if bundle.manifest_path else "",
        "rows": int(len(telemetry)),
        "horizon_steps": horizon_steps,
    }
    return telemetry, meta


def split_run_timewise(df: pd.DataFrame, test_ratio: float, val_ratio: float) -> pd.Series:
    n = len(df)
    if n < 3:
        return pd.Series(["train"] * n, index=df.index)

    test_n = max(1, int(round(n * test_ratio)))
    val_n = max(1, int(round(n * val_ratio)))
    if test_n + val_n >= n:
        test_n = max(1, n // 5)
        val_n = max(1, n // 10)
        if test_n + val_n >= n:
            val_n = 1
            test_n = 1

    split = np.array(["train"] * n, dtype=object)
    split[-test_n:] = "test"
    split[-(test_n + val_n):-test_n] = "val"
    return pd.Series(split, index=df.index)


def choose_model(kind: str, random_state: int = 42):
    if kind == "auto":
        kind = "xgboost" if HAS_XGBOOST else "hist_gbm"

    if kind == "xgboost":
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost is not installed; use --model hist_gbm or random_forest")
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=4,
        )
        return kind, model

    if kind == "random_forest":
        return kind, RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=4,
        )

    if kind == "hist_gbm":
        return kind, HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=300,
            random_state=random_state,
        )

    raise ValueError(f"Unknown model type: {kind}")


def build_pipeline(X: pd.DataFrame, model_kind: str):
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model_name, model = choose_model(model_kind)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return model_name, pipeline, numeric_cols, categorical_cols


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clean_feature_matrix(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    drop_cols = {
        target_column,
        "timestamp",
        "wall_time_epoch_ms",
        "sample_id",
    }
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()


def top_feature_importances(pipeline: Pipeline, top_k: int = 20) -> List[Dict[str, Any]]:
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed_feature_names = list(preprocessor.get_feature_names_out())
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_)
        else:
            return []
        order = np.argsort(importances)[::-1][:top_k]
        return [
            {
                "feature": transformed_feature_names[i],
                "importance": float(importances[i]),
            }
            for i in order
        ]
    except Exception:
        return []


def sample_for_plot(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def save_metrics_plot(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    if metrics_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = metrics_df.sort_values("horizon_s")
    ax.plot(plot_df["horizon_s"], plot_df["test_mae"], marker="o", label="test_mae")
    ax.plot(plot_df["horizon_s"], plot_df["test_rmse"], marker="o", label="test_rmse")
    ax.set_xlabel("Horizon (s)")
    ax.set_ylabel("Error")
    ax.set_title("Forecast error by horizon")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_by_horizon.png", dpi=160)
    plt.close(fig)


def save_scatter_plot(preds: pd.DataFrame, horizon: int, output_dir: Path, max_points: int) -> None:
    if preds.empty:
        return
    plot_df = sample_for_plot(preds, max_points)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(plot_df["y_true"], plot_df["y_pred"], s=12, alpha=0.6)
    lo = min(plot_df["y_true"].min(), plot_df["y_pred"].min())
    hi = max(plot_df["y_true"].max(), plot_df["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs predicted ({horizon}s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"pred_scatter_{horizon}s.png", dpi=160)
    plt.close(fig)


def save_timeseries_plot(preds: pd.DataFrame, horizon: int, output_dir: Path, max_points: int) -> None:
    if preds.empty:
        return
    plot_df = preds.sort_values("timestamp").copy()
    plot_df = sample_for_plot(plot_df, max_points)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df["timestamp"], plot_df["y_true"], label="true")
    ax.plot(plot_df["timestamp"], plot_df["y_pred"], label="pred")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Target")
    ax.set_title(f"Test prediction time series ({horizon}s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / f"pred_timeseries_{horizon}s.png", dpi=160)
    plt.close(fig)


def save_residual_hist(preds: pd.DataFrame, horizon: int, output_dir: Path, max_points: int) -> None:
    if preds.empty:
        return
    plot_df = sample_for_plot(preds, max_points)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(plot_df["error"], bins=40)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual distribution ({horizon}s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"pred_residual_hist_{horizon}s.png", dpi=160)
    plt.close(fig)


def save_feature_importance_plot(importance: List[Dict[str, Any]], horizon: int, output_dir: Path) -> None:
    if not importance:
        return
    imp_df = pd.DataFrame(importance).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp_df["feature"], imp_df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top feature importance ({horizon}s)")
    fig.tight_layout()
    fig.savefig(output_dir / f"feature_importance_{horizon}s.png", dpi=160)
    plt.close(fig)


def save_per_run_timeseries(preds: pd.DataFrame, horizon: int, output_dir: Path, max_points: int) -> None:
    if preds.empty or "run_key" not in preds.columns:
        return
    for run_key, group in preds.groupby("run_key"):
        plot_df = group.sort_values("timestamp").copy()
        plot_df = sample_for_plot(plot_df, max_points)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(plot_df["timestamp"], plot_df["y_true"], label="true")
        ax.plot(plot_df["timestamp"], plot_df["y_pred"], label="pred")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Target")
        ax.set_title(f"Run {run_key} prediction ({horizon}s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / f"pred_timeseries_{horizon}s_run_{run_key}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(data_dir, max_runs=args.max_runs)
    if not runs:
        raise SystemExit(f"No telemetry_wide_*.csv found in {data_dir}")

    print(f"[INFO] Found {len(runs)} run(s) in {data_dir}")

    run_frames: List[pd.DataFrame] = []
    run_meta: List[Dict[str, Any]] = []
    for bundle in runs:
        try:
            df_run, meta = build_run_dataset(
                bundle=bundle,
                target_col=args.target_col,
                horizons=args.horizons,
                lags=args.lags,
                rolling_windows=args.rolling_windows,
                merge_traffic=not args.no_traffic_merge,
            )
        except Exception as exc:
            print(f"[WARN] Skipping run {bundle.run_key}: {exc}")
            continue

        if len(df_run) < args.min_run_rows:
            print(f"[WARN] Skipping short run {bundle.run_key}: {len(df_run)} rows")
            continue

        df_run["split"] = split_run_timewise(df_run, args.test_ratio, args.val_ratio)
        run_frames.append(df_run)
        run_meta.append(meta)
        print(f"[INFO] Loaded run {bundle.run_key}: {len(df_run)} rows")

    if not run_frames:
        raise SystemExit("No usable runs after loading / cleaning.")

    full_df = pd.concat(run_frames, ignore_index=True)
    full_df = full_df.sort_values(["run_key", "timestamp"]).reset_index(drop=True)

    dataset_path = output_dir / "dataset_full.csv"
    full_df.to_csv(dataset_path, index=False)
    print(f"[INFO] Wrote dataset to {dataset_path}")

    meta_path = output_dir / "dataset_runs.json"
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote run metadata to {meta_path}")

    metrics_rows: List[Dict[str, Any]] = []
    all_predictions: List[pd.DataFrame] = []

    for horizon in args.horizons:
        target_name = f"target_{args.target_col}_{horizon}s"
        df_h = full_df.dropna(subset=[target_name]).copy()
        if df_h.empty:
            print(f"[WARN] No rows available for horizon {horizon}s")
            continue

        train_df = df_h[df_h["split"] == "train"].copy()
        val_df = df_h[df_h["split"] == "val"].copy()
        test_df = df_h[df_h["split"] == "test"].copy()

        if train_df.empty or test_df.empty:
            print(f"[WARN] Not enough train/test rows for horizon {horizon}s")
            continue

        X_train = clean_feature_matrix(train_df, target_name)
        X_val = clean_feature_matrix(val_df, target_name) if not val_df.empty else None
        X_test = clean_feature_matrix(test_df, target_name)
        y_train = pd.to_numeric(train_df[target_name], errors="coerce").to_numpy()
        y_val = pd.to_numeric(val_df[target_name], errors="coerce").to_numpy() if X_val is not None else None
        y_test = pd.to_numeric(test_df[target_name], errors="coerce").to_numpy()

        model_name, pipeline, numeric_cols, categorical_cols = build_pipeline(X_train, args.model)
        print(
            f"[INFO] Training {model_name} for horizon={horizon}s "
            f"with {len(X_train)} train / {len(X_test)} test rows"
        )
        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        val_pred = pipeline.predict(X_val) if X_val is not None and len(X_val) else np.array([])

        metrics = {
            "model": model_name,
            "horizon_s": horizon,
            "target": args.target_col,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "n_numeric_features": int(len(numeric_cols)),
            "n_categorical_features": int(len(categorical_cols)),
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "train_rmse": rmse(y_train, train_pred),
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "test_rmse": rmse(y_test, test_pred),
            "test_r2": float(r2_score(y_test, test_pred)),
        }
        if y_val is not None and len(y_val):
            metrics["val_mae"] = float(mean_absolute_error(y_val, val_pred))
            metrics["val_rmse"] = rmse(y_val, val_pred)
            metrics["val_r2"] = float(r2_score(y_val, val_pred))

        metrics_rows.append(metrics)

        preds = test_df[["run_key", "timestamp", "experiment_id", "mode", "active_policy"]].copy()
        preds["target_name"] = target_name
        preds["y_true"] = y_test
        preds["y_pred"] = test_pred
        preds["error"] = preds["y_pred"] - preds["y_true"]
        all_predictions.append(preds)

        model_path = output_dir / f"model_{args.target_col}_{horizon}s.joblib"
        joblib.dump(pipeline, model_path)
        print(f"[INFO] Saved model to {model_path}")

        importance = top_feature_importances(pipeline, top_k=25)
        importance_path = output_dir / f"feature_importance_{args.target_col}_{horizon}s.json"
        importance_path.write_text(json.dumps(importance, indent=2), encoding="utf-8")
        save_feature_importance_plot(importance, horizon, output_dir)

        save_scatter_plot(preds, horizon, output_dir, args.max_plot_points)
        save_timeseries_plot(preds, horizon, output_dir, args.max_plot_points)
        save_residual_hist(preds, horizon, output_dir, args.max_plot_points)
        save_per_run_timeseries(preds, horizon, output_dir, args.max_plot_points)

    if not metrics_rows:
        raise SystemExit("Training did not produce any metrics. Check input columns / horizons.")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = output_dir / "metrics.csv"
    metrics_json = output_dir / "metrics.json"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_json.write_text(metrics_df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"[INFO] Saved metrics to {metrics_csv}")
    save_metrics_plot(metrics_df, output_dir)

    if all_predictions:
        pred_df = pd.concat(all_predictions, ignore_index=True)
        pred_path = output_dir / "predictions_test.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"[INFO] Saved test predictions to {pred_path}")

    summary = metrics_df[["horizon_s", "model", "test_mae", "test_rmse", "test_r2"]].sort_values("horizon_s")
    print("\n=== Test metrics summary ===")
    print(summary.to_string(index=False))
    print(f"[INFO] Plots saved under {output_dir}")


if __name__ == "__main__":
    main()