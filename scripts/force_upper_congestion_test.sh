#!/usr/bin/env bash
set -Eeuo pipefail

DOCKER_CMD="${DOCKER_CMD:-sudo docker}"
read -r -a DOCKER <<< "$DOCKER_CMD"

OUTDIR="${OUTDIR:-/tmp/upper_congestion_test}"
RAW_DIR="${OUTDIR}/raw"
DATASET_DIR="${OUTDIR}/dataset"
DECISION_JSON="${OUTDIR}/decision_upper_congest.json"
TOPOLOGY_PNG="${OUTDIR}/topology_upper_congest.png"
PEAK_INFO_JSON="${OUTDIR}/peak_snapshot.json"
MODE_NAME="${MODE_NAME:-ml_dynamic}"
STATE_FILE="${STATE_FILE:-data/controller_state.json}"
STATE_TEMPLATE="${STATE_TEMPLATE:-}"
EXPERIMENT_ID="${EXPERIMENT_ID:-upper_congestion_test_${MODE_NAME}}"
COLLECTOR_MODE="${COLLECTOR_MODE:-${MODE_NAME}}"
STATE_BACKUP=""

H1_CONTAINER="${H1_CONTAINER:-clab-srte8-h1}"
H2_CONTAINER="${H2_CONTAINER:-clab-srte8-h2}"
DST_IP="${DST_IP:-192.168.8.2}"

MODEL_PATH="${MODEL_PATH:-/tmp/ml_verify_60s/model_network_mlu_pct_60s.joblib}"
DURATION="${DURATION:-120}"
INTERVAL="${INTERVAL:-1}"
SHAPE_RATE_MBIT="${SHAPE_RATE_MBIT:-120}"
CLIENT_BITRATE_MBIT="${CLIENT_BITRATE_MBIT:-300}"
CLIENT_PARALLEL="${CLIENT_PARALLEL:-3}"
SERVER_PORT="${SERVER_PORT:-5201}"

DEFAULT_SHAPE_IFACES=(
  "clab-srte8-r1:eth1"
  "clab-srte8-r2:eth2"
  "clab-srte8-r3:eth2"
  "clab-srte8-r4:eth2"
)
SHAPE_IFACES=("${DEFAULT_SHAPE_IFACES[@]}")
if [[ -n "${SHAPE_IFACES_CSV:-}" ]]; then
  IFS=',' read -r -a SHAPE_IFACES <<< "${SHAPE_IFACES_CSV}"
fi

TARGET_ARGS=(
  clab-srte8-r1:eth1,eth2
  clab-srte8-r2:eth1,eth2
  clab-srte8-r3:eth1,eth2,eth3
  clab-srte8-r4:eth1,eth2
  clab-srte8-r5:eth1,eth2
  clab-srte8-r6:eth1,eth2,eth3
  clab-srte8-r7:eth1,eth2
  clab-srte8-r8:eth1,eth2
)

CAPACITY_ARGS=()

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_container() {
  local name="$1"
  "${DOCKER[@]}" inspect "$name" >/dev/null 2>&1 || die "missing container: $name"
}

ensure_iperf3() {
  local container="$1"
  if "${DOCKER[@]}" exec "$container" sh -lc 'command -v iperf3 >/dev/null 2>&1'; then
    return 0
  fi
  die "container $container missing iperf3"
}

apply_shape_limit() {
  for spec in "${SHAPE_IFACES[@]}"; do
    local container="${spec%%:*}"
    local iface="${spec##*:}"
    log "Applying tbf ${SHAPE_RATE_MBIT}mbit on ${container}:${iface}"
    "${DOCKER[@]}" exec "$container" sh -lc \
      "tc qdisc replace dev ${iface} root tbf rate ${SHAPE_RATE_MBIT}mbit burst 32k latency 100ms"
  done
}

clear_shape_limit() {
  for spec in "${SHAPE_IFACES[@]}"; do
    local container="${spec%%:*}"
    local iface="${spec##*:}"
    "${DOCKER[@]}" exec "$container" sh -lc \
      "tc qdisc del dev ${iface} root >/dev/null 2>&1 || true" || true
  done
}

build_capacity_args() {
  local shaped_map=" ${SHAPE_IFACES[*]} "
  local specs=(
    clab-srte8-r1:eth1 clab-srte8-r1:eth2
    clab-srte8-r2:eth1 clab-srte8-r2:eth2
    clab-srte8-r3:eth1 clab-srte8-r3:eth2 clab-srte8-r3:eth3
    clab-srte8-r4:eth1 clab-srte8-r4:eth2
    clab-srte8-r5:eth1 clab-srte8-r5:eth2
    clab-srte8-r6:eth1 clab-srte8-r6:eth2 clab-srte8-r6:eth3
    clab-srte8-r7:eth1 clab-srte8-r7:eth2
    clab-srte8-r8:eth1 clab-srte8-r8:eth2
  )
  CAPACITY_ARGS=()
  for spec in "${specs[@]}"; do
    local cap="1000"
    if [[ "$shaped_map" == *" ${spec} "* ]]; then
      cap="${SHAPE_RATE_MBIT}"
    fi
    CAPACITY_ARGS+=("${spec}=${cap}")
  done
}

cleanup() {
  set +e
  clear_shape_limit
  "${DOCKER[@]}" exec "$H2_CONTAINER" sh -lc "pkill -x iperf3 >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
  if [[ -n "${STATE_BACKUP}" && -f "${STATE_BACKUP}" ]]; then
    cp "${STATE_BACKUP}" "${STATE_FILE}" >/dev/null 2>&1 || true
    rm -f "${STATE_BACKUP}" >/dev/null 2>&1 || true
  fi
}

choose_state_template() {
  if [[ -n "${STATE_TEMPLATE}" ]]; then
    return 0
  fi
  case "${MODE_NAME}" in
    igp)
      STATE_TEMPLATE="controller_state_igp.json"
      ;;
    static)
      STATE_TEMPLATE="controller_state_static.json"
      ;;
    ml|ml_dynamic)
      MODE_NAME="ml_dynamic"
      STATE_TEMPLATE="controller_state_ml.json"
      ;;
    *)
      die "unsupported MODE_NAME=${MODE_NAME}; use igp, static, or ml_dynamic"
      ;;
  esac
}

prepare_state_file() {
  choose_state_template
  [[ -f "${STATE_TEMPLATE}" ]] || die "missing state template: ${STATE_TEMPLATE}"
  [[ -f "${STATE_FILE}" ]] || die "missing state file: ${STATE_FILE}"
  STATE_BACKUP="$(mktemp /tmp/controller_state_backup.XXXXXX.json)"
  cp "${STATE_FILE}" "${STATE_BACKUP}"
  cp "${STATE_TEMPLATE}" "${STATE_FILE}"
}

main() {
  mkdir -p "$RAW_DIR" "$DATASET_DIR"

  if [[ "${DOCKER[0]}" == "sudo" ]]; then
    sudo -v
  fi

  require_container "$H1_CONTAINER"
  require_container "$H2_CONTAINER"
  ensure_iperf3 "$H1_CONTAINER"
  ensure_iperf3 "$H2_CONTAINER"
  build_capacity_args
  prepare_state_file

  trap cleanup EXIT

  log "Cleaning old iperf3 server"
  "${DOCKER[@]}" exec "$H2_CONTAINER" sh -lc "pkill -x iperf3 >/dev/null 2>&1 || true"
  log "Starting iperf3 server on ${H2_CONTAINER}:${SERVER_PORT}"
  "${DOCKER[@]}" exec "$H2_CONTAINER" sh -lc "nohup iperf3 -s -p ${SERVER_PORT} >/tmp/iperf3_${SERVER_PORT}.log 2>&1 &"
  sleep 1

  apply_shape_limit

  log "Starting collector into ${RAW_DIR}"
  python3 collector.py \
    --targets "${TARGET_ARGS[@]}" \
    --capacities "${CAPACITY_ARGS[@]}" \
    --interval "$INTERVAL" \
    --duration "$DURATION" \
    --outdir "$RAW_DIR" \
    --experiment-id "$EXPERIMENT_ID" \
    --mode "$COLLECTOR_MODE" \
    --traffic-profile high \
    --state-file "$STATE_FILE" \
    --probes \
      clab-srte8-h1:192.168.8.2:h1_to_h2 \
      clab-srte8-h2:192.168.1.2:h2_to_h1 \
    >/tmp/upper_congestion_collector.log 2>&1 &
  COLLECTOR_PID=$!

  sleep 3
  log "Sending stressed upper-path traffic"
  "${DOCKER[@]}" exec "$H1_CONTAINER" sh -lc \
    "iperf3 -c ${DST_IP} -p ${SERVER_PORT} -t $((DURATION - 10)) -P ${CLIENT_PARALLEL} --bitrate ${CLIENT_BITRATE_MBIT}M -i 1 -J >/tmp/upper_congest_client.json"

  wait "$COLLECTOR_PID"
  log "Collector finished"

  log "Building dataset_only output"
  python3 scripts/build_dataset_only.py \
    --data-dir "$RAW_DIR" \
    --output-dir "$DATASET_DIR" \
    --target-col network_mlu_pct \
    --horizons 60 \
    --min-run-rows 1

  [[ -f "$MODEL_PATH" ]] || die "missing model: $MODEL_PATH"

  local peak_ts
  local latest_run_key
  latest_run_key="$(DATASET_PATH="${DATASET_DIR}/dataset_full.csv" python3 - <<'PY'
import os
import pandas as pd

df = pd.read_csv(os.environ["DATASET_PATH"], low_memory=False)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp", "network_mlu_pct"]).copy()
if df.empty:
    raise SystemExit("no rows with timestamp/network_mlu_pct")
latest_run = (
    df.groupby("run_key", as_index=False)["timestamp"]
    .max()
    .sort_values("timestamp")
    .iloc[-1]
)
print(str(latest_run["run_key"]))
PY
)"
  [[ -n "$latest_run_key" ]] || die "failed to locate latest run key"
  peak_ts="$(DATASET_PATH="${DATASET_DIR}/dataset_full.csv" LATEST_RUN_KEY="${latest_run_key}" python3 - <<'PY'
import os
import pandas as pd

df = pd.read_csv(os.environ["DATASET_PATH"], low_memory=False)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp", "network_mlu_pct"]).copy()
if df.empty:
    raise SystemExit("no rows with timestamp/network_mlu_pct")
sub = df[df["run_key"].astype(str) == os.environ["LATEST_RUN_KEY"]].copy()
if sub.empty:
    raise SystemExit("latest run key has no rows")
peak = sub.loc[sub["network_mlu_pct"].idxmax()]
print(str(peak["timestamp"]))
PY
)"
  [[ -n "$peak_ts" ]] || die "failed to locate peak timestamp"
  log "Latest run key: ${latest_run_key}"
  log "Peak congestion timestamp: ${peak_ts}"
  DATASET_PATH="${DATASET_DIR}/dataset_full.csv" LATEST_RUN_KEY="${latest_run_key}" PEAK_INFO_PATH="${PEAK_INFO_JSON}" python3 - <<'PY'
import json
import os
import pandas as pd
df = pd.read_csv(os.environ["DATASET_PATH"], low_memory=False)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
sub = df.dropna(subset=["timestamp", "network_mlu_pct"]).copy()
sub = sub[sub["run_key"].astype(str) == os.environ["LATEST_RUN_KEY"]].copy()
peak = sub.loc[sub["network_mlu_pct"].idxmax()]
payload = {
    "timestamp": str(peak["timestamp"]),
    "network_mlu_pct": float(peak["network_mlu_pct"]),
    "mode": str(peak.get("mode", "")),
    "run_key": str(peak.get("run_key", "")),
}
with open(os.environ["PEAK_INFO_PATH"], "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
print(json.dumps(payload, indent=2))
PY

  log "Running decider on peak-congestion snapshot"
  python3 srte_decider.py \
    --dataset "${DATASET_DIR}/dataset_full.csv" \
    --model "$MODEL_PATH" \
    --paths-json candidate_paths_example.json \
    --state-file "$STATE_FILE" \
    --lab lab.clab.yml \
    --run-key "$latest_run_key" \
    --timestamp "$peak_ts" \
    --force-evaluate \
    --out-json "$DECISION_JSON"

  log "Drawing highlighted topology"
  python3 scripts/plot_topology.py \
    --lab lab.clab.yml \
    --paths-json candidate_paths_example.json \
    --decision-json "$DECISION_JSON" \
    --output "$TOPOLOGY_PNG"

  log "Artifacts:"
  log "  raw telemetry: ${RAW_DIR}"
  log "  dataset: ${DATASET_DIR}/dataset_full.csv"
  log "  peak: ${PEAK_INFO_JSON}"
  log "  decision: ${DECISION_JSON}"
  log "  topology: ${TOPOLOGY_PNG}"
}

main "$@"
