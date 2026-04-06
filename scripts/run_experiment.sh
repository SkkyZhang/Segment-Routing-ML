#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-}"
RUN_ID="${2:-}"

DOCKER_CMD="${DOCKER_CMD:-sudo docker}"
TOTAL_DURATION="${TOTAL_DURATION:-1800}"
TRAFFIC_PROFILE="${TRAFFIC_PROFILE:-medium}"
FLOW_PROTO="${FLOW_PROTO:-tcp}"
LOG_DIR="${LOG_DIR:-data}"
TOPK_WINDOW_SECS="${TOPK_WINDOW_SECS:-60}"
TOPK_K="${TOPK_K:-5}"
SEED="${SEED:-42}"
COLLECT_INTERVAL="${COLLECT_INTERVAL:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_experiment.sh <igp|static|ml> <run_id>

Examples:
  bash scripts/run_experiment.sh igp run1
  TOTAL_DURATION=600 SEED=101 bash scripts/run_experiment.sh static smoke

Optional environment variables:
  TOTAL_DURATION     Default: 1800
  TRAFFIC_PROFILE    Default: medium
  FLOW_PROTO         Default: tcp
  LOG_DIR            Default: data
  TOPK_WINDOW_SECS   Default: 60
  TOPK_K             Default: 5
  SEED               Default: 42
  COLLECT_INTERVAL   Default: 1
  DOCKER_CMD         Default: sudo docker
EOF
}

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_bin() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

maybe_prime_sudo() {
  local first_word
  first_word="$(printf '%s' "$DOCKER_CMD" | awk '{print $1}')"
  if [[ "$first_word" == "sudo" ]]; then
    log "Refreshing sudo credentials for concurrent docker commands"
    sudo -v || die "Failed to refresh sudo credentials"
  fi
}

mode_to_template() {
  case "$1" in
    igp)
      printf 'controller_state_igp.json'
      ;;
    static)
      printf 'controller_state_static.json'
      ;;
    ml)
      printf 'controller_state_ml.json'
      ;;
    *)
      return 1
      ;;
  esac
}

mode_to_collector_mode() {
  case "$1" in
    igp) printf 'igp' ;;
    static) printf 'static' ;;
    ml) printf 'ml_dynamic' ;;
    *) return 1 ;;
  esac
}

cleanup() {
  if [[ -n "${COLLECTOR_PID:-}" ]] && kill -0 "$COLLECTOR_PID" >/dev/null 2>&1; then
    log "Stopping collector (pid=${COLLECTOR_PID})"
    kill "$COLLECTOR_PID" >/dev/null 2>&1 || true
    wait "$COLLECTOR_PID" >/dev/null 2>&1 || true
  fi
}

main() {
  [[ -n "$MODE" ]] || { usage; exit 1; }
  [[ -n "$RUN_ID" ]] || { usage; exit 1; }

  require_bin bash
  require_bin cp
  require_bin python3
  maybe_prime_sudo

  local template collector_mode experiment_id
  template="$(mode_to_template "$MODE")" || die "Unsupported mode: $MODE"
  collector_mode="$(mode_to_collector_mode "$MODE")" || die "Unsupported mode: $MODE"
  experiment_id="exp_${MODE}_${RUN_ID}"

  [[ -f "$template" ]] || die "Template not found: $template"

  trap cleanup EXIT INT TERM

  log "Using mode=${MODE} run_id=${RUN_ID}"
  log "Copying ${template} -> data/controller_state.json"
  cp "$template" data/controller_state.json

  log "Starting collector for ${TOTAL_DURATION}s"
  python3 collector.py \
    --targets \
      clab-srte8-r1:eth1,eth2 \
      clab-srte8-r2:eth1,eth2 \
      clab-srte8-r3:eth1,eth2,eth3 \
      clab-srte8-r4:eth1,eth2 \
      clab-srte8-r5:eth1,eth2 \
      clab-srte8-r6:eth1,eth2,eth3 \
      clab-srte8-r7:eth1,eth2 \
      clab-srte8-r8:eth1,eth2 \
    --capacities \
      clab-srte8-r1:eth1=1000 clab-srte8-r1:eth2=1000 \
      clab-srte8-r2:eth1=1000 clab-srte8-r2:eth2=1000 \
      clab-srte8-r3:eth1=1000 clab-srte8-r3:eth2=1000 clab-srte8-r3:eth3=1000 \
      clab-srte8-r4:eth1=1000 clab-srte8-r4:eth2=1000 \
      clab-srte8-r5:eth1=1000 clab-srte8-r5:eth2=1000 \
      clab-srte8-r6:eth1=1000 clab-srte8-r6:eth2=1000 clab-srte8-r6:eth3=1000 \
      clab-srte8-r7:eth1=1000 clab-srte8-r7:eth2=1000 \
      clab-srte8-r8:eth1=1000 clab-srte8-r8:eth2=1000 \
    --interval "$COLLECT_INTERVAL" \
    --duration "$TOTAL_DURATION" \
    --outdir "$LOG_DIR" \
    --experiment-id "$experiment_id" \
    --mode "$collector_mode" \
    --traffic-profile "$TRAFFIC_PROFILE" \
    --state-file data/controller_state.json \
    --probes \
      clab-srte8-h1:192.168.8.2:h1_to_h2 \
      clab-srte8-h2:192.168.1.2:h2_to_h1 \
    &
  COLLECTOR_PID=$!

  sleep 2

  log "Starting traffic generator"
  DOCKER_CMD="$DOCKER_CMD" \
  TOTAL_DURATION="$TOTAL_DURATION" \
  TRAFFIC_PROFILE="$TRAFFIC_PROFILE" \
  FLOW_PROTO="$FLOW_PROTO" \
  LOG_DIR="$LOG_DIR" \
  TOPK_WINDOW_SECS="$TOPK_WINDOW_SECS" \
  TOPK_K="$TOPK_K" \
  SEED="$SEED" \
  STATE_FILE="data/controller_state.json" \
  bash traffic_gen_v2.sh

  log "Waiting for collector to finish"
  wait "$COLLECTOR_PID"
  COLLECTOR_PID=""

  log "Experiment completed: ${experiment_id}"
  log "Data written under ${LOG_DIR}/"
}

main "$@"
