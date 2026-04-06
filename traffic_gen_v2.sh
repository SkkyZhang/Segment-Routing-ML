#!/usr/bin/env bash
set -Eeuo pipefail

DOCKER_CMD="${DOCKER_CMD:-sudo docker}"
read -r -a DOCKER <<< "$DOCKER_CMD"

H1_CONTAINER="${H1_CONTAINER:-clab-srte4-h1}"
H2_CONTAINER="${H2_CONTAINER:-clab-srte4-h2}"
DST_IP="${DST_IP:-192.168.4.2}"

TOTAL_DURATION="${TOTAL_DURATION:-1800}"
LOG_DIR="${LOG_DIR:-data}"
SEED="${SEED:-42}"
TRAFFIC_PROFILE="${TRAFFIC_PROFILE:-medium}"
FLOW_PROTO="${FLOW_PROTO:-tcp}"

STATE_FILE="${STATE_FILE:-}"
STATE_READ_RETRIES="${STATE_READ_RETRIES:-2}"
STATE_READ_RETRY_SLEEP_S="${STATE_READ_RETRY_SLEEP_S:-0.03}"

ELEPHANT_PORT="${ELEPHANT_PORT:-5201}"
MICE_PORTS_STR="${MICE_PORTS_STR:-5202 5203 5204 5205 5206 5207 5208}"
read -r -a MICE_PORTS <<< "$MICE_PORTS_STR"

ELEPHANT_INTERVAL="${ELEPHANT_INTERVAL:-}"
ELEPHANT_DURATION="${ELEPHANT_DURATION:-}"
ELEPHANT_PARALLEL="${ELEPHANT_PARALLEL:-}"
ELEPHANT_BITRATE_MBPS="${ELEPHANT_BITRATE_MBPS:-}"

MICE_PROB="${MICE_PROB:-}"
MICE_MAX_PER_TICK="${MICE_MAX_PER_TICK:-}"
MICE_DURATION_MIN="${MICE_DURATION_MIN:-}"
MICE_DURATION_MAX="${MICE_DURATION_MAX:-}"
MICE_PARALLEL="${MICE_PARALLEL:-}"
MICE_BITRATE_MBPS="${MICE_BITRATE_MBPS:-}"

IPERF_INTERVAL_SECS="${IPERF_INTERVAL_SECS:-1}"
START_OFFSET_S="${START_OFFSET_S:-5}"

TS="$(date +%Y%m%d_%H%M%S)"
RAW_JSON_DIR="${LOG_DIR}/iperf_json_${TS}"
STATE_SNAPSHOT_DIR="${LOG_DIR}/state_snapshots_${TS}"
EVENTS_CSV="${LOG_DIR}/traffic_events_exp_mixed_${TS}.csv"
FLOW_INTERVALS_CSV="${LOG_DIR}/traffic_flow_intervals_exp_mixed_${TS}.csv"
MANIFEST_JSON="${LOG_DIR}/traffic_manifest_exp_mixed_${TS}.json"
LOCK_FILE="${LOG_DIR}/traffic_events_exp_mixed_${TS}.lock"

mkdir -p "$LOG_DIR" "$RAW_JSON_DIR" "$STATE_SNAPSHOT_DIR"
: > "$LOCK_FILE"
RANDOM="$SEED"

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S.%3N')] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_bin() {
  command -v "$1" >/dev/null 2>&1 || die "本机缺少命令: $1"
}

require_iperf3() {
  local container="$1"
  "${DOCKER[@]}" exec "$container" sh -lc 'command -v iperf3 >/dev/null 2>&1' \
    || die "容器 $container 里没有 iperf3"
}

set_default_if_empty() {
  local var_name="$1"
  local default_value="$2"
  if [[ -z "${!var_name}" ]]; then
    printf -v "$var_name" '%s' "$default_value"
  fi
}

apply_profile_defaults() {
  case "$TRAFFIC_PROFILE" in
    low)
      set_default_if_empty ELEPHANT_INTERVAL 45
      set_default_if_empty ELEPHANT_DURATION 12
      set_default_if_empty ELEPHANT_PARALLEL 2
      set_default_if_empty ELEPHANT_BITRATE_MBPS 150
      set_default_if_empty MICE_PROB 25
      set_default_if_empty MICE_MAX_PER_TICK 1
      set_default_if_empty MICE_DURATION_MIN 1
      set_default_if_empty MICE_DURATION_MAX 2
      set_default_if_empty MICE_PARALLEL 1
      set_default_if_empty MICE_BITRATE_MBPS 15
      ;;
    medium)
      set_default_if_empty ELEPHANT_INTERVAL 30
      set_default_if_empty ELEPHANT_DURATION 18
      set_default_if_empty ELEPHANT_PARALLEL 3
      set_default_if_empty ELEPHANT_BITRATE_MBPS 300
      set_default_if_empty MICE_PROB 50
      set_default_if_empty MICE_MAX_PER_TICK 1
      set_default_if_empty MICE_DURATION_MIN 1
      set_default_if_empty MICE_DURATION_MAX 3
      set_default_if_empty MICE_PARALLEL 1
      set_default_if_empty MICE_BITRATE_MBPS 20
      ;;
    high)
      set_default_if_empty ELEPHANT_INTERVAL 20
      set_default_if_empty ELEPHANT_DURATION 22
      set_default_if_empty ELEPHANT_PARALLEL 4
      set_default_if_empty ELEPHANT_BITRATE_MBPS 500
      set_default_if_empty MICE_PROB 75
      set_default_if_empty MICE_MAX_PER_TICK 2
      set_default_if_empty MICE_DURATION_MIN 1
      set_default_if_empty MICE_DURATION_MAX 4
      set_default_if_empty MICE_PARALLEL 1
      set_default_if_empty MICE_BITRATE_MBPS 30
      ;;
    *)
      die "未知 TRAFFIC_PROFILE=$TRAFFIC_PROFILE，可选: low / medium / high"
      ;;
  esac
}

to_json_array() {
  local first=1
  printf '['
  for item in "$@"; do
    if (( first == 0 )); then
      printf ','
    fi
    first=0
    printf '"%s"' "$item"
  done
  printf ']'
}

append_line_locked() {
  local file="$1"
  local line="$2"
  flock "$LOCK_FILE" bash -c 'printf "%s\n" "$1" >> "$2"' _ "$line" "$file"
}

write_manifest() {
  cat > "$MANIFEST_JSON" <<EOF
{
  "ts": "$TS",
  "seed": $SEED,
  "traffic_profile": "$TRAFFIC_PROFILE",
  "flow_proto": "$FLOW_PROTO",
  "h1_container": "$H1_CONTAINER",
  "h2_container": "$H2_CONTAINER",
  "dst_ip": "$DST_IP",
  "total_duration_s": $TOTAL_DURATION,
  "state_file": "$STATE_FILE",
  "events_csv": "$EVENTS_CSV",
  "flow_intervals_csv": "$FLOW_INTERVALS_CSV",
  "raw_json_dir": "$RAW_JSON_DIR",
  "state_snapshot_dir": "$STATE_SNAPSHOT_DIR",
  "elephant_port": $ELEPHANT_PORT,
  "mice_ports": $(to_json_array "${MICE_PORTS[@]}"),
  "elephant_interval_s": $ELEPHANT_INTERVAL,
  "elephant_duration_s": $ELEPHANT_DURATION,
  "elephant_parallel": $ELEPHANT_PARALLEL,
  "elephant_bitrate_mbps": $ELEPHANT_BITRATE_MBPS,
  "mice_prob_pct": $MICE_PROB,
  "mice_max_per_tick": $MICE_MAX_PER_TICK,
  "mice_duration_min_s": $MICE_DURATION_MIN,
  "mice_duration_max_s": $MICE_DURATION_MAX,
  "mice_parallel": $MICE_PARALLEL,
  "mice_bitrate_mbps": $MICE_BITRATE_MBPS,
  "iperf_interval_secs": $IPERF_INTERVAL_SECS
}
EOF
}

start_servers() {
  log "检查 iperf3"
  require_iperf3 "$H1_CONTAINER"
  require_iperf3 "$H2_CONTAINER"

  log "清理 $H2_CONTAINER 上旧的 iperf3 server"
  "${DOCKER[@]}" exec "$H2_CONTAINER" sh -lc 'pkill -x iperf3 >/dev/null 2>&1 || true'

  log "启动 $H2_CONTAINER 上的 iperf3 server"
  "${DOCKER[@]}" exec "$H2_CONTAINER" sh -lc "nohup iperf3 -s -p ${ELEPHANT_PORT} >/tmp/iperf3_${ELEPHANT_PORT}.log 2>&1 &"
  for p in "${MICE_PORTS[@]}"; do
    "${DOCKER[@]}" exec "$H2_CONTAINER" sh -lc "nohup iperf3 -s -p ${p} >/tmp/iperf3_${p}.log 2>&1 &"
  done

  sleep 1

  for p in "$ELEPHANT_PORT" "${MICE_PORTS[@]}"; do
    "${DOCKER[@]}" exec "$H2_CONTAINER" sh -lc "ss -lnt | grep -q ':${p} '" \
      || die "$H2_CONTAINER 上端口 ${p} 没有监听成功"
  done

  log "iperf3 server 已就绪"
}

snapshot_state_json() {
  local event_id="$1"
  local phase="$2"
  local out="${STATE_SNAPSHOT_DIR}/${event_id}_${phase}.json"

  python3 - "$STATE_FILE" "$out" "$FLOW_PROTO" "$TRAFFIC_PROFILE" "$STATE_READ_RETRIES" "$STATE_READ_RETRY_SLEEP_S" <<'PY'
import json
import os
import sys
import time

state_file, out_path, flow_proto, traffic_profile, retries, retry_sleep = sys.argv[1:]
retries = int(retries)
retry_sleep = float(retry_sleep)

default_state = {
    "mode": "unknown",
    "traffic_profile": traffic_profile,
    "active_policy": "unknown",
    "policy_id": "unknown",
    "candidate_path_id": "unknown",
    "policy_seq": 0,
    "decision_id": "none",
    "reroute_event": 0,
    "cooldown_active": 0,
    "event": "none",
    "event_ts": "",
    "cooldown_until": "",
    "elephant_path_hint": "unknown",
    "elephant_flow_id": "unknown",
    "elephant_topk_rank": "",
    "ingress_node": "unknown",
    "segment_list": "",
    "controller_epoch_s": "",
    "gain_mlu_pct": "",
    "gain_rtt_ms": "",
    "flow_proto": flow_proto,
}

if state_file and os.path.exists(state_file):
    for attempt in range(retries + 1):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                default_state.update(data)
            break
        except json.JSONDecodeError:
            if attempt < retries:
                time.sleep(retry_sleep)
                continue
            break
        except Exception:
            break

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(default_state, f, ensure_ascii=False, sort_keys=True)
PY
  printf '%s' "$out"
}

extract_state_summary() {
  local snapshot_path="$1"
  python3 - "$snapshot_path" <<'PY'
import json
import sys

path = sys.argv[1]
fields = [
    "mode",
    "active_policy",
    "policy_id",
    "candidate_path_id",
    "policy_seq",
    "decision_id",
    "cooldown_active",
    "elephant_path_hint",
    "elephant_topk_rank",
    "ingress_node",
    "controller_epoch_s",
]
default = ["unknown", "unknown", "unknown", "unknown", "0", "none", "0", "unknown", "", "unknown", ""]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for idx, key in enumerate(fields):
        value = data.get(key, default[idx])
        text = "" if value is None else str(value)
        text = text.replace(",", ";").replace("\n", " ").replace("\r", " ")
        out.append(text)
    print(",".join(out))
except Exception:
    print(",".join(default))
PY
}

parse_iperf_summary() {
  local json_path="$1"
  python3 - "$json_path" <<'PY'
import json
import sys

path = sys.argv[1]
out = {
    "throughput_mbps": "",
    "jitter_ms": "",
    "lost_percent": "",
    "retransmits": "",
}

def pick(obj, path):
    cur = obj
    for k in path:
        cur = cur[k]
    return cur

try:
    txt = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end < start:
        print(",".join(out.values()))
        raise SystemExit

    data = json.loads(txt[start:end + 1])

    throughput_candidates = [
        ("end", "sum_received", "bits_per_second"),
        ("end", "sum_sent", "bits_per_second"),
        ("end", "sum", "bits_per_second"),
        ("intervals", -1, "sum", "bits_per_second"),
    ]
    retrans_candidates = [
        ("end", "sum_sent", "retransmits"),
        ("end", "sum_sent", "retransmits_total"),
        ("end", "streams", 0, "sender", "retransmits"),
    ]
    jitter_candidates = [
        ("end", "sum", "jitter_ms"),
        ("end", "sum_received", "jitter_ms"),
        ("end", "sum_sent", "jitter_ms"),
    ]
    loss_candidates = [
        ("end", "sum", "lost_percent"),
        ("end", "sum_received", "lost_percent"),
        ("end", "sum_sent", "lost_percent"),
    ]

    for c in throughput_candidates:
        try:
            v = pick(data, c)
            if isinstance(v, (int, float)):
                out["throughput_mbps"] = f"{v / 1_000_000:.3f}"
                break
        except Exception:
            pass

    for c in jitter_candidates:
        try:
            v = pick(data, c)
            if isinstance(v, (int, float)):
                out["jitter_ms"] = f"{v:.3f}"
                break
        except Exception:
            pass

    for c in loss_candidates:
        try:
            v = pick(data, c)
            if isinstance(v, (int, float)):
                out["lost_percent"] = f"{v:.3f}"
                break
        except Exception:
            pass

    for c in retrans_candidates:
        try:
            v = pick(data, c)
            if isinstance(v, (int, float)):
                out["retransmits"] = str(int(v))
                break
        except Exception:
            pass

    print(f"{out['throughput_mbps']},{out['jitter_ms']},{out['lost_percent']},{out['retransmits']}")
except Exception:
    print(f"{out['throughput_mbps']},{out['jitter_ms']},{out['lost_percent']},{out['retransmits']}")
PY
}

append_interval_rows_from_json() {
  local json_path="$1"
  local flow_type="$2"
  local event_id="$3"
  local actual_start_ts="$4"
  local actual_start_epoch_ms="$5"
  local profile="$6"
  local port="$7"
  local duration="$8"
  local parallel="$9"
  local bitrate_mbps="${10}"
  local rc="${11}"
  local start_active_policy="${12}"
  local start_policy_id="${13}"
  local start_candidate_path_id="${14}"
  local start_elephant_path_hint="${15}"

  python3 - "$json_path" "$FLOW_INTERVALS_CSV" "$flow_type" "$event_id" "$actual_start_ts" "$actual_start_epoch_ms" "$profile" "$FLOW_PROTO" "$SEED" "$port" "$duration" "$parallel" "$bitrate_mbps" "$rc" "$start_active_policy" "$start_policy_id" "$start_candidate_path_id" "$start_elephant_path_hint" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone, timedelta

(
    json_path,
    csv_path,
    flow_type,
    event_id,
    actual_start_ts,
    actual_start_epoch_ms,
    profile,
    flow_proto,
    seed,
    port,
    duration,
    parallel,
    bitrate_mbps,
    rc,
    start_active_policy,
    start_policy_id,
    start_candidate_path_id,
    start_elephant_path_hint,
) = sys.argv[1:]

start_epoch_ms = int(actual_start_epoch_ms)

try:
    txt = open(json_path, "r", encoding="utf-8", errors="ignore").read().strip()
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON payload found")
    data = json.loads(txt[start:end + 1])
except Exception:
    raise SystemExit(0)

intervals = data.get("intervals") or []
rows = []
for idx, item in enumerate(intervals):
    sum_obj = item.get("sum") or {}
    start_s = float(sum_obj.get("start", idx))
    end_s = float(sum_obj.get("end", idx + 1))
    sec = float(sum_obj.get("seconds", max(end_s - start_s, 0.0)))
    bps = sum_obj.get("bits_per_second")
    throughput_mbps = round(float(bps) / 1_000_000, 3) if isinstance(bps, (int, float)) else ""
    jitter_ms = sum_obj.get("jitter_ms", "")
    lost_percent = sum_obj.get("lost_percent", "")
    retransmits = sum_obj.get("retransmits", sum_obj.get("retransmits_total", ""))
    interval_start_epoch_ms = start_epoch_ms + int(round(start_s * 1000))
    interval_end_epoch_ms = start_epoch_ms + int(round(end_s * 1000))
    interval_start_ts = datetime.fromtimestamp(interval_start_epoch_ms / 1000, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")
    interval_end_ts = datetime.fromtimestamp(interval_end_epoch_ms / 1000, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")

    rows.append(
        {
            "interval_start_ts": interval_start_ts,
            "interval_end_ts": interval_end_ts,
            "interval_start_epoch_ms": interval_start_epoch_ms,
            "interval_end_epoch_ms": interval_end_epoch_ms,
            "flow_type": flow_type,
            "event_id": event_id,
            "traffic_profile": profile,
            "flow_proto": flow_proto,
            "seed": seed,
            "port": port,
            "duration_s": duration,
            "parallel": parallel,
            "target_bitrate_mbps": bitrate_mbps,
            "rc": rc,
            "interval_index": idx,
            "interval_start_s": round(start_s, 3),
            "interval_end_s": round(end_s, 3),
            "interval_seconds": round(sec, 3),
            "throughput_mbps": throughput_mbps,
            "retransmits": retransmits,
            "jitter_ms": jitter_ms,
            "lost_percent": lost_percent,
            "start_active_policy": start_active_policy,
            "start_policy_id": start_policy_id,
            "start_candidate_path_id": start_candidate_path_id,
            "start_elephant_path_hint": start_elephant_path_hint,
        }
    )

if not rows:
    raise SystemExit(0)

with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    for row in rows:
        writer.writerow(row)
PY
}

build_client_cmd() {
  local port="$1"
  local duration="$2"
  local parallel="$3"
  local bitrate_mbps="$4"

  local cmd="iperf3 -c ${DST_IP} -p ${port} -t ${duration} -P ${parallel} -i ${IPERF_INTERVAL_SECS} -J"

  if [[ "$FLOW_PROTO" == "udp" ]]; then
    if [[ "$bitrate_mbps" == "0" || -z "$bitrate_mbps" ]]; then
      die "UDP 模式下必须设置 *_BITRATE_MBPS"
    fi
    cmd+=" -u -b ${bitrate_mbps}M"
  else
    if [[ "$bitrate_mbps" != "0" && -n "$bitrate_mbps" ]]; then
      cmd+=" --bitrate ${bitrate_mbps}M"
    fi
  fi

  printf '%s' "$cmd"
}

run_flow_async() {
  local flow_type="$1"
  local event_id="$2"
  local port="$3"
  local duration="$4"
  local parallel="$5"
  local bitrate_mbps="$6"
  local planned_start_s="$7"
  local profile="$8"

  (
    local actual_start_ts actual_end_ts actual_start_epoch_ms actual_end_epoch_ms
    local tmp_out raw_json_path cmd rc summary
    local start_state_path end_state_path start_state_csv end_state_csv
    local start_mode start_active_policy start_policy_id start_candidate_path_id start_policy_seq start_decision_id start_cooldown_active start_elephant_path_hint start_topk_rank start_ingress_node start_controller_epoch_s
    local end_mode end_active_policy end_policy_id end_candidate_path_id end_policy_seq end_decision_id end_cooldown_active end_elephant_path_hint end_topk_rank end_ingress_node end_controller_epoch_s
    local throughput_mbps jitter_ms lost_percent retransmits

    actual_start_ts="$(date '+%Y-%m-%dT%H:%M:%S.%3N')"
    actual_start_epoch_ms="$(date +%s%3N)"
    start_state_path="$(snapshot_state_json "$event_id" "start")"
    start_state_csv="$(extract_state_summary "$start_state_path")"
    IFS=',' read -r start_mode start_active_policy start_policy_id start_candidate_path_id start_policy_seq start_decision_id start_cooldown_active start_elephant_path_hint start_topk_rank start_ingress_node start_controller_epoch_s <<< "$start_state_csv"

    tmp_out="$(mktemp)"
    raw_json_path="${RAW_JSON_DIR}/${event_id}.json"
    cmd="$(build_client_cmd "$port" "$duration" "$parallel" "$bitrate_mbps")"

    rc=0
    if ! "${DOCKER[@]}" exec "$H1_CONTAINER" sh -lc "$cmd" >"$tmp_out" 2>&1; then
      rc=$?
    fi
    cp "$tmp_out" "$raw_json_path" || true

    actual_end_ts="$(date '+%Y-%m-%dT%H:%M:%S.%3N')"
    actual_end_epoch_ms="$(date +%s%3N)"
    end_state_path="$(snapshot_state_json "$event_id" "end")"
    end_state_csv="$(extract_state_summary "$end_state_path")"
    IFS=',' read -r end_mode end_active_policy end_policy_id end_candidate_path_id end_policy_seq end_decision_id end_cooldown_active end_elephant_path_hint end_topk_rank end_ingress_node end_controller_epoch_s <<< "$end_state_csv"

    summary="$(parse_iperf_summary "$raw_json_path")"
    IFS=',' read -r throughput_mbps jitter_ms lost_percent retransmits <<< "$summary"

    append_interval_rows_from_json \
      "$raw_json_path" \
      "$flow_type" \
      "$event_id" \
      "$actual_start_ts" \
      "$actual_start_epoch_ms" \
      "$profile" \
      "$port" \
      "$duration" \
      "$parallel" \
      "$bitrate_mbps" \
      "$rc" \
      "$start_active_policy" \
      "$start_policy_id" \
      "$start_candidate_path_id" \
      "$start_elephant_path_hint"

    log "flow_type=${flow_type} event_id=${event_id} port=${port} dur=${duration}s P=${parallel} bitrate=${bitrate_mbps}Mbps rc=${rc} throughput=${throughput_mbps:-NA}Mbps retrans=${retransmits:-NA} start_policy=${start_active_policy} end_policy=${end_active_policy}"

    append_line_locked "$EVENTS_CSV" "${actual_start_ts},${actual_end_ts},${actual_start_epoch_ms},${actual_end_epoch_ms},${flow_type},${event_id},${profile},${FLOW_PROTO},${SEED},${planned_start_s},$((planned_start_s + duration)),${port},${duration},${parallel},${bitrate_mbps},${rc},${throughput_mbps},${jitter_ms},${lost_percent},${retransmits},${start_mode},${start_active_policy},${start_policy_id},${start_candidate_path_id},${start_policy_seq},${start_decision_id},${start_cooldown_active},${start_elephant_path_hint},${start_topk_rank},${start_ingress_node},${start_controller_epoch_s},${end_mode},${end_active_policy},${end_policy_id},${end_candidate_path_id},${end_policy_seq},${end_decision_id},${end_cooldown_active},${end_elephant_path_hint},${end_topk_rank},${end_ingress_node},${end_controller_epoch_s},${raw_json_path},${start_state_path},${end_state_path}"

    rm -f "$tmp_out"
  ) &
}

main() {
  require_bin flock
  require_bin python3
  apply_profile_defaults
  write_manifest
  start_servers

  echo "actual_start_ts,actual_end_ts,actual_start_epoch_ms,actual_end_epoch_ms,flow_type,event_id,traffic_profile,flow_proto,seed,planned_start_s,planned_end_s,port,duration_s,parallel,target_bitrate_mbps,rc,throughput_mbps,jitter_ms,lost_percent,retransmits,start_mode,start_active_policy,start_policy_id,start_candidate_path_id,start_policy_seq,start_decision_id,start_cooldown_active,start_elephant_path_hint,start_topk_rank,start_ingress_node,start_controller_epoch_s,end_mode,end_active_policy,end_policy_id,end_candidate_path_id,end_policy_seq,end_decision_id,end_cooldown_active,end_elephant_path_hint,end_topk_rank,end_ingress_node,end_controller_epoch_s,raw_json_path,start_state_snapshot,end_state_snapshot" > "$EVENTS_CSV"

  echo "interval_start_ts,interval_end_ts,interval_start_epoch_ms,interval_end_epoch_ms,flow_type,event_id,traffic_profile,flow_proto,seed,port,duration_s,parallel,target_bitrate_mbps,rc,interval_index,interval_start_s,interval_end_s,interval_seconds,throughput_mbps,retransmits,jitter_ms,lost_percent,start_active_policy,start_policy_id,start_candidate_path_id,start_elephant_path_hint" > "$FLOW_INTERVALS_CSV"

  log "H1_CONTAINER=${H1_CONTAINER}"
  log "H2_CONTAINER=${H2_CONTAINER}"
  log "DST_IP=${DST_IP}"
  log "TOTAL_DURATION=${TOTAL_DURATION}s"
  log "TRAFFIC_PROFILE=${TRAFFIC_PROFILE}"
  log "FLOW_PROTO=${FLOW_PROTO}"
  log "STATE_FILE=${STATE_FILE:-none}"
  log "EVENTS_CSV=${EVENTS_CSV}"
  log "FLOW_INTERVALS_CSV=${FLOW_INTERVALS_CSV}"
  log "MANIFEST_JSON=${MANIFEST_JSON}"

  declare -A PORT_FREE_AT
  for p in "${MICE_PORTS[@]}"; do
    PORT_FREE_AT["$p"]=0
  done

  local elephant_id=0
  local mice_id=0
  local exp_start now elapsed t sleep_to_next dur p k

  exp_start="$(date +%s)"

  while true; do
    now="$(date +%s)"
    elapsed=$((now - exp_start))
    if (( elapsed >= TOTAL_DURATION )); then
      break
    fi
    t="$elapsed"

    if (( t >= START_OFFSET_S )) && (( (t - START_OFFSET_S) % ELEPHANT_INTERVAL == 0 )); then
      run_flow_async \
        "elephant" \
        "elephant_${elephant_id}" \
        "$ELEPHANT_PORT" \
        "$ELEPHANT_DURATION" \
        "$ELEPHANT_PARALLEL" \
        "$ELEPHANT_BITRATE_MBPS" \
        "$t" \
        "$TRAFFIC_PROFILE"
      ((elephant_id+=1))
    fi

    for ((k=0; k<MICE_MAX_PER_TICK; k++)); do
      if (( RANDOM % 100 >= MICE_PROB )); then
        continue
      fi

      for p in "${MICE_PORTS[@]}"; do
        if (( t >= PORT_FREE_AT[$p] )); then
          dur=$((RANDOM % (MICE_DURATION_MAX - MICE_DURATION_MIN + 1) + MICE_DURATION_MIN))
          run_flow_async \
            "mice" \
            "mice_${mice_id}" \
            "$p" \
            "$dur" \
            "$MICE_PARALLEL" \
            "$MICE_BITRATE_MBPS" \
            "$t" \
            "$TRAFFIC_PROFILE"
          PORT_FREE_AT["$p"]=$((t + dur + 1))
          ((mice_id+=1))
          break
        fi
      done
    done

    sleep_to_next=$((exp_start + t + 1 - $(date +%s)))
    if (( sleep_to_next > 0 )); then
      sleep "$sleep_to_next"
    else
      sleep 0.2
    fi
  done

  wait
  log "全部流量已完成"
  log "事件日志: $EVENTS_CSV"
  log "流级采样日志: $FLOW_INTERVALS_CSV"
  log "原始 JSON 目录: $RAW_JSON_DIR"
  log "状态快照目录: $STATE_SNAPSHOT_DIR"
  log "实验清单: $MANIFEST_JSON"
}

main "$@"