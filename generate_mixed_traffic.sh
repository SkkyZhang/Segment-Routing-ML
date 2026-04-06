#!/usr/bin/env bash
set -Eeuo pipefail

DOCKER_CMD="${DOCKER_CMD:-sudo docker}"
read -r -a DOCKER <<< "$DOCKER_CMD"

H1_CONTAINER="${H1_CONTAINER:-clab-srte4-h1}"
H2_CONTAINER="${H2_CONTAINER:-clab-srte4-h2}"
DST_IP="${DST_IP:-192.168.4.2}"

TOTAL_DURATION="${TOTAL_DURATION:-1800}"   # 采集时长（秒），建议至少 30 分钟
LOG_DIR="${LOG_DIR:-data}"
SEED="${SEED:-42}"
TRAFFIC_PROFILE="${TRAFFIC_PROFILE:-medium}"

FLOW_PROTO="${FLOW_PROTO:-tcp}"            # tcp / udp

ELEPHANT_PORT="${ELEPHANT_PORT:-5201}"
MICE_PORTS=(5202 5203 5204 5205 5206 5207 5208)

# 留空表示“由 profile 决定”，手动导出环境变量时可覆盖
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

TS="$(date +%Y%m%d_%H%M%S)"
EVENTS_CSV="${LOG_DIR}/traffic_events_exp_mixed_${TS}.csv"
MANIFEST_JSON="${LOG_DIR}/traffic_manifest_exp_mixed_${TS}.json"
LOCK_FILE="${LOG_DIR}/traffic_events_exp_mixed_${TS}.lock"

mkdir -p "$LOG_DIR"
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
  "mice_bitrate_mbps": $MICE_BITRATE_MBPS
}
EOF
}

append_csv_row() {
  local row="$1"
  flock "$LOCK_FILE" bash -c 'printf "%s\n" "$1" >> "$2"' _ "$row" "$EVENTS_CSV"
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

parse_iperf_metrics() {
  local file="$1"
  python3 - "$file" <<'PY'
import json
import sys

path = sys.argv[1]
out = {
    "throughput_mbps": "",
    "jitter_ms": "",
    "lost_percent": "",
}

try:
    txt = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end < start:
        print(f"{out['throughput_mbps']},{out['jitter_ms']},{out['lost_percent']}")
        raise SystemExit

    data = json.loads(txt[start:end + 1])

    candidates = [
        ("end", "sum_received", "bits_per_second"),
        ("end", "sum_sent", "bits_per_second"),
        ("end", "streams", 0, "sender", "bits_per_second"),
        ("intervals", -1, "sum", "bits_per_second"),
    ]

    def pick(obj, path):
        cur = obj
        for k in path:
            cur = cur[k]
        return cur

    value = None
    for c in candidates:
        try:
            v = pick(data, c)
            if isinstance(v, (int, float)):
                value = v
                break
        except Exception:
            pass

    if value is not None:
        out["throughput_mbps"] = f"{value / 1_000_000:.3f}"

    try:
        jit = data["end"]["sum"].get("jitter_ms")
        if isinstance(jit, (int, float)):
            out["jitter_ms"] = f"{jit:.3f}"
    except Exception:
        pass

    try:
        loss = data["end"]["sum"].get("lost_percent")
        if isinstance(loss, (int, float)):
            out["lost_percent"] = f"{loss:.3f}"
    except Exception:
        pass

    print(f"{out['throughput_mbps']},{out['jitter_ms']},{out['lost_percent']}")
except Exception:
    print(f"{out['throughput_mbps']},{out['jitter_ms']},{out['lost_percent']}")
PY
}

build_client_cmd() {
  local port="$1"
  local duration="$2"
  local parallel="$3"
  local bitrate_mbps="$4"

  local cmd="iperf3 -c ${DST_IP} -p ${port} -t ${duration} -P ${parallel} -J"

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
    local actual_start_ts actual_end_ts rc tmp_out cmd
    actual_start_ts="$(date '+%Y-%m-%dT%H:%M:%S.%3N')"
    tmp_out="$(mktemp)"
    cmd="$(build_client_cmd "$port" "$duration" "$parallel" "$bitrate_mbps")"

    rc=0
    if ! "${DOCKER[@]}" exec "$H1_CONTAINER" sh -lc "$cmd" >"$tmp_out" 2>&1; then
      rc=$?
    fi
    actual_end_ts="$(date '+%Y-%m-%dT%H:%M:%S.%3N')"

    local metrics throughput_mbps jitter_ms lost_percent
    metrics="$(parse_iperf_metrics "$tmp_out")"
    IFS=',' read -r throughput_mbps jitter_ms lost_percent <<< "$metrics"

    log "flow_type=${flow_type} event_id=${event_id} port=${port} dur=${duration}s P=${parallel} bitrate=${bitrate_mbps}Mbps rc=${rc} throughput=${throughput_mbps:-NA}Mbps"

    append_csv_row "${actual_start_ts},${actual_end_ts},${flow_type},${event_id},${profile},${FLOW_PROTO},${SEED},${planned_start_s},$((planned_start_s + duration)),${port},${duration},${parallel},${bitrate_mbps},${rc},${throughput_mbps},${jitter_ms},${lost_percent}"

    rm -f "$tmp_out"
  ) &
}

main() {
  require_bin flock
  require_bin python3
  apply_profile_defaults
  write_manifest
  start_servers

  echo "actual_start_ts,actual_end_ts,flow_type,event_id,traffic_profile,flow_proto,seed,planned_start_s,planned_end_s,port,duration_s,parallel,target_bitrate_mbps,rc,throughput_mbps,jitter_ms,lost_percent" > "$EVENTS_CSV"

  log "H1_CONTAINER=${H1_CONTAINER}"
  log "H2_CONTAINER=${H2_CONTAINER}"
  log "DST_IP=${DST_IP}"
  log "TOTAL_DURATION=${TOTAL_DURATION}s"
  log "TRAFFIC_PROFILE=${TRAFFIC_PROFILE}"
  log "FLOW_PROTO=${FLOW_PROTO}"
  log "EVENTS_CSV=${EVENTS_CSV}"
  log "MANIFEST_JSON=${MANIFEST_JSON}"

  declare -A PORT_FREE_AT
  for p in "${MICE_PORTS[@]}"; do
    PORT_FREE_AT["$p"]=0
  done

  local elephant_id=0
  local mice_id=0
  local exp_start now elapsed t sleep_to_next dur p

  exp_start="$(date +%s)"

  while true; do
    now="$(date +%s)"
    elapsed=$((now - exp_start))
    if (( elapsed >= TOTAL_DURATION )); then
      break
    fi
    t="$elapsed"

    if (( t >= 2 )) && (( t % ELEPHANT_INTERVAL == 5 )); then
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
  log "实验清单: $MANIFEST_JSON"
}

main "$@"