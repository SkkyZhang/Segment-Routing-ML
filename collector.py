import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DOCKER_CMD = shlex.split(os.environ.get("DOCKER_CMD", "sudo docker"))


def run_cmd(args: List[str], timeout: int = 5) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0 and result.stderr.strip():
            print(f"[WARN] Command failed: {' '.join(args)}")
            print(f"[WARN] stderr: {result.stderr.strip()}")
        return result
    except subprocess.TimeoutExpired as e:
        print(f"[WARN] Command timeout after {timeout}s: {' '.join(args)}")
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout="",
            stderr=str(e),
        )


def blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def sanitize_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    low = int(pos)
    high = min(low + 1, len(vals) - 1)
    frac = pos - low
    interp = vals[low] * (1.0 - frac) + vals[high] * frac
    return round(interp, 3)


def parse_targets(specs: List[str]) -> Dict[str, List[str]]:
    targets: Dict[str, List[str]] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid target spec: {spec}")
        container, iface_part = spec.split(":", 1)
        ifaces = [x.strip() for x in iface_part.split(",") if x.strip()]
        if not ifaces:
            raise ValueError(f"No interfaces provided in target spec: {spec}")
        targets[container.strip()] = ifaces
    return targets


def parse_capacities(specs: List[str]) -> Dict[Tuple[str, str], float]:
    capacities: Dict[Tuple[str, str], float] = {}
    for spec in specs:
        if "=" not in spec or ":" not in spec:
            raise ValueError(f"Invalid capacity spec: {spec}")
        lhs, rhs = spec.split("=", 1)
        container, iface = lhs.split(":", 1)
        capacities[(container.strip(), iface.strip())] = float(rhs.strip())
    return capacities


def get_iface_stats(
    container: str,
    ifaces: List[str],
    cmd_timeout: int = 5,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int], int]:
    safe_ifaces = " ".join(shlex.quote(i) for i in ifaces)
    shell_script = f"""
for i in {safe_ifaces}; do
    base="/sys/class/net/$i/statistics"
    if [ -r "$base/rx_bytes" ] && [ -r "$base/tx_bytes" ]; then
        rx_bytes=$(cat "$base/rx_bytes" 2>/dev/null || echo 0)
        tx_bytes=$(cat "$base/tx_bytes" 2>/dev/null || echo 0)
        rx_packets=$(cat "$base/rx_packets" 2>/dev/null || echo 0)
        tx_packets=$(cat "$base/tx_packets" 2>/dev/null || echo 0)
        rx_dropped=$(cat "$base/rx_dropped" 2>/dev/null || echo 0)
        tx_dropped=$(cat "$base/tx_dropped" 2>/dev/null || echo 0)
        rx_errors=$(cat "$base/rx_errors" 2>/dev/null || echo 0)
        tx_errors=$(cat "$base/tx_errors" 2>/dev/null || echo 0)
        printf "%s %s %s %s %s %s %s %s %s\\n" \
            "$i" "$rx_bytes" "$tx_bytes" "$rx_packets" "$tx_packets" \
            "$rx_dropped" "$tx_dropped" "$rx_errors" "$tx_errors"
    fi
done
"""
    result = run_cmd(
        [*DOCKER_CMD, "exec", container, "sh", "-lc", shell_script],
        timeout=cmd_timeout,
    )
    out = result.stdout.strip()

    stats: Dict[str, Dict[str, int]] = {}
    read_ok: Dict[str, int] = {iface: 0 for iface in ifaces}

    if result.returncode != 0:
        return stats, read_ok, 0

    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        iface = parts[0]
        if iface not in read_ok:
            continue
        try:
            stats[iface] = {
                "rx_bytes": int(parts[1]),
                "tx_bytes": int(parts[2]),
                "rx_packets": int(parts[3]),
                "tx_packets": int(parts[4]),
                "rx_dropped": int(parts[5]),
                "tx_dropped": int(parts[6]),
                "rx_errors": int(parts[7]),
                "tx_errors": int(parts[8]),
            }
            read_ok[iface] = 1
        except ValueError:
            read_ok[iface] = 0

    return stats, read_ok, 1


def parse_tc_qdisc_output(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "qdisc_kind": "",
        "qdisc_backlog_bytes": None,
        "qdisc_backlog_pkts": None,
        "qdisc_drops": None,
        "qdisc_requeues": None,
        "qdisc_overlimits": None,
    }
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if first_line.startswith("qdisc "):
        parts = first_line.split()
        if len(parts) >= 2:
            out["qdisc_kind"] = parts[1]

    drop_match = re.search(r"\bSent\b.*?\bdropped\s+(\d+).*?\boverlimits\s+(\d+).*?\brequeues\s+(\d+)", text, re.IGNORECASE | re.DOTALL)
    if drop_match:
        out["qdisc_drops"] = int(drop_match.group(1))
        out["qdisc_overlimits"] = int(drop_match.group(2))
        out["qdisc_requeues"] = int(drop_match.group(3))

    backlog_match = re.search(r"\bbacklog\s+([0-9]+)([KMG]?b)\s+([0-9]+)p", text, re.IGNORECASE)
    if backlog_match:
        raw_value = int(backlog_match.group(1))
        unit = backlog_match.group(2).lower()
        multiplier = {
            "b": 1,
            "kb": 1024,
            "mb": 1024 * 1024,
            "gb": 1024 * 1024 * 1024,
        }.get(unit, 1)
        out["qdisc_backlog_bytes"] = raw_value * multiplier
        out["qdisc_backlog_pkts"] = int(backlog_match.group(3))

    return out


def get_qdisc_stats(
    container: str,
    ifaces: List[str],
    enabled: bool,
    cmd_timeout: int = 5,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int], int]:
    stats: Dict[str, Dict[str, Any]] = {}
    read_ok: Dict[str, int] = {iface: 0 for iface in ifaces}
    if not enabled:
        return stats, read_ok, 1

    overall_ok = 1
    for iface in ifaces:
        result = run_cmd(
            [*DOCKER_CMD, "exec", container, "sh", "-lc", f"tc -s qdisc show dev {shlex.quote(iface)}"],
            timeout=cmd_timeout,
        )
        if result.returncode != 0:
            overall_ok = 0
            continue
        parsed = parse_tc_qdisc_output(result.stdout or "")
        stats[iface] = parsed
        read_ok[iface] = 1

    return stats, read_ok, overall_ok


def get_ping_stats(
    container: str,
    dst: str,
    count: int = 5,
    timeout: int = 1,
    interval: float = 0.2,
    cmd_timeout: int = 5,
) -> Dict[str, Any]:
    effective_cmd_timeout = max(
        cmd_timeout,
        int(count * max(interval, 0.0) + count * max(timeout, 1) + 2),
    )

    ping_cmd = [
        *DOCKER_CMD,
        "exec",
        container,
        "ping",
        "-n",
        "-c",
        str(count),
        "-W",
        str(timeout),
        "-i",
        str(interval),
        dst,
    ]
    result = run_cmd(ping_cmd, timeout=effective_cmd_timeout)
    out = (result.stdout or "") + "\n" + (result.stderr or "")

    reply_rtts = [
        float(x)
        for x in re.findall(r"time[=<]([0-9.]+)\s*ms", out, flags=re.IGNORECASE)
    ]

    stats: Dict[str, Any] = {
        "ping_cmd_ok": 1 if result.returncode in (0, 1) else 0,
        "ping_ok": 1 if reply_rtts else 0,
        "ping_sent": count,
        "ping_received": None,
        "ping_loss_pct": None,
        "ping_reply_count": len(reply_rtts),
        "ping_rtts_ms": ";".join(f"{v:.3f}" for v in reply_rtts),
        "ping_rtt_min_ms": None,
        "ping_rtt_avg_ms": None,
        "ping_rtt_max_ms": None,
        "ping_rtt_mdev_ms": None,
        "ping_rtt_p50_ms": percentile(reply_rtts, 0.50),
        "ping_rtt_p95_ms": percentile(reply_rtts, 0.95),
        "ping_rtt_p99_ms": percentile(reply_rtts, 0.99),
    }

    loss_match = re.search(
        r"(\d+)\s+packets transmitted,\s+(\d+)\s+(?:packets )?received.*?([0-9.]+)%\s+packet loss",
        out,
        re.IGNORECASE | re.DOTALL,
    )
    if loss_match:
        stats["ping_sent"] = int(loss_match.group(1))
        stats["ping_received"] = int(loss_match.group(2))
        stats["ping_loss_pct"] = float(loss_match.group(3))
        stats["ping_ok"] = 1 if stats["ping_received"] > 0 else 0
    else:
        stats["ping_received"] = len(reply_rtts)
        if count > 0:
            stats["ping_loss_pct"] = round((1.0 - len(reply_rtts) / count) * 100.0, 3)

    rtt_match = re.search(
        r"=\s*([0-9.]+)/([0-9.]+)/([0-9.]+)(?:/([0-9.]+))?\s*ms",
        out,
        re.IGNORECASE,
    )
    if rtt_match:
        stats["ping_rtt_min_ms"] = float(rtt_match.group(1))
        stats["ping_rtt_avg_ms"] = float(rtt_match.group(2))
        stats["ping_rtt_max_ms"] = float(rtt_match.group(3))
        if rtt_match.group(4) is not None:
            stats["ping_rtt_mdev_ms"] = float(rtt_match.group(4))
    elif reply_rtts:
        stats["ping_rtt_min_ms"] = round(min(reply_rtts), 3)
        stats["ping_rtt_avg_ms"] = round(sum(reply_rtts) / len(reply_rtts), 3)
        stats["ping_rtt_max_ms"] = round(max(reply_rtts), 3)

    return stats


def load_runtime_state(
    state_file: Optional[str],
    cli_mode: str,
    cli_traffic_profile: str,
    cli_active_policy: str,
    read_retries: int = 2,
    retry_sleep_s: float = 0.03,
) -> Tuple[Dict[str, Any], int]:
    default_state: Dict[str, Any] = {
        "mode": cli_mode,
        "traffic_profile": cli_traffic_profile,
        "active_policy": cli_active_policy,
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
    }

    if not state_file:
        return default_state, 0

    if not os.path.exists(state_file):
        return default_state, 0

    for attempt in range(read_retries + 1):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                default_state.update(data)
                return default_state, 1
            print(f"[WARN] State file is not a JSON object: {state_file}")
            return default_state, 0
        except json.JSONDecodeError as e:
            if attempt < read_retries:
                time.sleep(retry_sleep_s)
                continue
            print(f"[WARN] Failed to parse state file {state_file}: {e}")
            return default_state, 0
        except Exception as e:
            print(f"[WARN] Failed to read state file {state_file}: {e}")
            return default_state, 0

    return default_state, 0


def delta_with_reset(cur: int, prev: int) -> Tuple[int, int]:
    delta = cur - prev
    if delta < 0:
        return 0, 1
    return delta, 0


def pick_dominant_egress_iface(
    ifaces: List[str],
    tx_map: Dict[str, float],
    idle_threshold_mbps: float = 0.1,
) -> str:
    best_iface = "unknown"
    best_rate = -1.0

    for iface in ifaces:
        rate = tx_map.get(iface, 0.0)
        if rate > best_rate:
            best_rate = rate
            best_iface = iface

    if best_rate < idle_threshold_mbps:
        return "idle"
    return best_iface


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extended telemetry collector for containerlab SR-TE experiments."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="Container/interface specs, e.g. clab-srte4-r1:eth1,eth2 clab-srte4-r2:eth1,eth2",
    )
    parser.add_argument(
        "--capacities",
        nargs="+",
        default=[],
        help="Link capacities in Mbps, e.g. clab-srte4-r1:eth1=1000 clab-srte4-r1:eth2=1000",
    )

    parser.add_argument("--ping-src", default="clab-srte4-h1")
    parser.add_argument("--ping-dst", default="192.168.4.2")
    parser.add_argument("--ping-count", type=int, default=5)
    parser.add_argument("--ping-timeout", type=int, default=1)
    parser.add_argument("--ping-interval", type=float, default=0.2)

    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--outdir", default="data")
    parser.add_argument("--cmd-timeout", type=int, default=5)

    parser.add_argument("--experiment-id", default="exp_default")
    parser.add_argument("--mode", default="baseline")
    parser.add_argument("--traffic-profile", default="unknown")
    parser.add_argument("--active-policy", default="unknown")
    parser.add_argument("--state-file", default=None, help="Optional JSON file updated atomically by controller")

    parser.add_argument("--util-threshold-pct", type=float, default=70.0)
    parser.add_argument("--dominant-idle-threshold-mbps", type=float, default=0.1)
    parser.add_argument("--collect-qdisc", action="store_true")

    args = parser.parse_args()

    estimated_probe_budget = (
        args.ping_count * max(args.ping_interval, 0.0) + max(args.ping_timeout, 1)
    )
    if estimated_probe_budget > args.interval:
        print(
            "[WARN] The probe budget is larger than the sampling interval. "
            "The actual loop period may exceed --interval. "
            "Consider reducing --ping-count / --ping-interval or increasing --interval."
        )

    targets = parse_targets(args.targets)
    capacities = parse_capacities(args.capacities)

    os.makedirs(args.outdir, exist_ok=True)
    ts_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    long_outfile = os.path.join(args.outdir, f"telemetry_long_{ts_prefix}.csv")
    wide_outfile = os.path.join(args.outdir, f"telemetry_wide_{ts_prefix}.csv")

    prev_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    prev_qdisc: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for container, ifaces in targets.items():
        stats, _, _ = get_iface_stats(container, ifaces, cmd_timeout=args.cmd_timeout)
        prev_stats[container] = stats
        qdisc_stats, _, _ = get_qdisc_stats(container, ifaces, enabled=args.collect_qdisc, cmd_timeout=args.cmd_timeout)
        prev_qdisc[container] = qdisc_stats

    prev_sample_time = time.monotonic()
    start_time = prev_sample_time
    sample_id = 0

    runtime_fields = [
        "state_read_ok",
        "mode",
        "traffic_profile",
        "active_policy",
        "policy_id",
        "candidate_path_id",
        "policy_seq",
        "decision_id",
        "reroute_event",
        "cooldown_active",
        "event",
        "event_ts",
        "cooldown_until",
        "elephant_path_hint",
        "elephant_flow_id",
        "elephant_topk_rank",
        "ingress_node",
        "segment_list",
        "controller_epoch_s",
        "gain_mlu_pct",
        "gain_rtt_ms",
    ]

    ping_fields = [
        "ping_src",
        "ping_dst",
        "ping_cmd_ok",
        "ping_ok",
        "ping_sent",
        "ping_received",
        "ping_reply_count",
        "ping_loss_pct",
        "ping_rtts_ms",
        "ping_rtt_min_ms",
        "ping_rtt_avg_ms",
        "ping_rtt_max_ms",
        "ping_rtt_mdev_ms",
        "ping_rtt_p50_ms",
        "ping_rtt_p95_ms",
        "ping_rtt_p99_ms",
    ]

    long_fieldnames = [
        "timestamp",
        "wall_time_epoch_ms",
        "sample_id",
        "experiment_id",
        "elapsed_s",
        "dt_s",
        *runtime_fields,
        *ping_fields,
        "router",
        "iface",
        "counter_cmd_ok",
        "counter_read_ok",
        "counter_reset",
        "qdisc_cmd_ok",
        "qdisc_read_ok",
        "link_capacity_mbps",
        "missing_capacity",
        "rx_bytes",
        "tx_bytes",
        "rx_packets",
        "tx_packets",
        "rx_dropped",
        "tx_dropped",
        "rx_errors",
        "tx_errors",
        "rx_delta_bytes",
        "tx_delta_bytes",
        "rx_delta_packets",
        "tx_delta_packets",
        "rx_drop_delta",
        "tx_drop_delta",
        "rx_err_delta",
        "tx_err_delta",
        "rx_mbps",
        "tx_mbps",
        "rx_pps",
        "tx_pps",
        "util_pct",
        "dominant_egress_iface",
        "qdisc_kind",
        "qdisc_backlog_bytes",
        "qdisc_backlog_pkts",
        "qdisc_drops",
        "qdisc_requeues",
        "qdisc_overlimits",
        "qdisc_drop_delta",
        "qdisc_requeue_delta",
        "qdisc_overlimit_delta",
        "network_mlu_pct",
        "util_threshold_pct",
        "is_hot_link",
    ]

    wide_fieldnames = [
        "timestamp",
        "wall_time_epoch_ms",
        "sample_id",
        "experiment_id",
        "elapsed_s",
        "dt_s",
        *runtime_fields,
        *ping_fields,
        "network_mlu_pct",
        "hot_links",
    ]

    per_iface_wide_suffixes = [
        "counter_cmd_ok",
        "counter_read_ok",
        "counter_reset",
        "qdisc_cmd_ok",
        "qdisc_read_ok",
        "rx_bytes",
        "tx_bytes",
        "rx_packets",
        "tx_packets",
        "rx_dropped",
        "tx_dropped",
        "rx_errors",
        "tx_errors",
        "rx_delta_bytes",
        "tx_delta_bytes",
        "rx_delta_packets",
        "tx_delta_packets",
        "rx_drop_delta",
        "tx_drop_delta",
        "rx_err_delta",
        "tx_err_delta",
        "rx_mbps",
        "tx_mbps",
        "rx_pps",
        "tx_pps",
        "link_capacity_mbps",
        "util_pct",
        "is_hot_link",
        "qdisc_kind",
        "qdisc_backlog_bytes",
        "qdisc_backlog_pkts",
        "qdisc_drops",
        "qdisc_requeues",
        "qdisc_overlimits",
        "qdisc_drop_delta",
        "qdisc_requeue_delta",
        "qdisc_overlimit_delta",
    ]

    for container, ifaces in targets.items():
        router_key = sanitize_key(container)
        wide_fieldnames.append(f"{router_key}__dominant_egress_iface")
        for iface in ifaces:
            iface_key = sanitize_key(iface)
            prefix = f"{router_key}__{iface_key}"
            for suffix in per_iface_wide_suffixes:
                wide_fieldnames.append(f"{prefix}__{suffix}")

    print(f"Writing long telemetry to {long_outfile}")
    print(f"Writing wide telemetry to {wide_outfile}")

    try:
        with open(long_outfile, "w", newline="", encoding="utf-8") as long_f, open(
            wide_outfile, "w", newline="", encoding="utf-8"
        ) as wide_f:
            long_writer = csv.DictWriter(long_f, fieldnames=long_fieldnames)
            wide_writer = csv.DictWriter(wide_f, fieldnames=wide_fieldnames)
            long_writer.writeheader()
            wide_writer.writeheader()

            while True:
                loop_start = time.monotonic()
                elapsed = loop_start - start_time
                if elapsed > args.duration:
                    break

                dt = loop_start - prev_sample_time
                if dt <= 0:
                    dt = args.interval

                ts = datetime.now().isoformat(timespec="milliseconds")
                wall_time_epoch_ms = int(time.time() * 1000)

                runtime_state, state_read_ok = load_runtime_state(
                    args.state_file,
                    cli_mode=args.mode,
                    cli_traffic_profile=args.traffic_profile,
                    cli_active_policy=args.active_policy,
                )

                current_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
                current_qdisc: Dict[str, Dict[str, Dict[str, Any]]] = {}
                interval_rows: List[Dict[str, Any]] = []
                dominant_iface_by_router: Dict[str, str] = {}

                for container, ifaces in targets.items():
                    stats, read_ok, cmd_ok = get_iface_stats(
                        container,
                        ifaces,
                        cmd_timeout=args.cmd_timeout,
                    )
                    qdisc_stats, qdisc_read_ok, qdisc_cmd_ok = get_qdisc_stats(
                        container,
                        ifaces,
                        enabled=args.collect_qdisc,
                        cmd_timeout=args.cmd_timeout,
                    )
                    current_stats[container] = stats
                    current_qdisc[container] = qdisc_stats

                    tx_map: Dict[str, float] = {}

                    for iface in ifaces:
                        prev = prev_stats.get(container, {}).get(
                            iface,
                            {
                                "rx_bytes": 0,
                                "tx_bytes": 0,
                                "rx_packets": 0,
                                "tx_packets": 0,
                                "rx_dropped": 0,
                                "tx_dropped": 0,
                                "rx_errors": 0,
                                "tx_errors": 0,
                            },
                        )
                        counter_read_ok = read_ok.get(iface, 0)

                        if counter_read_ok == 1 and iface in stats:
                            cur = stats[iface]
                        else:
                            cur = prev

                        rx_delta, rx_reset = delta_with_reset(cur["rx_bytes"], prev["rx_bytes"])
                        tx_delta, tx_reset = delta_with_reset(cur["tx_bytes"], prev["tx_bytes"])
                        rx_pkt_delta, rx_pkt_reset = delta_with_reset(cur["rx_packets"], prev["rx_packets"])
                        tx_pkt_delta, tx_pkt_reset = delta_with_reset(cur["tx_packets"], prev["tx_packets"])
                        rx_drop_delta, rx_drop_reset = delta_with_reset(cur["rx_dropped"], prev["rx_dropped"])
                        tx_drop_delta, tx_drop_reset = delta_with_reset(cur["tx_dropped"], prev["tx_dropped"])
                        rx_err_delta, rx_err_reset = delta_with_reset(cur["rx_errors"], prev["rx_errors"])
                        tx_err_delta, tx_err_reset = delta_with_reset(cur["tx_errors"], prev["tx_errors"])

                        counter_reset = 1 if any(
                            (
                                rx_reset,
                                tx_reset,
                                rx_pkt_reset,
                                tx_pkt_reset,
                                rx_drop_reset,
                                tx_drop_reset,
                                rx_err_reset,
                                tx_err_reset,
                            )
                        ) else 0

                        rx_mbps = (rx_delta * 8 / dt) / 1_000_000
                        tx_mbps = (tx_delta * 8 / dt) / 1_000_000
                        rx_pps = rx_pkt_delta / dt
                        tx_pps = tx_pkt_delta / dt
                        tx_map[iface] = tx_mbps

                        capacity_mbps = capacities.get((container, iface))
                        missing_capacity = 1 if capacity_mbps is None or capacity_mbps <= 0 else 0
                        if missing_capacity == 0:
                            util_pct_num = (tx_mbps / capacity_mbps) * 100.0
                            is_hot_link = 1 if util_pct_num >= args.util_threshold_pct else 0
                            link_capacity_mbps = round(capacity_mbps, 3)
                            util_pct = round(util_pct_num, 3)
                        else:
                            util_pct_num = None
                            is_hot_link = 0
                            link_capacity_mbps = ""
                            util_pct = ""

                        q_prev = prev_qdisc.get(container, {}).get(iface, {})
                        q_cur = current_qdisc.get(container, {}).get(iface, {})
                        qdisc_read_ok_val = qdisc_read_ok.get(iface, 0)
                        qdisc_drops = q_cur.get("qdisc_drops") if qdisc_read_ok_val == 1 else q_prev.get("qdisc_drops")
                        qdisc_requeues = q_cur.get("qdisc_requeues") if qdisc_read_ok_val == 1 else q_prev.get("qdisc_requeues")
                        qdisc_overlimits = q_cur.get("qdisc_overlimits") if qdisc_read_ok_val == 1 else q_prev.get("qdisc_overlimits")

                        qdisc_drop_delta = ""
                        qdisc_requeue_delta = ""
                        qdisc_overlimit_delta = ""
                        if isinstance(qdisc_drops, int) and isinstance(q_prev.get("qdisc_drops"), int):
                            qdisc_drop_delta = delta_with_reset(qdisc_drops, q_prev["qdisc_drops"])[0]
                        if isinstance(qdisc_requeues, int) and isinstance(q_prev.get("qdisc_requeues"), int):
                            qdisc_requeue_delta = delta_with_reset(qdisc_requeues, q_prev["qdisc_requeues"])[0]
                        if isinstance(qdisc_overlimits, int) and isinstance(q_prev.get("qdisc_overlimits"), int):
                            qdisc_overlimit_delta = delta_with_reset(qdisc_overlimits, q_prev["qdisc_overlimits"])[0]

                        row: Dict[str, Any] = {
                            "timestamp": ts,
                            "wall_time_epoch_ms": wall_time_epoch_ms,
                            "sample_id": sample_id,
                            "experiment_id": args.experiment_id,
                            "elapsed_s": round(elapsed, 3),
                            "dt_s": round(dt, 3),

                            "state_read_ok": state_read_ok,
                            "mode": runtime_state.get("mode", args.mode),
                            "traffic_profile": runtime_state.get("traffic_profile", args.traffic_profile),
                            "active_policy": runtime_state.get("active_policy", args.active_policy),
                            "policy_id": runtime_state.get("policy_id", "unknown"),
                            "candidate_path_id": runtime_state.get("candidate_path_id", "unknown"),
                            "policy_seq": runtime_state.get("policy_seq", 0),
                            "decision_id": runtime_state.get("decision_id", "none"),
                            "reroute_event": runtime_state.get("reroute_event", 0),
                            "cooldown_active": runtime_state.get("cooldown_active", 0),
                            "event": runtime_state.get("event", "none"),
                            "event_ts": runtime_state.get("event_ts", ""),
                            "cooldown_until": runtime_state.get("cooldown_until", ""),
                            "elephant_path_hint": runtime_state.get("elephant_path_hint", "unknown"),
                            "elephant_flow_id": runtime_state.get("elephant_flow_id", "unknown"),
                            "elephant_topk_rank": runtime_state.get("elephant_topk_rank", ""),
                            "ingress_node": runtime_state.get("ingress_node", "unknown"),
                            "segment_list": runtime_state.get("segment_list", ""),
                            "controller_epoch_s": runtime_state.get("controller_epoch_s", ""),
                            "gain_mlu_pct": runtime_state.get("gain_mlu_pct", ""),
                            "gain_rtt_ms": runtime_state.get("gain_rtt_ms", ""),

                            "ping_src": args.ping_src,
                            "ping_dst": args.ping_dst,
                            "ping_cmd_ok": "",
                            "ping_ok": "",
                            "ping_sent": "",
                            "ping_received": "",
                            "ping_reply_count": "",
                            "ping_loss_pct": "",
                            "ping_rtts_ms": "",
                            "ping_rtt_min_ms": "",
                            "ping_rtt_avg_ms": "",
                            "ping_rtt_max_ms": "",
                            "ping_rtt_mdev_ms": "",
                            "ping_rtt_p50_ms": "",
                            "ping_rtt_p95_ms": "",
                            "ping_rtt_p99_ms": "",

                            "router": container,
                            "iface": iface,
                            "counter_cmd_ok": cmd_ok,
                            "counter_read_ok": counter_read_ok,
                            "counter_reset": counter_reset,
                            "qdisc_cmd_ok": qdisc_cmd_ok,
                            "qdisc_read_ok": qdisc_read_ok_val,
                            "link_capacity_mbps": link_capacity_mbps,
                            "missing_capacity": missing_capacity,

                            "rx_bytes": cur["rx_bytes"],
                            "tx_bytes": cur["tx_bytes"],
                            "rx_packets": cur["rx_packets"],
                            "tx_packets": cur["tx_packets"],
                            "rx_dropped": cur["rx_dropped"],
                            "tx_dropped": cur["tx_dropped"],
                            "rx_errors": cur["rx_errors"],
                            "tx_errors": cur["tx_errors"],

                            "rx_delta_bytes": rx_delta,
                            "tx_delta_bytes": tx_delta,
                            "rx_delta_packets": rx_pkt_delta,
                            "tx_delta_packets": tx_pkt_delta,
                            "rx_drop_delta": rx_drop_delta,
                            "tx_drop_delta": tx_drop_delta,
                            "rx_err_delta": rx_err_delta,
                            "tx_err_delta": tx_err_delta,

                            "rx_mbps": round(rx_mbps, 3),
                            "tx_mbps": round(tx_mbps, 3),
                            "rx_pps": round(rx_pps, 3),
                            "tx_pps": round(tx_pps, 3),

                            "util_pct": util_pct,
                            "dominant_egress_iface": "",
                            "qdisc_kind": blank_if_none(q_cur.get("qdisc_kind") if qdisc_read_ok_val == 1 else q_prev.get("qdisc_kind")),
                            "qdisc_backlog_bytes": blank_if_none(q_cur.get("qdisc_backlog_bytes") if qdisc_read_ok_val == 1 else q_prev.get("qdisc_backlog_bytes")),
                            "qdisc_backlog_pkts": blank_if_none(q_cur.get("qdisc_backlog_pkts") if qdisc_read_ok_val == 1 else q_prev.get("qdisc_backlog_pkts")),
                            "qdisc_drops": blank_if_none(qdisc_drops),
                            "qdisc_requeues": blank_if_none(qdisc_requeues),
                            "qdisc_overlimits": blank_if_none(qdisc_overlimits),
                            "qdisc_drop_delta": blank_if_none(qdisc_drop_delta),
                            "qdisc_requeue_delta": blank_if_none(qdisc_requeue_delta),
                            "qdisc_overlimit_delta": blank_if_none(qdisc_overlimit_delta),

                            "network_mlu_pct": "",
                            "util_threshold_pct": args.util_threshold_pct,
                            "is_hot_link": is_hot_link,

                            "_util_pct_num": util_pct_num,
                        }
                        interval_rows.append(row)

                    dominant_iface = pick_dominant_egress_iface(
                        ifaces,
                        tx_map,
                        idle_threshold_mbps=args.dominant_idle_threshold_mbps,
                    )
                    dominant_iface_by_router[container] = dominant_iface
                    for row in interval_rows[-len(ifaces):]:
                        row["dominant_egress_iface"] = dominant_iface

                ping_stats = get_ping_stats(
                    args.ping_src,
                    args.ping_dst,
                    count=args.ping_count,
                    timeout=args.ping_timeout,
                    interval=args.ping_interval,
                    cmd_timeout=args.cmd_timeout,
                )

                util_values = [row["_util_pct_num"] for row in interval_rows if isinstance(row["_util_pct_num"], (int, float))]
                network_mlu_pct = round(max(util_values), 3) if util_values else ""

                hot_links = [
                    f"{row['router']}:{row['iface']}={row['util_pct']}%"
                    for row in interval_rows
                    if row["is_hot_link"] == 1
                ]
                hot_links_str = "|".join(hot_links) if hot_links else ""

                wide_row: Dict[str, Any] = {
                    "timestamp": ts,
                    "wall_time_epoch_ms": wall_time_epoch_ms,
                    "sample_id": sample_id,
                    "experiment_id": args.experiment_id,
                    "elapsed_s": round(elapsed, 3),
                    "dt_s": round(dt, 3),
                    "state_read_ok": state_read_ok,
                    "mode": runtime_state.get("mode", args.mode),
                    "traffic_profile": runtime_state.get("traffic_profile", args.traffic_profile),
                    "active_policy": runtime_state.get("active_policy", args.active_policy),
                    "policy_id": runtime_state.get("policy_id", "unknown"),
                    "candidate_path_id": runtime_state.get("candidate_path_id", "unknown"),
                    "policy_seq": runtime_state.get("policy_seq", 0),
                    "decision_id": runtime_state.get("decision_id", "none"),
                    "reroute_event": runtime_state.get("reroute_event", 0),
                    "cooldown_active": runtime_state.get("cooldown_active", 0),
                    "event": runtime_state.get("event", "none"),
                    "event_ts": runtime_state.get("event_ts", ""),
                    "cooldown_until": runtime_state.get("cooldown_until", ""),
                    "elephant_path_hint": runtime_state.get("elephant_path_hint", "unknown"),
                    "elephant_flow_id": runtime_state.get("elephant_flow_id", "unknown"),
                    "elephant_topk_rank": runtime_state.get("elephant_topk_rank", ""),
                    "ingress_node": runtime_state.get("ingress_node", "unknown"),
                    "segment_list": runtime_state.get("segment_list", ""),
                    "controller_epoch_s": runtime_state.get("controller_epoch_s", ""),
                    "gain_mlu_pct": runtime_state.get("gain_mlu_pct", ""),
                    "gain_rtt_ms": runtime_state.get("gain_rtt_ms", ""),
                    "ping_src": args.ping_src,
                    "ping_dst": args.ping_dst,
                    "ping_cmd_ok": blank_if_none(ping_stats.get("ping_cmd_ok")),
                    "ping_ok": blank_if_none(ping_stats.get("ping_ok")),
                    "ping_sent": blank_if_none(ping_stats.get("ping_sent")),
                    "ping_received": blank_if_none(ping_stats.get("ping_received")),
                    "ping_reply_count": blank_if_none(ping_stats.get("ping_reply_count")),
                    "ping_loss_pct": blank_if_none(ping_stats.get("ping_loss_pct")),
                    "ping_rtts_ms": blank_if_none(ping_stats.get("ping_rtts_ms")),
                    "ping_rtt_min_ms": blank_if_none(ping_stats.get("ping_rtt_min_ms")),
                    "ping_rtt_avg_ms": blank_if_none(ping_stats.get("ping_rtt_avg_ms")),
                    "ping_rtt_max_ms": blank_if_none(ping_stats.get("ping_rtt_max_ms")),
                    "ping_rtt_mdev_ms": blank_if_none(ping_stats.get("ping_rtt_mdev_ms")),
                    "ping_rtt_p50_ms": blank_if_none(ping_stats.get("ping_rtt_p50_ms")),
                    "ping_rtt_p95_ms": blank_if_none(ping_stats.get("ping_rtt_p95_ms")),
                    "ping_rtt_p99_ms": blank_if_none(ping_stats.get("ping_rtt_p99_ms")),
                    "network_mlu_pct": network_mlu_pct,
                    "hot_links": hot_links_str,
                }

                for container, dominant_iface in dominant_iface_by_router.items():
                    wide_row[f"{sanitize_key(container)}__dominant_egress_iface"] = dominant_iface

                for row in interval_rows:
                    row["network_mlu_pct"] = network_mlu_pct
                    row["ping_cmd_ok"] = blank_if_none(ping_stats.get("ping_cmd_ok"))
                    row["ping_ok"] = blank_if_none(ping_stats.get("ping_ok"))
                    row["ping_sent"] = blank_if_none(ping_stats.get("ping_sent"))
                    row["ping_received"] = blank_if_none(ping_stats.get("ping_received"))
                    row["ping_reply_count"] = blank_if_none(ping_stats.get("ping_reply_count"))
                    row["ping_loss_pct"] = blank_if_none(ping_stats.get("ping_loss_pct"))
                    row["ping_rtts_ms"] = blank_if_none(ping_stats.get("ping_rtts_ms"))
                    row["ping_rtt_min_ms"] = blank_if_none(ping_stats.get("ping_rtt_min_ms"))
                    row["ping_rtt_avg_ms"] = blank_if_none(ping_stats.get("ping_rtt_avg_ms"))
                    row["ping_rtt_max_ms"] = blank_if_none(ping_stats.get("ping_rtt_max_ms"))
                    row["ping_rtt_mdev_ms"] = blank_if_none(ping_stats.get("ping_rtt_mdev_ms"))
                    row["ping_rtt_p50_ms"] = blank_if_none(ping_stats.get("ping_rtt_p50_ms"))
                    row["ping_rtt_p95_ms"] = blank_if_none(ping_stats.get("ping_rtt_p95_ms"))
                    row["ping_rtt_p99_ms"] = blank_if_none(ping_stats.get("ping_rtt_p99_ms"))
                    row["util_threshold_pct"] = args.util_threshold_pct

                    prefix = f"{sanitize_key(row['router'])}__{sanitize_key(row['iface'])}"
                    for suffix in per_iface_wide_suffixes:
                        wide_row[f"{prefix}__{suffix}"] = row[suffix]

                    row.pop("_util_pct_num", None)
                    long_writer.writerow(row)

                wide_writer.writerow(wide_row)
                long_f.flush()
                wide_f.flush()

                rtt_avg = ping_stats.get("ping_rtt_avg_ms")
                rtt_p95 = ping_stats.get("ping_rtt_p95_ms")
                loss_pct = ping_stats.get("ping_loss_pct")
                rtt_avg_str = "NA" if rtt_avg is None else f"{rtt_avg:.3f}"
                rtt_p95_str = "NA" if rtt_p95 is None else f"{rtt_p95:.3f}"
                loss_pct_str = "NA" if loss_pct is None else f"{loss_pct:.1f}"
                hot_links_show = hot_links_str if hot_links_str else "none"

                print(
                    f"[{ts}] "
                    f"sample={sample_id} "
                    f"mode={runtime_state.get('mode', args.mode)} "
                    f"policy={runtime_state.get('active_policy', args.active_policy)} "
                    f"policy_id={runtime_state.get('policy_id', 'unknown')} "
                    f"path={runtime_state.get('candidate_path_id', 'unknown')} "
                    f"MLU={network_mlu_pct if network_mlu_pct != '' else 'NA'}% "
                    f"RTT(avg)={rtt_avg_str} ms "
                    f"RTT(p95)={rtt_p95_str} ms "
                    f"loss={loss_pct_str}% "
                    f"hot_links={hot_links_show}"
                )

                prev_stats = current_stats
                prev_qdisc = current_qdisc
                prev_sample_time = loop_start
                sample_id += 1

                sleep_time = args.interval - (time.monotonic() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        print("Done.")


if __name__ == "__main__":
    main()