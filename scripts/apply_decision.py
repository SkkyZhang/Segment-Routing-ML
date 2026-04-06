#!/usr/bin/env python3
"""Apply a chosen candidate path to the running lab using FRR static routes.

This script translates a chosen candidate corridor into hop-by-hop FRR static
routes for the destination prefix. It is a practical control-plane realization
for this lab: instead of pushing native SR policies, it installs deterministic
static routes along the selected path inside the FRR routers.

Forward direction:
  192.168.8.0/24  (h1 -> h2)

Optional reverse direction:
  192.168.1.0/24  (h2 -> h1)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import srte_decider


IP_RE = re.compile(r"ip address\s+(\S+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply chosen SR-TE decision as FRR static routes.")
    parser.add_argument("--decision-json", default="", help="Decision JSON containing chosen_candidate_path_id")
    parser.add_argument("--path-id", default="", help="Apply this candidate path directly without a decision JSON")
    parser.add_argument("--paths-json", default="candidate_paths_example.json", help="Candidate paths JSON")
    parser.add_argument("--lab", default="lab.clab.yml", help="Path to lab.clab.yml")
    parser.add_argument("--topology-name", default="", help="Override topology name (default: parsed from lab.clab.yml)")
    parser.add_argument("--dest-prefix", default="192.168.8.0/24", help="Forward destination prefix")
    parser.add_argument("--return-prefix", default="192.168.1.0/24", help="Reverse destination prefix")
    parser.add_argument("--no-reverse", action="store_true", help="Do not install reverse-direction routes")
    parser.add_argument("--docker-cmd", default=os.environ.get("DOCKER_CMD", "sudo docker"), help="Docker invocation, e.g. 'sudo docker'")
    parser.add_argument("--dry-run", action="store_true", help="Print the FRR commands without executing them")
    parser.add_argument("--state-file", default="data/controller_state.json", help="Optional controller state file to update after apply")
    parser.add_argument("--out-json", default="", help="Optional apply report JSON")
    return parser.parse_args()


def parse_topology_name(lab_path: str) -> str:
    text = Path(lab_path).read_text(encoding="utf-8")
    match = re.search(r"^\s*name:\s*([A-Za-z0-9_-]+)\s*$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not parse topology name from {lab_path}")
    return match.group(1)


def parse_router_iface_ips(router_name: str) -> Dict[str, str]:
    conf_path = ROOT / router_name / "frr.conf"
    if not conf_path.exists():
        raise FileNotFoundError(f"Missing FRR config: {conf_path}")
    iface_ips: Dict[str, str] = {}
    current_iface = ""
    for raw in conf_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("interface "):
            current_iface = line.split(None, 1)[1].strip()
            continue
        if current_iface:
            match = IP_RE.search(line)
            if match:
                iface_ips[current_iface] = match.group(1).split("/", 1)[0]
    return iface_ips


def load_all_router_iface_ips(
    paths_map: Dict[str, Dict[str, Any]],
    peer_map: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    routers: Set[str] = set()
    for spec in paths_map.values():
        for iface_spec in spec.get("interfaces", []):
            norm_iface = srte_decider.normalize_iface_spec(str(iface_spec))
            routers.add(srte_decider.normalize_node_name(norm_iface.split(":", 1)[0]))
            peer = peer_map.get(norm_iface)
            if peer:
                routers.add(srte_decider.normalize_node_name(peer.split(":", 1)[0]))
    return {router: parse_router_iface_ips(router) for router in sorted(routers)}


def decision_path_id(args: argparse.Namespace) -> str:
    if args.path_id:
        return args.path_id
    if not args.decision_json:
        raise ValueError("Provide either --path-id or --decision-json")
    data = json.loads(Path(args.decision_json).read_text(encoding="utf-8"))
    path_id = str(data.get("chosen_candidate_path_id", "")).strip()
    if not path_id:
        raise ValueError(f"No chosen_candidate_path_id found in {args.decision_json}")
    return path_id


def iface_router(iface_spec: str) -> str:
    return srte_decider.normalize_node_name(str(iface_spec).split(":", 1)[0])


def build_path_route_plan(
    path_ifaces: List[str],
    peer_map: Dict[str, str],
    iface_ips: Dict[str, Dict[str, str]],
    prefix: str,
) -> List[Tuple[str, str]]:
    plan: List[Tuple[str, str]] = []
    for iface_spec in path_ifaces:
        norm_iface = srte_decider.normalize_iface_spec(iface_spec)
        peer = peer_map.get(norm_iface)
        if not peer:
            raise ValueError(f"No peer found in lab for {iface_spec}")
        router = iface_router(norm_iface)
        peer_router, peer_iface = peer.split(":", 1)
        next_hop = iface_ips.get(peer_router, {}).get(peer_iface)
        if not next_hop:
            raise ValueError(f"No next-hop IP found for peer {peer_router}:{peer_iface}")
        plan.append((router, next_hop))
    return plan


def build_reverse_ifaces(path_ifaces: List[str], peer_map: Dict[str, str]) -> List[str]:
    reverse: List[str] = []
    for iface_spec in reversed(path_ifaces):
        norm_iface = srte_decider.normalize_iface_spec(iface_spec)
        peer = peer_map.get(norm_iface)
        if not peer:
            raise ValueError(f"No peer found in lab for {iface_spec}")
        reverse.append(peer)
    return reverse


def build_cleanup_candidates(
    paths_map: Dict[str, Dict[str, Any]],
    peer_map: Dict[str, str],
    iface_ips: Dict[str, Dict[str, str]],
    prefix: str,
) -> Dict[str, Set[str]]:
    cleanup: Dict[str, Set[str]] = {}
    for spec in paths_map.values():
        for router, next_hop in build_path_route_plan(spec.get("interfaces", []), peer_map, iface_ips, prefix):
            cleanup.setdefault(router, set()).add(next_hop)
    return cleanup


def docker_exec_prefix(docker_cmd: str) -> List[str]:
    parts = shlex.split(docker_cmd)
    if not parts:
        raise ValueError("docker command is empty")
    return parts + ["exec"]


def run_router_config(
    docker_cmd: str,
    container: str,
    config_lines: List[str],
    dry_run: bool,
) -> Dict[str, Any]:
    base = docker_exec_prefix(docker_cmd) + [container, "vtysh", "-c", "configure terminal"]
    for line in config_lines:
        base.extend(["-c", line])
    if dry_run:
        return {"container": container, "cmd": base, "executed": False}
    result = subprocess.run(base, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed configuring {container}: rc={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return {"container": container, "cmd": base, "executed": True, "stdout": result.stdout.strip()}


def update_state_file(state_path: Path, path_id: str) -> None:
    if not state_path.exists():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return
    state["candidate_path_id"] = path_id
    state["active_policy"] = "srte_ml_dynamic" if str(state.get("mode", "")).strip() == "ml_dynamic" else state.get("active_policy", "")
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    topology_name = args.topology_name or parse_topology_name(args.lab)
    peer_map = srte_decider.parse_lab_peer_map(args.lab)
    paths_map = srte_decider.load_paths_map(args.paths_json, peer_map=peer_map)
    iface_ips = load_all_router_iface_ips(paths_map, peer_map)
    chosen_path = decision_path_id(args)
    if chosen_path not in paths_map:
        raise SystemExit(f"Unknown candidate path: {chosen_path}")

    forward_ifaces = paths_map[chosen_path]["interfaces"]
    reverse_ifaces = build_reverse_ifaces(forward_ifaces, peer_map)

    forward_plan = build_path_route_plan(forward_ifaces, peer_map, iface_ips, args.dest_prefix)
    reverse_plan = build_path_route_plan(reverse_ifaces, peer_map, iface_ips, args.return_prefix)

    cleanup_forward = build_cleanup_candidates(paths_map, peer_map, iface_ips, args.dest_prefix)
    cleanup_reverse = build_cleanup_candidates(
        {cid: {"interfaces": build_reverse_ifaces(spec["interfaces"], peer_map)} for cid, spec in paths_map.items()},
        peer_map,
        iface_ips,
        args.return_prefix,
    )

    per_router_lines: Dict[str, List[str]] = {}
    for router, next_hops in cleanup_forward.items():
        for nh in sorted(next_hops):
            per_router_lines.setdefault(router, []).append(f"no ip route {args.dest_prefix} {nh}")
    if not args.no_reverse:
        for router, next_hops in cleanup_reverse.items():
            for nh in sorted(next_hops):
                per_router_lines.setdefault(router, []).append(f"no ip route {args.return_prefix} {nh}")

    for router, next_hop in forward_plan:
        per_router_lines.setdefault(router, []).append(f"ip route {args.dest_prefix} {next_hop}")
    if not args.no_reverse:
        for router, next_hop in reverse_plan:
            per_router_lines.setdefault(router, []).append(f"ip route {args.return_prefix} {next_hop}")

    apply_log: List[Dict[str, Any]] = []
    for router in sorted(per_router_lines):
        container = f"clab-{topology_name}-{router}"
        apply_log.append(
            run_router_config(
                docker_cmd=args.docker_cmd,
                container=container,
                config_lines=per_router_lines[router],
                dry_run=args.dry_run,
            )
        )

    if not args.dry_run:
        update_state_file(Path(args.state_file), chosen_path)

    report = {
        "topology_name": topology_name,
        "chosen_candidate_path_id": chosen_path,
        "forward_prefix": args.dest_prefix,
        "return_prefix": None if args.no_reverse else args.return_prefix,
        "forward_plan": [{"router": r, "next_hop": nh} for r, nh in forward_plan],
        "reverse_plan": [] if args.no_reverse else [{"router": r, "next_hop": nh} for r, nh in reverse_plan],
        "dry_run": args.dry_run,
        "apply_log": apply_log,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[INFO] Wrote apply report to {out_path}")


if __name__ == "__main__":
    main()
