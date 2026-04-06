#!/usr/bin/env bash
set -Eeuo pipefail

DOCKER_CMD="${DOCKER_CMD:-sudo docker}"
read -r -a DOCKER <<< "$DOCKER_CMD"

if [[ -n "${TOPO_NAME+x}" ]]; then
  TOPO_NAME_EXPLICIT=1
else
  TOPO_NAME_EXPLICIT=0
fi
TOPO_NAME="${TOPO_NAME:-srte8}"
PING_COUNT="${PING_COUNT:-1}"
PING_TIMEOUT="${PING_TIMEOUT:-1}"

pass_count=0
fail_count=0

container_name() {
  printf 'clab-%s-%s' "$TOPO_NAME" "$1"
}

pass() {
  pass_count=$((pass_count + 1))
  printf '[PASS] %s\n' "$1"
}

fail() {
  fail_count=$((fail_count + 1))
  printf '[FAIL] %s\n' "$1"
}

info() {
  printf '\n[INFO] %s\n' "$1"
}

docker_cmd() {
  "${DOCKER[@]}" "$@"
}

list_running_clab_containers() {
  docker_cmd ps --format '{{.Names}}' 2>/dev/null | grep '^clab-' || true
}

detect_topology_name() {
  local names topology_names unique_count detected

  names="$(list_running_clab_containers)"
  if [[ -z "$names" ]]; then
    printf ''
    return 0
  fi

  topology_names="$(
    printf '%s\n' "$names" |
      awk -F- 'NF >= 3 {print $2}' |
      sort -u
  )"

  unique_count="$(printf '%s\n' "$topology_names" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [[ "$unique_count" == "1" ]]; then
    detected="$(printf '%s\n' "$topology_names" | head -n 1)"
    printf '%s' "$detected"
  fi
}

preflight_topology() {
  local running_names detected

  info "Checking topology context"

  running_names="$(list_running_clab_containers)"
  if [[ -z "$running_names" ]]; then
    printf '[ERROR] No running containerlab containers were found.\n'
    printf '[ERROR] Deploy the lab first: sudo containerlab deploy -t lab.clab.yml\n'
    exit 2
  fi

  detected="$(detect_topology_name)"
  if [[ "$TOPO_NAME_EXPLICIT" != "1" && -n "$detected" && "$detected" != "$TOPO_NAME" ]]; then
    printf '[INFO] Auto-detected running topology: %s\n' "$detected"
    TOPO_NAME="$detected"
  fi

  if ! printf '%s\n' "$running_names" | grep -q "^clab-${TOPO_NAME}-"; then
    printf '[ERROR] Running containerlab containers exist, but none match topology "%s".\n' "$TOPO_NAME"
    printf '%s\n' "$running_names"
    exit 2
  fi

  printf '[INFO] Using topology name: %s\n' "$TOPO_NAME"
}

container_running() {
  local name="$1"
  local status
  status="$(docker_cmd inspect -f '{{.State.Status}}' "$name" 2>/dev/null || true)"
  [[ "$status" == "running" ]]
}

check_step_ping() {
  local node="$1"
  local target_ip="$2"
  local label="$3"
  local cname

  cname="$(container_name "$node")"
  if ! container_running "$cname"; then
    fail "$label skipped because $cname is not running"
    return 1
  fi

  if docker_cmd exec "$cname" ping -n -c "$PING_COUNT" -W "$PING_TIMEOUT" "$target_ip" >/dev/null 2>&1; then
    pass "$label"
    return 0
  fi

  fail "$label"
  return 1
}

check_host_route() {
  local host="$1"
  local dst="$2"
  local expect_via="$3"
  local cname out

  cname="$(container_name "$host")"
  if ! container_running "$cname"; then
    fail "$host route check skipped because $cname is not running"
    return
  fi

  out="$(docker_cmd exec "$cname" ip route get "$dst" 2>/dev/null || true)"
  if printf '%s\n' "$out" | grep -q "via ${expect_via}"; then
    pass "$host routes to $dst via $expect_via"
  else
    fail "$host does not route to $dst via $expect_via"
    printf '%s\n' "$out"
  fi
}

check_router_prefix() {
  local router="$1"
  local prefix="$2"
  local cname out

  cname="$(container_name "$router")"
  if ! container_running "$cname"; then
    fail "$router prefix check skipped because $cname is not running"
    return
  fi

  out="$(docker_cmd exec "$cname" vtysh -c "show ip route ${prefix}" 2>/dev/null || true)"
  if printf '%s\n' "$out" | grep -Eq 'Known via|Routing entry'; then
    pass "$router has route for $prefix"
  else
    fail "$router missing route for $prefix"
    printf '%s\n' "$out"
  fi
}

check_path_family() {
  local name="$1"
  shift

  info "Checking path family: ${name}"

  local step failure=0
  for step in "$@"; do
    IFS='|' read -r node target_ip label <<< "$step"
    if ! check_step_ping "$node" "$target_ip" "$label"; then
      failure=1
    fi
  done

  if (( failure == 0 )); then
    pass "${name} path family is fully reachable"
  else
    fail "${name} path family has failed steps"
  fi
}

main() {
  preflight_topology

  check_path_family "upper_corridor" \
    "r1|10.0.12.2|upper step r1->r2" \
    "r2|10.0.23.2|upper step r2->r3" \
    "r3|10.0.34.2|upper step r3->r4" \
    "r4|10.0.48.2|upper step r4->r8"

  check_path_family "lower_corridor" \
    "r1|10.0.15.2|lower step r1->r5" \
    "r5|10.0.56.2|lower step r5->r6" \
    "r6|10.0.67.2|lower step r6->r7" \
    "r7|10.0.78.2|lower step r7->r8"

  check_path_family "cross_path" \
    "r1|10.0.12.2|cross step r1->r2" \
    "r2|10.0.23.2|cross step r2->r3" \
    "r3|10.0.36.2|cross step r3->r6" \
    "r6|10.0.67.2|cross step r6->r7" \
    "r7|10.0.78.2|cross step r7->r8"

  info "Checking routing views"
  check_host_route h1 192.168.8.2 192.168.1.1
  check_host_route h2 192.168.1.2 192.168.8.1
  check_router_prefix r1 192.168.8.0/24
  check_router_prefix r8 192.168.1.0/24

  printf '\n[SUMMARY] pass=%d fail=%d\n' "$pass_count" "$fail_count"
  if (( fail_count > 0 )); then
    exit 1
  fi
}

main "$@"
