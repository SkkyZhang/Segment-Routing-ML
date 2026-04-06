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

ROUTERS=(r1 r2 r3 r4 r5 r6 r7 r8)
HOSTS=(h1 h2)

declare -A EXPECTED_NEIGHBORS=(
  [r1]=2
  [r2]=2
  [r3]=3
  [r4]=2
  [r5]=2
  [r6]=3
  [r7]=2
  [r8]=2
)

declare -A LINK_CHECKS=(
  ["r1|10.0.12.2"]="r1->r2"
  ["r1|10.0.15.2"]="r1->r5"
  ["r2|10.0.12.1"]="r2->r1"
  ["r2|10.0.23.2"]="r2->r3"
  ["r3|10.0.23.1"]="r3->r2"
  ["r3|10.0.34.2"]="r3->r4"
  ["r3|10.0.36.2"]="r3->r6"
  ["r4|10.0.34.1"]="r4->r3"
  ["r4|10.0.48.2"]="r4->r8"
  ["r5|10.0.15.1"]="r5->r1"
  ["r5|10.0.56.2"]="r5->r6"
  ["r6|10.0.56.1"]="r6->r5"
  ["r6|10.0.67.2"]="r6->r7"
  ["r6|10.0.36.1"]="r6->r3"
  ["r7|10.0.67.1"]="r7->r6"
  ["r7|10.0.78.2"]="r7->r8"
  ["r8|10.0.48.1"]="r8->r4"
  ["r8|10.0.78.1"]="r8->r7"
  ["h1|192.168.1.1"]="h1->gw"
  ["h2|192.168.8.1"]="h2->gw"
  ["h1|192.168.8.2"]="h1->h2"
  ["h2|192.168.1.2"]="h2->h1"
)

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

require_bin() {
  command -v "$1" >/dev/null 2>&1 || {
    printf '[ERROR] Missing command: %s\n' "$1" >&2
    exit 2
  }
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
    return 0
  fi

  printf ''
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
    printf '[ERROR] Active containers:\n'
    printf '%s\n' "$running_names"
    printf '[ERROR] If you still use the old topology, run with: TOPO_NAME=<name> bash scripts/check_connectivity.sh\n'
    exit 2
  fi

  printf '[INFO] Using topology name: %s\n' "$TOPO_NAME"
}

container_exists() {
  local name="$1"
  docker_cmd inspect "$name" >/dev/null 2>&1
}

container_running() {
  local name="$1"
  local status
  status="$(docker_cmd inspect -f '{{.State.Status}}' "$name" 2>/dev/null || true)"
  [[ "$status" == "running" ]]
}

check_containers() {
  info "Checking expected containers"

  local node cname
  for node in "${ROUTERS[@]}" "${HOSTS[@]}"; do
    cname="$(container_name "$node")"
    if ! container_exists "$cname"; then
      fail "$cname does not exist"
      continue
    fi
    if ! container_running "$cname"; then
      fail "$cname is not running"
      continue
    fi
    pass "$cname is running"
  done
}

check_ospf_neighbors() {
  info "Checking OSPF adjacencies"

  local router cname expected actual output
  for router in "${ROUTERS[@]}"; do
    cname="$(container_name "$router")"
    if ! container_running "$cname"; then
      fail "$cname skipped OSPF check because it is not running"
      continue
    fi

    output="$(docker_cmd exec "$cname" vtysh -c 'show ip ospf neighbor' 2>/dev/null || true)"
    actual="$(printf '%s\n' "$output" | grep -c 'Full/' || true)"
    expected="${EXPECTED_NEIGHBORS[$router]}"

    if [[ "$actual" == "$expected" ]]; then
      pass "$router has $actual/$expected Full OSPF neighbors"
    else
      fail "$router has $actual/$expected Full OSPF neighbors"
      if [[ -n "$output" ]]; then
        printf '%s\n' "$output"
      fi
    fi
  done
}

check_ping() {
  local node="$1"
  local target_ip="$2"
  local label="$3"
  local cname

  cname="$(container_name "$node")"
  if ! container_running "$cname"; then
    fail "$label skipped because $cname is not running"
    return
  fi

  if docker_cmd exec "$cname" ping -n -c "$PING_COUNT" -W "$PING_TIMEOUT" "$target_ip" >/dev/null 2>&1; then
    pass "$label ping to $target_ip"
  else
    fail "$label ping to $target_ip"
  fi
}

check_link_pings() {
  info "Checking link and end-to-end reachability"

  local key node target_ip label
  for key in "${!LINK_CHECKS[@]}"; do
    node="${key%%|*}"
    target_ip="${key##*|}"
    label="${LINK_CHECKS[$key]}"
    check_ping "$node" "$target_ip" "$label"
  done
}

print_route_snapshots() {
  info "Capturing key route snapshots"

  local cname
  cname="$(container_name h1)"
  if container_running "$cname"; then
    printf '[INFO] h1 route to 192.168.8.2\n'
    docker_cmd exec "$cname" ip route get 192.168.8.2 2>/dev/null || true
  fi

  cname="$(container_name h2)"
  if container_running "$cname"; then
    printf '[INFO] h2 route to 192.168.1.2\n'
    docker_cmd exec "$cname" ip route get 192.168.1.2 2>/dev/null || true
  fi
}

main() {
  require_bin bash

  preflight_topology
  check_containers
  check_ospf_neighbors
  check_link_pings
  print_route_snapshots

  printf '\n[SUMMARY] pass=%d fail=%d\n' "$pass_count" "$fail_count"
  if (( fail_count > 0 )); then
    exit 1
  fi
}

main "$@"
