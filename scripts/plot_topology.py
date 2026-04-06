#!/usr/bin/env python3
"""Draw the SR-TE topology and candidate paths."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


PATH_COLORS = {
    "upper_corridor": "#d94841",
    "lower_corridor": "#2b8a3e",
    "cross_path": "#1c7ed6",
}

NODE_POSITIONS = {
    "h1": (0.0, 1.0),
    "r1": (1.2, 1.0),
    "r2": (3.0, 2.2),
    "r3": (5.0, 2.2),
    "r4": (7.0, 2.2),
    "r8": (8.8, 1.0),
    "r5": (3.0, -0.2),
    "r6": (5.0, -0.2),
    "r7": (7.0, -0.2),
    "h2": (10.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw SR-TE topology and candidate paths")
    parser.add_argument("--lab", default="lab.clab.yml", help="Path to lab.clab.yml")
    parser.add_argument("--paths-json", default="candidate_paths_example.json", help="Candidate paths JSON")
    parser.add_argument("--decision-json", default="", help="Optional decision JSON to highlight chosen path")
    parser.add_argument("--output", default="data/topology_paths.png", help="Output PNG path")
    parser.add_argument(
        "--highlight-path",
        default="",
        help="Optional candidate path id to highlight even without decision JSON",
    )
    return parser.parse_args()


def sanitize_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())


def normalize_iface_spec(iface_spec: str) -> str:
    node, iface = str(iface_spec).split(":", 1)
    node = re.sub(r"^clab-[^-]+-", "", node)
    return f"{node}:{iface}"


def router_iface_to_link_key(router_iface: str) -> str:
    router, iface = router_iface.split(":", 1)
    return f"{sanitize_key(router)}__{sanitize_key(iface)}"


def load_json(path: str) -> Dict[str, object]:
    p = Path(path)
    if not path or not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def parse_lab_links(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8")
    mapping: Dict[str, str] = {}
    pattern = re.compile(r'endpoints:\s*\["([^"]+)",\s*"([^"]+)"\]')
    for left, right in pattern.findall(text):
        mapping[left] = right
        mapping[right] = left
    return mapping


def sorted_edge(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def candidate_path_edges(candidate_spec: Dict[str, object], peer_map: Dict[str, str]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for iface_spec in candidate_spec.get("interfaces", []):
        normalized = normalize_iface_spec(str(iface_spec))
        peer = peer_map.get(normalized)
        if not peer:
            continue
        left_node = normalized.split(":", 1)[0]
        right_node = str(peer).split(":", 1)[0]
        edges.append(sorted_edge(left_node, right_node))
    return edges


def all_topology_edges(peer_map: Dict[str, str]) -> List[Tuple[str, str]]:
    seen = set()
    edges: List[Tuple[str, str]] = []
    for left, right in peer_map.items():
        left_node = left.split(":", 1)[0]
        right_node = right.split(":", 1)[0]
        edge = sorted_edge(left_node, right_node)
        if edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
    return edges


def draw_edge(ax, edge: Tuple[str, str], color: str, linewidth: float, alpha: float, zorder: int) -> None:
    x1, y1 = NODE_POSITIONS[edge[0]]
    x2, y2 = NODE_POSITIONS[edge[1]]
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha, solid_capstyle="round", zorder=zorder)


def draw_arrow_edge(ax, edge: Tuple[str, str], color: str, linewidth: float, zorder: int) -> None:
    x1, y1 = NODE_POSITIONS[edge[0]]
    x2, y2 = NODE_POSITIONS[edge[1]]
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=linewidth,
        color=color,
        alpha=1.0,
        shrinkA=18,
        shrinkB=18,
        zorder=zorder,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(arrow)


def draw_nodes(ax) -> None:
    for node, (x, y) in NODE_POSITIONS.items():
        is_host = node.startswith("h")
        face = "#f8f9fa" if is_host else "#fff3bf"
        edge = "#495057" if is_host else "#5f3dc4"
        ax.scatter([x], [y], s=900 if is_host else 1100, c=face, edgecolors=edge, linewidths=2.2, zorder=5)
        ax.text(x, y, node, ha="center", va="center", fontsize=12, weight="bold", zorder=6)


def draw_topology(
    topology_edges: Sequence[Tuple[str, str]],
    path_edges_map: Dict[str, List[Tuple[str, str]]],
    chosen_path: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("#fffdf7")
    ax.set_facecolor("#fffdf7")

    for edge in topology_edges:
        draw_edge(ax, edge, color="#ced4da", linewidth=3.0, alpha=1.0, zorder=1)

    if chosen_path and chosen_path in path_edges_map:
        color = PATH_COLORS.get(chosen_path, "#212529")
        for edge in path_edges_map[chosen_path]:
            draw_edge(ax, edge, color="#fffdf7", linewidth=15.0, alpha=1.0, zorder=3)
            draw_edge(ax, edge, color=color, linewidth=9.5, alpha=1.0, zorder=4)
            draw_arrow_edge(ax, edge, color=color, linewidth=4.2, zorder=5)
    elif not chosen_path:
        for path_id, edges in path_edges_map.items():
            color = PATH_COLORS.get(path_id, "#343a40")
            for edge in edges:
                draw_edge(ax, edge, color=color, linewidth=5.0, alpha=0.45, zorder=2)

    draw_nodes(ax)

    ax.text(5.0, 3.05, "Upper Corridor", ha="center", va="center", fontsize=12, color=PATH_COLORS["upper_corridor"])
    ax.text(5.0, -1.0, "Lower Corridor", ha="center", va="center", fontsize=12, color=PATH_COLORS["lower_corridor"])
    ax.text(5.0, 1.1, "Cross Link r3-r6", ha="center", va="center", fontsize=11, color=PATH_COLORS["cross_path"])

    legend_items = [
        Line2D([0], [0], color="#ced4da", lw=4, label="Physical Topology"),
        Line2D([0], [0], color=PATH_COLORS["upper_corridor"], lw=6, label="upper_corridor"),
        Line2D([0], [0], color=PATH_COLORS["lower_corridor"], lw=6, label="lower_corridor"),
        Line2D([0], [0], color=PATH_COLORS["cross_path"], lw=6, label="cross_path"),
    ]
    if chosen_path:
        legend_items.append(
            Line2D([0], [0], color=PATH_COLORS.get(chosen_path, "#212529"), lw=10, label=f"chosen: {chosen_path}")
        )
    ax.legend(handles=legend_items, loc="upper center", ncol=5 if chosen_path else 4, frameon=False, fontsize=10)

    title = "SR-TE 8-node topology and candidate paths"
    if chosen_path:
        title += f" | selected path: {chosen_path}"
        ax.text(
            5.0,
            3.28,
            f"SELECTED PATH: {chosen_path}",
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            color=PATH_COLORS.get(chosen_path, "#212529"),
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff3bf", "edgecolor": PATH_COLORS.get(chosen_path, "#212529"), "linewidth": 2},
            zorder=10,
        )
    ax.set_title(title, fontsize=15, weight="bold", pad=18)
    ax.set_xlim(-0.8, 10.8)
    ax.set_ylim(-1.5, 3.5)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    lab_path = Path(args.lab)
    paths_path = Path(args.paths_json)
    output_path = Path(args.output)

    peer_map = parse_lab_links(lab_path)
    paths_map = load_json(str(paths_path))
    if not isinstance(paths_map, dict) or not paths_map:
        raise SystemExit(f"Invalid paths JSON: {paths_path}")

    path_edges_map = {
        str(path_id): candidate_path_edges(spec, peer_map)
        for path_id, spec in paths_map.items()
    }
    chosen_path = ""
    if args.decision_json:
        decision_path = Path(args.decision_json)
        if not decision_path.exists():
            raise SystemExit(f"decision json not found: {decision_path}")
        decision = load_json(args.decision_json)
        chosen_path = str(decision.get("chosen_candidate_path_id", "")).strip()
        if not chosen_path:
            raise SystemExit(f"chosen_candidate_path_id not found in {decision_path}")
    if args.highlight_path:
        chosen_path = args.highlight_path.strip()

    draw_topology(
        topology_edges=all_topology_edges(peer_map),
        path_edges_map=path_edges_map,
        chosen_path=chosen_path,
        output_path=output_path,
    )
    print(f"[INFO] Wrote topology plot to {output_path}")


if __name__ == "__main__":
    main()
