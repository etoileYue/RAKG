import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def load_graph_data(json_path: Path) -> Tuple[List[dict], List[Tuple[str, str, str, str]]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entities = data.get("entities", [])
    raw_relations = data.get("relations", [])
    relations: List[Tuple[str, str, str, str]] = []

    for item in raw_relations:
        if isinstance(item, list) and len(item) >= 3:
            head = str(item[0]).strip()
            relation = str(item[1]).strip()
            tail = str(item[2]).strip()
            evidence = str(item[3]).strip() if len(item) > 3 else ""
            if head and relation and tail:
                relations.append((head, relation, tail, evidence))
        elif isinstance(item, dict):
            head = str(item.get("head", "")).strip() or str(item.get("source", "")).strip()
            relation = str(item.get("relation", "")).strip() or str(item.get("predicate", "")).strip()
            tail = str(item.get("tail", "")).strip() or str(item.get("target", "")).strip()
            evidence = str(item.get("description", "")).strip()
            if head and relation and tail:
                relations.append((head, relation, tail, evidence))

    return entities, relations


def build_nx_graph(entities: List[dict], relations: List[Tuple[str, str, str, str]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    name_to_type: Dict[str, str] = {}
    for entity in entities:
        name = str(entity.get("name", "")).strip()
        if not name:
            continue
        entity_type = str(entity.get("type", "Unknown")).strip() or "Unknown"
        name_to_type[name] = entity_type
        graph.add_node(name, entity_type=entity_type)

    seen = set()
    for head, relation, tail, evidence in relations:
        edge_key = (head, relation, tail)
        if edge_key in seen:
            continue
        seen.add(edge_key)

        if head not in graph:
            graph.add_node(head, entity_type="Unknown")
        if tail not in graph:
            graph.add_node(tail, entity_type="Unknown")

        graph.add_edge(head, tail, relation=relation, evidence=evidence)

    return graph


def draw_graph(
    graph: nx.MultiDiGraph,
    output_path: Path,
    show_edge_labels: bool = False,
    k: float = 0.45,
    scale: float = 1.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(18, 14))
    pos = nx.spring_layout(graph, k=k, scale=scale, iterations=500, seed=42)

    node_types = nx.get_node_attributes(graph, "entity_type")
    unique_types = sorted(set(node_types.values()))
    cmap = plt.get_cmap("tab20")
    type_to_color = {t: cmap(i % 20) for i, t in enumerate(unique_types)}
    node_colors = [type_to_color.get(node_types.get(n, "Unknown"), "#999999") for n in graph.nodes()]

    node_degrees = dict(graph.degree())
    node_sizes = [380 + 45 * node_degrees.get(n, 1) for n in graph.nodes()]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        linewidths=0.8,
        edgecolors="black",
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        alpha=0.35,
        arrows=True,
        arrowsize=11,
        width=0.9,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_labels(graph, pos, font_size=6.5)

    if show_edge_labels:
        edge_labels = {(u, v): d.get("relation", "") for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_size=6,
            alpha=0.75,
            rotate=False,
        )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=t, markerfacecolor=type_to_color[t], markersize=8)
        for t in unique_types
    ]
    if legend_handles:
        plt.legend(
            handles=legend_handles,
            title="Entity Type",
            loc="upper right",
            fontsize=8,
            title_fontsize=10,
            frameon=True,
        )

    plt.title("Knowledge Graph", fontsize=20, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RAKG graph JSON and export to PNG.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../../../data/short/processed/RAKG_graph_re/10.json"),
        help="Input RAKG graph json file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../../../data/short/processed/RAKG_graph_re/pic/10.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show-edge-labels",
        action="store_true",
        help="Whether to render relation text on edges.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.45,
        help="Spring layout k. Smaller usually means more compact layout.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Spring layout scale. Smaller means tighter overall spread.",
    )
    args = parser.parse_args()   
    
    for idx in range(1,11):
        filename = Path(f"../../../data/short/processed/RAKG_graph_re/{idx}.json")
        output = Path(f"../../../data/short/processed/RAKG_graph_re/pic_/{idx}.png")

        entities, relations = load_graph_data(filename)
        graph = build_nx_graph(entities, relations)
        draw_graph(
            graph,
            output,
            show_edge_labels=args.show_edge_labels,
            k=args.k,
            scale=args.scale,
        )

        print(f"Graph saved to: {output}")
        print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        print(f"Entities in file: {len(entities)}, Relations in file: {len(relations)}")


if __name__ == "__main__":
    main()
