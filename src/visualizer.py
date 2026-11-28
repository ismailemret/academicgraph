"""
Network visualization module using NetworkX and Matplotlib (Member 3).
Generates full network overviews and ego-centric subgraphs in PNG format.
"""

import os
import re
from typing import Any, Optional

import matplotlib.pyplot as plt
import networkx as nx


def _safe_filename(text: str) -> str:
    """Sanitize a node identifier into a filesystem-safe slug."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip().lower()).strip("_")
    return cleaned or "subgraph"


def _to_nx_graph(custom_graph: Any) -> nx.Graph:
    """Convert a custom undirected graph or adjacency dict into an nx.Graph for rendering."""
    g = nx.Graph()
    if hasattr(custom_graph, "adj"):
        for u, neighbors in custom_graph.adj.items():
            g.add_node(u)
            if isinstance(neighbors, dict):
                for v, w in neighbors.items():
                    g.add_edge(u, v, weight=w)
            else:
                for v in neighbors:
                    g.add_edge(u, v, weight=1.0)
    elif isinstance(custom_graph, dict):
        for u, neighbors in custom_graph.items():
            g.add_node(u)
            if isinstance(neighbors, dict):
                for v, w in neighbors.items():
                    g.add_edge(u, v, weight=w)
            else:
                for v in neighbors:
                    g.add_edge(u, v, weight=1.0)
    return g


class NetworkVisualizer:
    """Renders and exports network overview and ego-subgraph visualizations."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_overview(
        self,
        custom_graph: Any,
        title: str = "Genel Ağ Görünümü",
        filename: str = "overview_network.png",
        max_nodes: int = 150,
    ) -> str:
        """
        Render an overview plot of the graph with node sizes proportional to degree.
        Caps rendered nodes to the top max_nodes by degree to preserve readability.
        """
        g = _to_nx_graph(custom_graph)
        if g.number_of_nodes() == 0:
            raise ValueError("Çizilecek boş graf: düğüm bulunamadı.")

        if g.number_of_nodes() > max_nodes:
            top_nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            g = g.subgraph([n for n, _ in top_nodes]).copy()

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(g, k=0.45, iterations=40, seed=42)
        degrees = dict(g.degree())
        node_sizes = [max(80, degrees[n] * 60) for n in g.nodes()]

        nx.draw_networkx_edges(g, pos, alpha=0.25, edge_color="#888888")
        nx.draw_networkx_nodes(
            g, pos, node_size=node_sizes, node_color="#2b6cb0", alpha=0.85
        )

        # Label only high-degree hubs to prevent overlapping text clutter
        label_threshold = sorted(degrees.values(), reverse=True)[min(25, len(degrees) - 1)]
        labels = {n: str(n) for n, d in degrees.items() if d >= label_threshold}
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)

        plt.title(f"{title} (Düğüm: {g.number_of_nodes()}, Kenar: {g.number_of_edges()})")
        plt.axis("off")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def plot_ego_subgraph(
        self,
        custom_graph: Any,
        center_node: str,
        radius: int = 1,
        title: Optional[str] = None,
    ) -> str:
        """Render the k-hop neighborhood subgraph centered on a selected node."""
        g = _to_nx_graph(custom_graph)
        matched_node = next(
            (n for n in g.nodes() if str(n).strip().lower() == center_node.strip().lower()),
            None,
        )
        if matched_node is None:
            raise ValueError(f"'{center_node}' graf içinde bulunamadı.")

        sub = nx.ego_graph(g, matched_node, radius=radius)
        plt.figure(figsize=(11, 8))
        pos = nx.spring_layout(sub, k=0.6, iterations=50, seed=42)

        colors = ["#e53e3e" if n == matched_node else "#3182ce" for n in sub.nodes()]
        sizes = [550 if n == matched_node else 260 for n in sub.nodes()]

        nx.draw_networkx_edges(sub, pos, alpha=0.45, edge_color="#718096")
        nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=sizes, alpha=0.9)
        nx.draw_networkx_labels(sub, pos, font_size=8, font_weight="bold")

        edge_labels = {
            (u, v): f"{d['weight']:.2f}" if isinstance(d.get("weight"), float) else str(d.get("weight"))
            for u, v, d in sub.edges(data=True)
            if d.get("weight") is not None
        }
        if edge_labels and sub.number_of_edges() <= 40:
            nx.draw_networkx_edge_labels(sub, pos, edge_labels=edge_labels, font_size=7)

        plt.title(title or f"Yakın Çevre Alt Grafı: {matched_node}")
        plt.axis("off")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, f"subgraph_{_safe_filename(str(matched_node))}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path