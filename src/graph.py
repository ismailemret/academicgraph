from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

from src.models import Academician, Field, University


class UndirectedGraph:
    """Custom adjacency-map undirected graph implementation with optional edge weights."""

    def __init__(self, name: str, weighted: bool = False) -> None:
        self.name: str = name
        self.weighted: bool = weighted
        self._adj: dict[str, dict[str, float]] = {}
        self.node_attrs: dict[str, dict[str, Any]] = {}

    def add_node(self, node_id: str, **attrs: Any) -> None:
        if node_id not in self._adj:
            self._adj[node_id] = {}
            self.node_attrs[node_id] = {}
        if attrs:
            self.node_attrs[node_id].update(attrs)

    def add_edge(
        self,
        u: str,
        v: str,
        weight: float = 1.0,
        increment: bool = False,
    ) -> None:
        if u == v:
            return
        self.add_node(u)
        self.add_node(v)

        if increment and v in self._adj[u]:
            new_weight = self._adj[u][v] + weight
        else:
            new_weight = weight if self.weighted else 1.0

        self._adj[u][v] = new_weight
        self._adj[v][u] = new_weight

    def has_node(self, node_id: str) -> bool:
        return node_id in self._adj

    def has_edge(self, u: str, v: str) -> bool:
        return u in self._adj and v in self._adj[u]

    def get_weight(self, u: str, v: str) -> float:
        if not self.has_edge(u, v):
            raise KeyError(f"Edge ({u!r}, {v!r}) does not exist in graph '{self.name}'.")
        return self._adj[u][v]

    def get_neighbors(self, node_id: str) -> dict[str, float]:
        if node_id not in self._adj:
            raise KeyError(f"Node {node_id!r} does not exist in graph '{self.name}'.")
        return dict(self._adj[node_id])

    def degree(self, node_id: str) -> int:
        if node_id not in self._adj:
            raise KeyError(f"Node {node_id!r} does not exist in graph '{self.name}'.")
        return len(self._adj[node_id])

    def weighted_degree(self, node_id: str) -> float:
        if node_id not in self._adj:
            raise KeyError(f"Node {node_id!r} does not exist in graph '{self.name}'.")
        return sum(self._adj[node_id].values())

    def nodes(self, node_type: str | None = None) -> list[str]:
        if node_type is None:
            return list(self._adj.keys())
        return [
            nid
            for nid, attrs in self.node_attrs.items()
            if attrs.get("node_type") == node_type
        ]

    def edges(self) -> list[tuple[str, str, float]]:
        result: list[tuple[str, str, float]] = []
        visited: set[tuple[str, str]] = set()
        for u, neighbors in self._adj.items():
            for v, weight in neighbors.items():
                pair = (u, v) if u <= v else (v, u)
                if pair not in visited:
                    visited.add(pair)
                    result.append((pair[0], pair[1], weight))
        return result

    def node_count(self, node_type: str | None = None) -> int:
        return len(self.nodes(node_type=node_type))

    def edge_count(self) -> int:
        return sum(len(neighbors) for neighbors in self._adj.values()) // 2

    def get_label(self, node_id: str) -> str:
        attrs = self.node_attrs.get(node_id, {})
        return str(attrs.get("label", node_id))

    def to_networkx(self) -> Any:
        """Converts this custom graph into a NetworkX Graph for visualization or community analysis."""
        import networkx as nx

        nx_graph = nx.Graph(name=self.name)
        for node_id, attrs in self.node_attrs.items():
            nx_graph.add_node(node_id, **attrs)
        for u, v, weight in self.edges():
            nx_graph.add_edge(u, v, weight=weight)
        return nx_graph


class AcademicNetwork:
    """Builds and exposes the four required undirected graphs from a single dataset."""

    def __init__(self, academicians: list[Academician]) -> None:
        self.academicians: list[Academician] = list(academicians)
        self.academicians_by_id: dict[str, Academician] = {
            a.acad_id: a for a in self.academicians
        }
        self.universities: dict[str, University] = {}
        self.fields: dict[str, Field] = {}

        # 1. Academician-Field Graph (bipartite, unweighted)
        self.acad_field_graph = UndirectedGraph("Academician-Field", weighted=False)
        # 2. Academician Similarity Graph (weighted by Jaccard index)
        self.acad_similarity_graph = UndirectedGraph("Academician-Similarity", weighted=True)
        # 3. Field Co-occurrence Graph (weighted by shared academician count)
        self.field_graph = UndirectedGraph("Field-Cooccurrence", weighted=True)
        # 4. University-Field Graph (bipartite, weighted by active academician count)
        self.uni_field_graph = UndirectedGraph("University-Field", weighted=True)

        self._build_entities_and_graphs()

    @staticmethod
    def acad_key(acad_id: str) -> str:
        return f"acad:{acad_id}"

    @staticmethod
    def field_key(field_name: str) -> str:
        return f"field:{field_name}"

    @staticmethod
    def uni_key(uni_name: str) -> str:
        return f"uni:{uni_name}"

    def _build_entities_and_graphs(self) -> None:
        for acad in self.academicians:
            uni_obj = self.universities.setdefault(acad.university, University(acad.university))
            uni_obj.add_academician(acad)

            a_key = self.acad_key(acad.acad_id)
            u_key = self.uni_key(acad.university)

            self.acad_field_graph.add_node(
                a_key,
                node_type="academician",
                label=acad.name,
                acad_id=acad.acad_id,
                university=acad.university,
            )
            self.acad_similarity_graph.add_node(
                acad.acad_id,
                node_type="academician",
                label=acad.name,
                university=acad.university,
            )
            self.uni_field_graph.add_node(
                u_key,
                node_type="university",
                label=acad.university,
            )

            for fld_name in acad.fields:
                fld_obj = self.fields.setdefault(fld_name, Field(fld_name))
                fld_obj.add_academician(acad)

                f_key = self.field_key(fld_name)
                self.acad_field_graph.add_node(f_key, node_type="field", label=fld_name)
                self.field_graph.add_node(fld_name, node_type="field", label=fld_name)
                self.uni_field_graph.add_node(f_key, node_type="field", label=fld_name)

                self.acad_field_graph.add_edge(a_key, f_key, weight=1.0, increment=False)
                self.uni_field_graph.add_edge(u_key, f_key, weight=1.0, increment=True)

            for f1, f2 in combinations(acad.fields, 2):
                self.field_graph.add_edge(f1, f2, weight=1.0, increment=True)

        shared_counts: dict[tuple[str, str], int] = defaultdict(int)
        for fld_name in self.fields:
            f_key = self.field_key(fld_name)
            acad_nodes = sorted(self.acad_field_graph.get_neighbors(f_key).keys())
            for a_key_1, a_key_2 in combinations(acad_nodes, 2):
                id_1 = self.acad_field_graph.node_attrs[a_key_1]["acad_id"]
                id_2 = self.acad_field_graph.node_attrs[a_key_2]["acad_id"]
                pair = (id_1, id_2) if id_1 < id_2 else (id_2, id_1)
                shared_counts[pair] += 1

        for (id_1, id_2), shared in shared_counts.items():
            deg_1 = len(self.academicians_by_id[id_1].fields)
            deg_2 = len(self.academicians_by_id[id_2].fields)
            union_size = deg_1 + deg_2 - shared
            if union_size > 0:
                jaccard = shared / union_size
                self.acad_similarity_graph.add_edge(id_1, id_2, weight=jaccard, increment=False)

    def summary(self) -> dict[str, Any]:
        """Returns entity and graph node/edge statistics across all 4 graphs."""
        graphs = [
            self.acad_field_graph,
            self.acad_similarity_graph,
            self.field_graph,
            self.uni_field_graph,
        ]
        return {
            "academicians": len(self.academicians),
            "universities": len(self.universities),
            "fields": len(self.fields),
            "graphs": {
                g.name: {"nodes": g.node_count(), "edges": g.edge_count()}
                for g in graphs
            },
        }