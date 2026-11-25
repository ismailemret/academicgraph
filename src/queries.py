"""
Query engine and advisor recommendation module (Member 3).
Returns pure Python data structures without writing to stdout.
"""

from collections import deque
from typing import Any, Dict, List, Optional


def _norm(text: str) -> str:
    """Normalize text by trimming, lowercasing, and collapsing whitespace."""
    return " ".join(text.strip().lower().split())


class QueryEngine:
    """
    Executes the 10 required queries across the 4 undirected graph structures
    and academician dataset. Compatible with both U2 graph objects and dict adapters.
    """

    def __init__(
        self,
        academicians: List[Any],
        bipartite_graph: Any,
        similarity_graph: Any,
        field_graph: Any,
        uni_field_graph: Any,
        university_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        self.academicians = academicians
        self.bipartite_graph = bipartite_graph
        self.similarity_graph = similarity_graph
        self.field_graph = field_graph
        self.uni_field_graph = uni_field_graph
        self.university_aliases: Dict[str, str] = university_aliases or {}

        self._acad_by_name: Dict[str, Any] = {}
        self._acad_by_id: Dict[str, Any] = {}
        for acad in academicians:
            aid = self._get_attr(acad, "id")
            name = self._get_attr(acad, "name")
            if aid:
                self._acad_by_id[aid] = acad
            if name:
                self._acad_by_name[_norm(name)] = acad

    def _resolve_university(self, raw_name: str) -> str:
        """Resolve a university name or alias to its canonical normalized key."""
        normed = _norm(raw_name)
        canonical = self.university_aliases.get(normed)
        return _norm(canonical) if canonical else normed

    @staticmethod
    def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
        """Read attribute or dictionary key safely."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _neighbors(self, graph: Any, node: str) -> Dict[str, float]:
        """Return adjacent nodes and edge weights from a graph instance."""
        if hasattr(graph, "get_neighbors"):
            res = graph.get_neighbors(node)
            if isinstance(res, dict):
                return res
            return {n: 1.0 for n in res}
        if hasattr(graph, "adj") and node in graph.adj:
            neighbors = graph.adj[node]
            if isinstance(neighbors, dict):
                return neighbors
            return {n: 1.0 for n in neighbors}
        if isinstance(graph, dict) and node in graph:
            neighbors = graph[node]
            if isinstance(neighbors, dict):
                return neighbors
            return {n: 1.0 for n in neighbors}
        return {}

    def _nodes(self, graph: Any) -> List[str]:
        """Return all node identifiers from a graph instance."""
        if hasattr(graph, "get_nodes"):
            return list(graph.get_nodes())
        if hasattr(graph, "nodes"):
            return list(graph.nodes() if callable(graph.nodes) else graph.nodes)
        if hasattr(graph, "adj"):
            return list(graph.adj.keys())
        if isinstance(graph, dict):
            return list(graph.keys())
        return []

    @staticmethod
    def jaccard(fields_a: set, fields_b: set) -> float:
        """Compute Jaccard similarity: |A & B| / |A | B|."""
        if not fields_a and not fields_b:
            return 0.0
        union = fields_a | fields_b
        return len(fields_a & fields_b) / len(union) if union else 0.0

    # --- Query 1: Basic Lookups ---
    def list_by_field(self, field_name: str) -> List[Dict[str, Any]]:
        """Return all academicians working in the given research field."""
        target = _norm(field_name)
        results = []
        for acad in self.academicians:
            fields = self._get_attr(acad, "fields", [])
            if any(_norm(f) == target for f in fields):
                results.append(
                    {
                        "id": self._get_attr(acad, "id"),
                        "name": self._get_attr(acad, "name"),
                        "title": self._get_attr(acad, "title"),
                        "university": self._get_attr(acad, "university"),
                        "fields": list(fields),
                    }
                )
        return results

    def list_fields_of_academician(self, academician_name: str) -> Optional[Dict[str, Any]]:
        """Return profile details and research fields for a single academician."""
        acad = self._acad_by_name.get(_norm(academician_name))
        if not acad:
            return None
        return {
            "id": self._get_attr(acad, "id"),
            "name": self._get_attr(acad, "name"),
            "title": self._get_attr(acad, "title"),
            "university": self._get_attr(acad, "university"),
            "fields": list(self._get_attr(acad, "fields", [])),
        }

    def list_by_university(self, university_name: str) -> List[Dict[str, Any]]:
        """Return all academicians affiliated with the given university or alias."""
        target = self._resolve_university(university_name)
        results = []
        for acad in self.academicians:
            uni = self._get_attr(acad, "university", "")
            if self._resolve_university(uni) == target:
                results.append(
                    {
                        "id": self._get_attr(acad, "id"),
                        "name": self._get_attr(acad, "name"),
                        "title": self._get_attr(acad, "title"),
                        "department": self._get_attr(acad, "department"),
                        "fields": list(self._get_attr(acad, "fields", [])),
                    }
                )
        return results

    # --- Query 2: Shortest Path via BFS on Academician-Field Graph ---
    def shortest_path_bfs(self, source_name: str, target_name: str) -> Optional[List[str]]:
        """
        Find the shortest path between two academicians in the bipartite
        Academician-Field graph using Breadth-First Search.
        Returns None if either academician does not exist, or [] if unreachable.
        """
        src_acad = self._acad_by_name.get(_norm(source_name))
        dst_acad = self._acad_by_name.get(_norm(target_name))
        if not src_acad or not dst_acad:
            return None

        start_label = self._get_attr(src_acad, "name")
        goal_label = self._get_attr(dst_acad, "name")
        if start_label == goal_label:
            return [start_label]

        nodes = set(self._nodes(self.bipartite_graph))
        start_node = start_label if start_label in nodes else self._get_attr(src_acad, "id")
        goal_node = goal_label if goal_label in nodes else self._get_attr(dst_acad, "id")
        if start_node not in nodes or goal_node not in nodes:
            return None

        queue = deque([start_node])
        visited = {start_node: None}

        while queue:
            curr = queue.popleft()
            if curr == goal_node:
                path = []
                while curr is not None:
                    acad_obj = self._acad_by_id.get(curr)
                    path.append(self._get_attr(acad_obj, "name") if acad_obj else curr)
                    curr = visited[curr]
                return list(reversed(path))

            for nxt in self._neighbors(self.bipartite_graph, curr):
                if nxt not in visited:
                    visited[nxt] = curr
                    queue.append(nxt)

        return []

    # --- Query 3: Degree Centrality ---
    def top_central_nodes(self, top_n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Rank academicians by field count and fields by academician count."""
        acad_degrees = []
        for acad in self.academicians:
            fields = self._get_attr(acad, "fields", [])
            acad_degrees.append(
                {
                    "name": self._get_attr(acad, "name"),
                    "university": self._get_attr(acad, "university"),
                    "degree": len(set(fields)),
                }
            )
        acad_degrees.sort(key=lambda x: (-x["degree"], x["name"]))

        field_counts: Dict[str, int] = {}
        for acad in self.academicians:
            for f in set(self._get_attr(acad, "fields", [])):
                field_counts[f] = field_counts.get(f, 0) + 1

        field_degrees = [
            {"field": f, "degree": cnt}
            for f, cnt in sorted(field_counts.items(), key=lambda x: (-x[1], x[0]))
        ]

        return {
            "top_academicians": acad_degrees[:top_n],
            "top_fields": field_degrees[:top_n],
        }

    # --- Query 4: Connected Components on Similarity Graph ---
    def connected_components_summary(self) -> Dict[str, Any]:
        """Analyze connected components in the Academician Similarity graph."""
        nodes = self._nodes(self.similarity_graph)
        visited = set()
        components: List[List[str]] = []

        for node in nodes:
            if node in visited:
                continue
            comp = []
            queue = deque([node])
            visited.add(node)
            while queue:
                curr = queue.popleft()
                acad_obj = self._acad_by_id.get(curr)
                comp.append(self._get_attr(acad_obj, "name") if acad_obj else curr)
                for nxt in self._neighbors(self.similarity_graph, curr):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            components.append(comp)

        components.sort(key=len, reverse=True)
        largest = components[0] if components else []
        return {
            "component_count": len(components),
            "largest_component_size": len(largest),
            "largest_component_members": largest,
            "is_fully_connected": len(components) <= 1,
        }

    # --- Query 5: Top Similar Academicians ---
    def top_similar_academicians(self, academician_name: str, top_n: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Return top-N most similar academicians ranked by Jaccard similarity."""
        target = self._acad_by_name.get(_norm(academician_name))
        if not target:
            return None

        target_id = self._get_attr(target, "id")
        target_fields = set(self._get_attr(target, "fields", []))
        scored = []

        for other in self.academicians:
            other_id = self._get_attr(other, "id")
            if other_id == target_id:
                continue
            other_fields = set(self._get_attr(other, "fields", []))
            sim = self.jaccard(target_fields, other_fields)
            if sim > 0:
                scored.append(
                    {
                        "name": self._get_attr(other, "name"),
                        "university": self._get_attr(other, "university"),
                        "similarity": round(sim, 4),
                        "shared_fields": sorted(target_fields & other_fields),
                    }
                )

        scored.sort(key=lambda x: (-x["similarity"], x["name"]))
        return scored[:top_n]

    # --- Query 6: Strongest Fields per University ---
    def top_fields_per_university(self, top_n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Return top-N research fields by headcount for each university."""
        uni_field_counts: Dict[str, Dict[str, int]] = {}
        for acad in self.academicians:
            uni = self._get_attr(acad, "university", "Unknown")
            uni_field_counts.setdefault(uni, {})
            for f in set(self._get_attr(acad, "fields", [])):
                uni_field_counts[uni][f] = uni_field_counts[uni].get(f, 0) + 1

        result: Dict[str, List[Dict[str, Any]]] = {}
        for uni in sorted(uni_field_counts):
            ranked = sorted(uni_field_counts[uni].items(), key=lambda x: (-x[1], x[0]))[:top_n]
            result[uni] = [{"field": f, "academician_count": cnt} for f, cnt in ranked]
        return result

    # --- Query 7: Most Similar University Pairs ---
    def most_similar_universities(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Rank university pairs by Jaccard similarity of their aggregate field sets."""
        uni_profiles: Dict[str, set] = {}
        for acad in self.academicians:
            uni = self._get_attr(acad, "university", "Unknown")
            uni_profiles.setdefault(uni, set()).update(self._get_attr(acad, "fields", []))

        unis = sorted(uni_profiles.keys())
        pairs = []
        for i in range(len(unis)):
            for j in range(i + 1, len(unis)):
                u1, u2 = unis[i], unis[j]
                sim = self.jaccard(uni_profiles[u1], uni_profiles[u2])
                if sim > 0:
                    pairs.append(
                        {
                            "university_1": u1,
                            "university_2": u2,
                            "similarity": round(sim, 4),
                            "shared_field_count": len(uni_profiles[u1] & uni_profiles[u2]),
                        }
                    )
        pairs.sort(key=lambda x: (-x["similarity"], x["university_1"], x["university_2"]))
        return pairs[:top_n]

    # --- Query 8: Cross-University Collaboration Recommendations ---
    def cross_university_collaborations(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Rank academician pairs from different universities by Jaccard similarity."""
        pairs = []
        n = len(self.academicians)
        for i in range(n):
            a1 = self.academicians[i]
            u1 = self._get_attr(a1, "university")
            f1 = set(self._get_attr(a1, "fields", []))
            for j in range(i + 1, n):
                a2 = self.academicians[j]
                u2 = self._get_attr(a2, "university")
                if _norm(u1) == _norm(u2):
                    continue
                f2 = set(self._get_attr(a2, "fields", []))
                sim = self.jaccard(f1, f2)
                if sim > 0:
                    pairs.append(
                        {
                            "academician_1": self._get_attr(a1, "name"),
                            "university_1": u1,
                            "academician_2": self._get_attr(a2, "name"),
                            "university_2": u2,
                            "similarity": round(sim, 4),
                            "shared_fields": sorted(f1 & f2),
                        }
                    )
        pairs.sort(key=lambda x: (-x["similarity"], x["academician_1"], x["academician_2"]))
        return pairs[:top_n]

    # --- Query 9: Rare Research Fields ---
    def rare_fields(self, max_universities: int = 2) -> List[Dict[str, Any]]:
        """Identify fields studied at at most max_universities institutions."""
        field_unis: Dict[str, set] = {}
        field_acad_counts: Dict[str, int] = {}

        for acad in self.academicians:
            uni = self._get_attr(acad, "university", "Unknown")
            for f in set(self._get_attr(acad, "fields", [])):
                field_unis.setdefault(f, set()).add(uni)
                field_acad_counts[f] = field_acad_counts.get(f, 0) + 1

        rare = [
            {
                "field": f,
                "university_count": len(unis),
                "universities": sorted(unis),
                "academician_count": field_acad_counts[f],
            }
            for f, unis in field_unis.items()
            if len(unis) <= max_universities
        ]
        rare.sort(key=lambda x: (x["university_count"], -x["academician_count"], x["field"]))
        return rare

    # --- Query 10: Advisor Recommendation ---
    def recommend_advisors(
        self,
        interest_fields: List[str],
        university_filter: Optional[str] = None,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Rank candidate advisors by Jaccard similarity against user interest fields,
        optionally restricted to a single university (supports canonical names and aliases).
        """
        query_norm_map = {_norm(f): f.strip() for f in interest_fields if f.strip()}
        if not query_norm_map:
            return []

        query_set = set(query_norm_map.keys())
        uni_target = self._resolve_university(university_filter) if university_filter else None
        candidates = []

        for acad in self.academicians:
            uni = self._get_attr(acad, "university", "")
            if uni_target and self._resolve_university(uni) != uni_target:
                continue

            raw_fields = self._get_attr(acad, "fields", [])
            acad_norm_set = {_norm(f) for f in raw_fields}
            sim = self.jaccard(query_set, acad_norm_set)
            if sim > 0:
                matched = [f for f in raw_fields if _norm(f) in query_set]
                candidates.append(
                    {
                        "name": self._get_attr(acad, "name"),
                        "title": self._get_attr(acad, "title"),
                        "university": uni,
                        "score": round(sim, 4),
                        "matched_fields": sorted(set(matched)),
                        "all_fields": list(raw_fields),
                    }
                )

        candidates.sort(key=lambda x: (-x["score"], x["name"]))
        return candidates[:top_n]