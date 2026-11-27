from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Set as AbstractSet
from itertools import combinations
from typing import Any

from src.graph import UndirectedGraph
from src.models import Academician, FieldHierarchy


def bfs_shortest_path(graph: UndirectedGraph, start: str, goal: str) -> list[str]:
    """Finds the unweighted shortest path between start and goal using BFS."""
    if not graph.has_node(start):
        raise KeyError(f"Start node {start!r} not found in graph '{graph.name}'.")
    if not graph.has_node(goal):
        raise KeyError(f"Goal node {goal!r} not found in graph '{graph.name}'.")

    if start == goal:
        return [start]

    visited: set[str] = {start}
    parent: dict[str, str | None] = {start: None}
    queue: deque[str] = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                if neighbor == goal:
                    path: list[str] = []
                    node: str | None = goal
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                queue.append(neighbor)

    return []


def format_bfs_path(graph: UndirectedGraph, path: list[str]) -> str:
    """Formats a node ID path into human-readable labels separated by arrows."""
    return " -> ".join(graph.get_label(node_id) for node_id in path)


def top_degree_nodes(
    graph: UndirectedGraph,
    top_n: int = 10,
    node_type: str | None = None,
    weighted: bool = False,
) -> list[tuple[str, float | int]]:
    """Returns top_n nodes ranked by degree (or weighted degree), optionally filtered by node_type."""
    candidates = graph.nodes(node_type=node_type)
    scored: list[tuple[str, float | int]] = []

    for node_id in candidates:
        score: float | int = (
            graph.weighted_degree(node_id) if weighted else graph.degree(node_id)
        )
        scored.append((node_id, score))

    scored.sort(key=lambda item: (-item[1], graph.get_label(item[0])))
    return scored[:top_n]


def connected_components(graph: UndirectedGraph) -> list[list[str]]:
    """Finds all connected components in an undirected graph, sorted by descending size."""
    visited: set[str] = set()
    components: list[list[str]] = []

    for node_id in graph.nodes():
        if node_id in visited:
            continue
        comp: list[str] = []
        queue: deque[str] = deque([node_id])
        visited.add(node_id)

        while queue:
            current = queue.popleft()
            comp.append(current)
            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        comp.sort()
        components.append(comp)

    components.sort(key=lambda c: (-len(c), c[0]))
    return components


def component_summary(graph: UndirectedGraph) -> dict[str, Any]:
    """Summarizes component count, largest component size, and isolated node count."""
    comps = connected_components(graph)
    return {
        "component_count": len(comps),
        "largest_component_size": len(comps[0]) if comps else 0,
        "isolated_node_count": sum(1 for c in comps if len(c) == 1),
        "components": comps,
    }


def jaccard_similarity(set_a: AbstractSet[str], set_b: AbstractSet[str]) -> float:
    """Computes |A & B| / |A | B| safely for two sets."""
    if not set_a and not set_b:
        return 0.0
    shared = len(set_a & set_b)
    if shared == 0:
        return 0.0
    union_size = len(set_a) + len(set_b) - shared
    return shared / union_size


def overlap_similarity(set_a: AbstractSet[str], set_b: AbstractSet[str]) -> float:
    """Computes Szymkiewicz-Simpson overlap coefficient |A & B| / min(|A|, |B|)."""
    if not set_a or not set_b:
        return 0.0
    shared = len(set_a & set_b)
    return shared / min(len(set_a), len(set_b))


def cosine_similarity_sets(set_a: AbstractSet[str], set_b: AbstractSet[str]) -> float:
    """Computes Ochiai/cosine similarity |A & B| / sqrt(|A| * |B|)."""
    if not set_a or not set_b:
        return 0.0
    shared = len(set_a & set_b)
    return shared / math.sqrt(len(set_a) * len(set_b))


def compute_similarities_naive(
    academicians: list[Academician],
    min_similarity: float = 0.0,
) -> list[tuple[str, str, float]]:
    """Method A (Brute-force): Evaluates all N*(N-1)/2 academician pairs."""
    results: list[tuple[str, str, float]] = []
    for acad_a, acad_b in combinations(academicians, 2):
        score = jaccard_similarity(acad_a.field_set, acad_b.field_set)
        if score > 0.0 and score >= min_similarity:
            id_1, id_2 = (
                (acad_a.acad_id, acad_b.acad_id)
                if acad_a.acad_id < acad_b.acad_id
                else (acad_b.acad_id, acad_a.acad_id)
            )
            results.append((id_1, id_2, score))

    results.sort(key=lambda item: (-item[2], item[0], item[1]))
    return results


def compute_similarities_inverted_index(
    academicians: list[Academician],
    min_similarity: float = 0.0,
) -> list[tuple[str, str, float]]:
    """Method B (Inverted Index, integer-indexed): Evaluates only pairs sharing >= 1 field."""
    ordered = sorted(academicians, key=lambda a: a.acad_id)
    field_to_indices: dict[str, list[int]] = defaultdict(list)
    field_counts: list[int] = [len(a.fields) for a in ordered]

    for idx, acad in enumerate(ordered):
        for fld in acad.fields:
            field_to_indices[fld].append(idx)

    shared_counts: dict[tuple[int, int], int] = defaultdict(int)
    for indices in field_to_indices.values():
        for i, j in combinations(indices, 2):
            shared_counts[(i, j)] += 1

    results: list[tuple[str, str, float]] = []
    for (i, j), shared in shared_counts.items():
        union_size = field_counts[i] + field_counts[j] - shared
        score = shared / union_size
        if score >= min_similarity:
            results.append((ordered[i].acad_id, ordered[j].acad_id, score))

    results.sort(key=lambda item: (-item[2], item[0], item[1]))
    return results


def compute_similarities(
    academicians: list[Academician],
    metric: str = "jaccard",
    min_similarity: float = 0.0,
) -> list[tuple[str, str, float]]:
    """Extensible similarity dispatcher supporting 'jaccard', 'overlap', and 'cosine'."""
    metric_key = metric.strip().lower()
    if metric_key == "jaccard":
        return compute_similarities_inverted_index(academicians, min_similarity=min_similarity)

    ordered = sorted(academicians, key=lambda a: a.acad_id)
    field_to_indices: dict[str, list[int]] = defaultdict(list)
    field_counts: list[int] = [len(a.fields) for a in ordered]

    for idx, acad in enumerate(ordered):
        for fld in acad.fields:
            field_to_indices[fld].append(idx)

    shared_counts: dict[tuple[int, int], int] = defaultdict(int)
    for indices in field_to_indices.values():
        for i, j in combinations(indices, 2):
            shared_counts[(i, j)] += 1

    results: list[tuple[str, str, float]] = []
    for (i, j), shared in shared_counts.items():
        if metric_key == "overlap":
            score = shared / min(field_counts[i], field_counts[j])
        elif metric_key == "cosine":
            score = shared / math.sqrt(field_counts[i] * field_counts[j])
        else:
            raise ValueError(f"Unsupported similarity metric: {metric!r}")

        if score >= min_similarity:
            results.append((ordered[i].acad_id, ordered[j].acad_id, score))

    results.sort(key=lambda item: (-item[2], item[0], item[1]))
    return results


def rank_advisors(
    academicians: list[Academician],
    interest_fields: Iterable[str],
    university: str | None = None,
    top_n: int = 10,
    hierarchy: FieldHierarchy | None = None,
    hierarchy_decay: float = 0.5,
) -> list[tuple[Academician, float, list[str]]]:
    """Scores and ranks academicians against target interest fields (supports optional taxonomy)."""
    normalized_targets = {f.strip() for f in interest_fields if f and f.strip()}
    if not normalized_targets:
        return []

    uni_filter = university.strip().lower() if university and university.strip() else None
    ranked: list[tuple[Academician, float, list[str]]] = []

    if hierarchy is None:
        target_lower_map = {f.lower(): f for f in normalized_targets}
        for acad in academicians:
            if uni_filter and acad.university.lower() != uni_filter:
                continue

            acad_lower_set = {f.lower() for f in acad.fields}
            matched_lower = acad_lower_set & set(target_lower_map.keys())
            if not matched_lower:
                continue

            union_size = len(acad_lower_set) + len(target_lower_map) - len(matched_lower)
            score = len(matched_lower) / union_size
            matched_display = sorted(
                f for f in acad.fields if f.lower() in matched_lower
            )
            ranked.append((acad, score, matched_display))
    else:
        for acad in academicians:
            if uni_filter and acad.university.lower() != uni_filter:
                continue
            if not acad.fields:
                continue

            weighted_intersection = 0.0
            matched_notes: dict[str, tuple[float, str]] = {}

            for target in normalized_targets:
                best_w = 0.0
                best_field = ""
                for acad_fld in acad.fields:
                    w = hierarchy.relation_weight(target, acad_fld, decay=hierarchy_decay)
                    if w > best_w:
                        best_w = w
                        best_field = acad_fld
                if best_w > 0.0:
                    weighted_intersection += best_w
                    label = best_field if best_w == 1.0 else f"{best_field} (~{target})"
                    prev = matched_notes.get(best_field)
                    if prev is None or best_w > prev[0]:
                        matched_notes[best_field] = (best_w, label)

            if weighted_intersection <= 0.0:
                continue

            union_size = len(acad.fields) + len(normalized_targets) - weighted_intersection
            score = weighted_intersection / union_size
            matched_display = [
                lbl for _, lbl in sorted(matched_notes.values(), key=lambda x: (-x[0], x[1]))
            ]
            ranked.append((acad, score, matched_display))

    ranked.sort(key=lambda item: (-item[1], -len(item[2]), item[0].name))
    return ranked[:top_n]


def detect_academic_communities(
    similarity_graph: UndirectedGraph,
    method: str = "louvain",
    seed: int = 42,
) -> dict[str, Any]:
    """Bonus: Detects communities on the similarity graph and checks university boundary crossings."""
    import networkx as nx

    nx_graph = similarity_graph.to_networkx()
    if nx_graph.number_of_nodes() == 0:
        return {
            "method": method,
            "total_communities": 0,
            "cross_university_communities": 0,
            "single_university_communities": 0,
            "cross_university_ratio": 0.0,
            "modularity": 0.0,
            "communities": [],
        }

    method_key = method.strip().lower()
    if method_key == "louvain":
        raw_communities = list(
            nx.community.louvain_communities(nx_graph, weight="weight", seed=seed)
        )
    elif method_key == "greedy_modularity":
        raw_communities = list(
            nx.community.greedy_modularity_communities(nx_graph, weight="weight")
        )
    else:
        raise ValueError(f"Unsupported community detection method: {method!r}")

    modularity = (
        float(nx.community.modularity(nx_graph, raw_communities, weight="weight"))
        if nx_graph.number_of_edges() > 0
        else 0.0
    )

    sorted_comms = sorted(
        (sorted(c) for c in raw_communities),
        key=lambda members: (-len(members), members[0]),
    )

    community_details: list[dict[str, Any]] = []
    cross_uni_count = 0

    for idx, members in enumerate(sorted_comms, start=1):
        uni_counter: Counter[str] = Counter()
        member_names: list[str] = []
        for nid in members:
            member_names.append(similarity_graph.get_label(nid))
            uni_name = str(similarity_graph.node_attrs.get(nid, {}).get("university", "Unknown"))
            uni_counter[uni_name] += 1

        dominant_uni, dominant_cnt = uni_counter.most_common(1)[0]
        crosses = len(uni_counter) > 1
        if crosses:
            cross_uni_count += 1

        community_details.append(
            {
                "community_id": idx,
                "size": len(members),
                "members": members,
                "member_names": member_names,
                "universities": dict(uni_counter),
                "university_count": len(uni_counter),
                "crosses_universities": crosses,
                "dominant_university": dominant_uni,
                "dominant_ratio": dominant_cnt / len(members),
            }
        )

    total = len(community_details)
    return {
        "method": method_key,
        "total_communities": total,
        "cross_university_communities": cross_uni_count,
        "single_university_communities": total - cross_uni_count,
        "cross_university_ratio": (cross_uni_count / total) if total > 0 else 0.0,
        "modularity": modularity,
        "communities": community_details,
    }