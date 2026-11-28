import math
import unittest

from src.algorithms import (
    bfs_shortest_path,
    component_summary,
    compute_similarities,
    compute_similarities_inverted_index,
    compute_similarities_naive,
    cosine_similarity_sets,
    detect_academic_communities,
    format_bfs_path,
    jaccard_similarity,
    overlap_similarity,
    rank_advisors,
    top_degree_nodes,
)
from src.graph import AcademicNetwork, UndirectedGraph
from src.models import Academician, FieldHierarchy


class TestAcademicNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_academicians = [
            Academician(
                acad_id="acad_1",
                name="Ayse Yilmaz",
                title="Prof. Dr.",
                university="Bursa Technical University",
                department="Computer Engineering",
                fields=["Machine Learning", "Computer Vision"],
            ),
            Academician(
                acad_id="acad_2",
                name="John Smith",
                title="Professor",
                university="Massachusetts Institute of Technology",
                department="Computer Science",
                fields=["Machine Learning", "Robotics"],
            ),
            Academician(
                acad_id="acad_3",
                name="Can Demir",
                title="Assoc. Prof. Dr.",
                university="Bursa Technical University",
                department="Computer Engineering",
                fields=["Robotics", "Control Systems"],
            ),
            Academician(
                acad_id="acad_4",
                name="Zeynep Kaya",
                title="Asst. Prof. Dr.",
                university="Istanbul Technical University",
                department="Computer Engineering",
                fields=["Cryptography"],
            ),
        ]
        self.network = AcademicNetwork(self.sample_academicians)

    def test_four_graphs_structure_and_weights(self) -> None:
        g1 = self.network.acad_field_graph
        self.assertEqual(g1.node_count(), 9)
        self.assertEqual(g1.edge_count(), 7)

        g2 = self.network.acad_similarity_graph
        self.assertEqual(g2.node_count(), 4)
        self.assertEqual(g2.edge_count(), 2)
        self.assertTrue(math.isclose(g2.get_weight("acad_1", "acad_2"), 1 / 3))
        self.assertTrue(math.isclose(g2.get_weight("acad_2", "acad_3"), 1 / 3))
        self.assertFalse(g2.has_edge("acad_1", "acad_3"))

        g3 = self.network.field_graph
        self.assertEqual(g3.node_count(), 5)
        self.assertEqual(g3.edge_count(), 3)
        self.assertEqual(g3.get_weight("Machine Learning", "Computer Vision"), 1.0)

        g4 = self.network.uni_field_graph
        self.assertEqual(g4.node_count(), 8)
        self.assertEqual(g4.edge_count(), 7)

        summary = self.network.summary()
        self.assertEqual(summary["academicians"], 4)
        self.assertEqual(summary["universities"], 3)
        self.assertEqual(summary["fields"], 5)

    def test_graph_edge_cases_and_errors(self) -> None:
        g = UndirectedGraph("Test", weighted=True)
        g.add_edge("A", "A", weight=5.0)
        self.assertEqual(g.node_count(), 0)

        g.add_edge("A", "B", weight=2.0, increment=True)
        g.add_edge("B", "A", weight=3.0, increment=True)
        self.assertEqual(g.edge_count(), 1)
        self.assertEqual(g.get_weight("A", "B"), 5.0)
        self.assertEqual(g.weighted_degree("A"), 5.0)

        with self.assertRaises(KeyError):
            g.get_weight("A", "C")
        with self.assertRaises(KeyError):
            g.degree("MISSING")

    def test_bfs_shortest_path(self) -> None:
        g1 = self.network.acad_field_graph
        start = AcademicNetwork.acad_key("acad_1")
        goal = AcademicNetwork.acad_key("acad_3")

        path = bfs_shortest_path(g1, start, goal)
        formatted = format_bfs_path(g1, path)
        self.assertEqual(
            formatted,
            "Ayse Yilmaz -> Machine Learning -> John Smith -> Robotics -> Can Demir",
        )

        self.assertEqual(bfs_shortest_path(g1, start, start), [start])
        unreachable = AcademicNetwork.acad_key("acad_4")
        self.assertEqual(bfs_shortest_path(g1, start, unreachable), [])

    def test_connected_components_and_degrees(self) -> None:
        summary = component_summary(self.network.acad_similarity_graph)
        self.assertEqual(summary["component_count"], 2)
        self.assertEqual(summary["largest_component_size"], 3)
        self.assertEqual(summary["isolated_node_count"], 1)

        top_fields = top_degree_nodes(
            self.network.acad_field_graph, top_n=2, node_type="field"
        )
        top_field_names = {self.network.acad_field_graph.get_label(nid) for nid, _ in top_fields}
        self.assertEqual(top_field_names, {"Machine Learning", "Robotics"})

        btu_top = self.network.universities["Bursa Technical University"].top_fields(top_n=5)
        self.assertEqual(len(btu_top), 4)

    def test_naive_vs_inverted_index_equivalence(self) -> None:
        naive = compute_similarities_naive(self.sample_academicians)
        inv = compute_similarities_inverted_index(self.sample_academicians)

        self.assertEqual(len(naive), len(inv))
        for (a1, b1, s1), (a2, b2, s2) in zip(naive, inv):
            self.assertEqual((a1, b1), (a2, b2))
            self.assertTrue(math.isclose(s1, s2, rel_tol=1e-12))

    def test_alternative_similarity_metrics(self) -> None:
        self.assertEqual(jaccard_similarity(set(), set()), 0.0)
        self.assertEqual(overlap_similarity(set(), {"ML"}), 0.0)
        self.assertEqual(cosine_similarity_sets(set(), {"ML"}), 0.0)

        overlap_res = compute_similarities(self.sample_academicians, metric="overlap")
        self.assertTrue(math.isclose(overlap_res[0][2], 0.5))

        cosine_res = compute_similarities(self.sample_academicians, metric="cosine")
        self.assertTrue(math.isclose(cosine_res[0][2], 0.5))

    def test_advisor_recommendation(self) -> None:
        recs = rank_advisors(
            self.sample_academicians,
            interest_fields=["Machine Learning", "Computer Vision"],
            university="Bursa Technical University",
        )
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0][0].acad_id, "acad_1")
        self.assertTrue(math.isclose(recs[0][1], 1.0))

    def test_bonus_field_hierarchy_and_community_detection(self) -> None:
        hierarchy = FieldHierarchy.from_json("data/field_hierarchy.json")
        self.assertTrue(
            math.isclose(hierarchy.relation_weight("Artificial Intelligence", "Deep Learning"), 0.25)
        )

        hierarchical_recs = rank_advisors(
            self.sample_academicians,
            interest_fields=["Deep Learning"],
            hierarchy=hierarchy,
        )
        self.assertEqual(len(hierarchical_recs), 2)
        self.assertTrue(math.isclose(hierarchical_recs[0][1], 0.5 / 2.5))

        comm_report = detect_academic_communities(self.network.acad_similarity_graph, method="louvain")
        self.assertGreaterEqual(comm_report["total_communities"], 2)
        self.assertGreaterEqual(comm_report["cross_university_communities"], 1)


if __name__ == "__main__":
    unittest.main()