"""Tests for impact analysis."""

import pytest
import networkx as nx
from netgraph.impact.analysis import ImpactAnalyzer


def _sample_network():
    G = nx.Graph()
    # Core-Distribution-Access hierarchy
    G.add_node("core1", type="router")
    G.add_node("core2", type="router")
    G.add_node("dist1", type="router")
    G.add_node("dist2", type="router")
    G.add_node("acc1", type="network_device")
    G.add_node("acc2", type="network_device")
    G.add_node("acc3", type="network_device")

    G.add_edge("core1", "core2")
    G.add_edge("core1", "dist1")
    G.add_edge("core1", "dist2")
    G.add_edge("core2", "dist1")
    G.add_edge("core2", "dist2")
    G.add_edge("dist1", "acc1")
    G.add_edge("dist1", "acc2")
    G.add_edge("dist2", "acc3")
    return G


class TestImpactAnalyzer:
    def test_blast_radius_node(self):
        G = _sample_network()
        analyzer = ImpactAnalyzer()
        br = analyzer.blast_radius_node(G, "core1")
        assert br.failed_entity == "core1"
        assert len(br.directly_affected) > 0

    def test_blast_radius_critical_node(self):
        # dist2 is only connection to acc3
        G = _sample_network()
        analyzer = ImpactAnalyzer()
        br = analyzer.blast_radius_node(G, "dist2")
        assert "acc3" in br.directly_affected or "acc3" in br.indirectly_affected

    def test_blast_radius_edge(self):
        G = _sample_network()
        analyzer = ImpactAnalyzer()
        br = analyzer.blast_radius_edge(G, "dist2", "acc3")
        assert br.entity_type == "edge"
        # Removing this link isolates acc3
        assert br.disconnected_pairs > 0 or "acc3" in br.isolated_nodes

    def test_redundancy_score(self):
        G = _sample_network()
        analyzer = ImpactAnalyzer()
        rs = analyzer.redundancy_score(G, "dist1")
        assert 0.0 <= rs.score <= 1.0

    def test_spof_detection(self):
        G = _sample_network()
        analyzer = ImpactAnalyzer()
        rs = analyzer.redundancy_score(G, "dist2")
        assert rs.is_single_point_of_failure

    def test_critical_paths(self):
        G = _sample_network()
        analyzer = ImpactAnalyzer()
        paths = analyzer.find_critical_paths(G, top_k=3)
        assert len(paths) > 0
        assert "betweenness_centrality" in paths[0]

    def test_what_if(self):
        G = _sample_network()
        analyzer = ImpactAnalyzer()
        results = analyzer.what_if(G, [
            {"type": "node", "target": "core1"},
            {"type": "edge", "target": ("core2", "dist2")},
        ])
        assert len(results) == 2
        assert results[0]["removed"] == "core1"

    def test_resilience(self):
        G = _sample_network()
        analyzer = ImpactAnalyzer()
        score = analyzer.compute_resilience(G)
        assert 0.0 <= score <= 1.0

    def test_full_analysis(self):
        G = _sample_network()
        analyzer = ImpactAnalyzer()
        report = analyzer.analyze(G)
        assert report.overall_resilience >= 0
        assert "resilience_grade" in report.summary
        assert len(report.blast_radii) == G.number_of_nodes()

    def test_nonexistent_node(self):
        G = _sample_network()
        analyzer = ImpactAnalyzer()
        br = analyzer.blast_radius_node(G, "nonexistent")
        assert br.total_affected == 0
