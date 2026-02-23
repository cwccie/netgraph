"""Tests for anomaly detection."""

import pytest
import networkx as nx
from netgraph.detect.anomaly import AnomalyDetector, AnomalyReport


def _normal_graph():
    G = nx.Graph()
    for i in range(6):
        G.add_node(f"n{i}", type="router")
    G.add_edge("n0", "n1")
    G.add_edge("n1", "n2")
    G.add_edge("n2", "n3")
    G.add_edge("n3", "n4")
    G.add_edge("n4", "n5")
    G.add_edge("n5", "n0")
    G.add_edge("n0", "n3")
    return G


class TestAnomalyDetector:
    def test_fit(self):
        G = _normal_graph()
        detector = AnomalyDetector(hidden_dim=8, latent_dim=4)
        detector.fit(G, epochs=10, verbose=False)
        assert detector._trained

    def test_detect_returns_report(self):
        G = _normal_graph()
        detector = AnomalyDetector(hidden_dim=8, latent_dim=4)
        detector.fit(G, epochs=10, verbose=False)
        report = detector.detect(G)
        assert isinstance(report, AnomalyReport)
        assert isinstance(report.graph_score, float)

    def test_detect_node_scores(self):
        G = _normal_graph()
        detector = AnomalyDetector(hidden_dim=8, latent_dim=4)
        detector.fit(G, epochs=10, verbose=False)
        report = detector.detect(G)
        assert len(report.node_scores) == 6

    def test_structural_detection(self):
        # Create graph with an articulation point
        G = nx.Graph()
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        G.add_edge("c", "a")
        G.add_edge("c", "d")  # c is articulation point
        G.add_edge("d", "e")
        for n in G.nodes():
            G.nodes[n]["type"] = "router"

        detector = AnomalyDetector(hidden_dim=8, latent_dim=4)
        detector.fit(G, epochs=10, verbose=False)
        report = detector.detect(G)

        structural = report.details.get("structural", {})
        assert "c" in structural.get("articulation_points", [])

    def test_temporal_detection(self):
        G1 = _normal_graph()
        G2 = _normal_graph()
        G2.add_node("intruder", type="unknown")
        G2.add_edge("n0", "intruder")

        detector = AnomalyDetector(hidden_dim=8, latent_dim=4)
        detector.fit(G1, epochs=10, verbose=False)
        detector.detect(G1)  # baseline
        report = detector.detect(G2)  # changed

        # Should detect temporal change
        assert len(report.temporal_anomalies) > 0 or report.details.get("temporal")

    def test_detect_without_training(self):
        G = _normal_graph()
        detector = AnomalyDetector()
        report = detector.detect(G)
        # Should still work with statistical/structural detection
        assert isinstance(report, AnomalyReport)
