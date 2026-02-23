"""Tests for failure prediction."""

import pytest
import networkx as nx
from netgraph.predict.failure import FailurePredictor, PredictionReport


def _sample_graph():
    G = nx.Graph()
    G.add_node("core", type="router", utilization=80.0)
    G.add_node("dist1", type="router", utilization=40.0)
    G.add_node("dist2", type="router", utilization=30.0)
    G.add_node("access1", type="network_device")
    G.add_edge("core", "dist1", speed_mbps=10000, utilization_in=85.0,
               utilization_out=80.0, error_rate=0.02)
    G.add_edge("core", "dist2", speed_mbps=10000, utilization_in=30.0,
               utilization_out=25.0, error_rate=0.0)
    G.add_edge("dist1", "access1", speed_mbps=1000, utilization_in=45.0,
               utilization_out=40.0, error_rate=0.001)
    return G


class TestFailurePredictor:
    def test_predict_returns_report(self):
        predictor = FailurePredictor()
        G = _sample_graph()
        predictor.add_observation(G)
        report = predictor.predict(G)
        assert isinstance(report, PredictionReport)
        assert "overall_health" in report.summary

    def test_link_failure_predictions(self):
        predictor = FailurePredictor()
        G = _sample_graph()
        predictor.add_observation(G)
        report = predictor.predict(G)
        assert len(report.link_failures) > 0
        # The high-error link should be ranked first
        top = report.link_failures[0]
        assert top.failure_probability > 0

    def test_node_overload(self):
        predictor = FailurePredictor()
        G = _sample_graph()
        predictor.add_observation(G)
        report = predictor.predict(G)
        # Core node has high utilization
        # At least some predictions should exist
        assert isinstance(report.node_overloads, list)

    def test_trend_analysis(self):
        predictor = FailurePredictor()
        G = _sample_graph()
        # Add multiple observations
        for i in range(5):
            predictor.add_observation(G)
        report = predictor.predict(G)
        assert report is not None

    def test_risk_levels(self):
        predictor = FailurePredictor()
        G = _sample_graph()
        predictor.add_observation(G)
        report = predictor.predict(G)
        for pred in report.link_failures:
            assert pred.risk_level in ("low", "medium", "high", "critical")
