"""Tests for graph feature engineering."""

import pytest
import numpy as np
import networkx as nx
from netgraph.graph.features import FeatureEngineering, graph_to_matrices


def _sample_graph():
    G = nx.Graph()
    G.add_node("A", type="router", capabilities=["Router"])
    G.add_node("B", type="router", capabilities=["Router", "Bridge"])
    G.add_node("C", type="network_device", capabilities=["Bridge"])
    G.add_edge("A", "B", protocol="ospf", speed_mbps=10000)
    G.add_edge("B", "C", protocol="lldp", speed_mbps=1000)
    G.add_edge("A", "C", protocol="static", speed_mbps=1000)
    return G


class TestFeatureEngineering:
    def test_extract_returns_all_keys(self):
        G = _sample_graph()
        fe = FeatureEngineering()
        result = fe.extract(G)
        assert "X" in result
        assert "A" in result
        assert "A_hat" in result
        assert "node_map" in result
        assert "edge_list" in result

    def test_node_feature_shape(self):
        G = _sample_graph()
        fe = FeatureEngineering()
        result = fe.extract(G)
        assert result["X"].shape[0] == 3  # 3 nodes
        assert result["X"].shape[1] == len(fe.node_feature_names)

    def test_adjacency_shape(self):
        G = _sample_graph()
        fe = FeatureEngineering()
        result = fe.extract(G)
        assert result["A"].shape == (3, 3)
        assert result["A_hat"].shape == (3, 3)

    def test_adjacency_symmetric(self):
        G = _sample_graph()
        fe = FeatureEngineering()
        result = fe.extract(G)
        np.testing.assert_array_almost_equal(result["A"], result["A"].T)

    def test_normalized_adjacency_has_self_loops(self):
        G = _sample_graph()
        fe = FeatureEngineering()
        result = fe.extract(G)
        # Diagonal should be non-zero
        assert np.all(np.diag(result["A_hat"]) > 0)

    def test_edge_features(self):
        G = _sample_graph()
        fe = FeatureEngineering()
        result = fe.extract(G)
        assert result["E"].shape[0] == 3  # 3 edges
        assert result["E"].shape[1] == len(fe.edge_feature_names)

    def test_node_map(self):
        G = _sample_graph()
        fe = FeatureEngineering()
        result = fe.extract(G)
        assert len(result["node_map"]) == 3
        assert set(result["node_map"].keys()) == {"A", "B", "C"}

    def test_normalization(self):
        G = _sample_graph()
        fe = FeatureEngineering(normalize=True)
        result = fe.extract(G)
        # Normalized columns should have mean ~0
        means = result["X"].mean(axis=0)
        for m in means:
            assert abs(m) < 0.1

    def test_no_normalization(self):
        G = _sample_graph()
        fe = FeatureEngineering(normalize=False)
        result = fe.extract(G)
        # Degree feature should be actual values
        assert result["X"].dtype == np.float32

    def test_convenience_function(self):
        G = _sample_graph()
        result = graph_to_matrices(G)
        assert "X" in result
        assert "A_hat" in result
