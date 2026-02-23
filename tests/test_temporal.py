"""Tests for temporal graph builder."""

import pytest
import numpy as np
import networkx as nx
from netgraph.graph.temporal import TemporalGraphBuilder, SubgraphExtractor


def _graph_v1():
    G = nx.Graph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "A")
    return G


def _graph_v2():
    G = _graph_v1()
    G.add_node("D")
    G.add_edge("C", "D")
    return G


class TestTemporalGraph:
    def test_add_snapshot(self):
        tb = TemporalGraphBuilder()
        snap = tb.add_snapshot(_graph_v1(), timestamp=100.0)
        assert snap.node_count == 3
        assert snap.edge_count == 3
        assert len(tb.snapshots) == 1

    def test_diff_added_node(self):
        tb = TemporalGraphBuilder()
        s1 = tb.add_snapshot(_graph_v1(), timestamp=100.0)
        s2 = tb.add_snapshot(_graph_v2(), timestamp=200.0)
        delta = tb.diff(s1, s2)
        assert "D" in delta.added_nodes
        assert len(delta.removed_nodes) == 0

    def test_diff_added_edge(self):
        tb = TemporalGraphBuilder()
        s1 = tb.add_snapshot(_graph_v1(), timestamp=100.0)
        s2 = tb.add_snapshot(_graph_v2(), timestamp=200.0)
        delta = tb.diff(s1, s2)
        assert ("C", "D") in delta.added_edges or ("D", "C") in delta.added_edges

    def test_diff_stable(self):
        tb = TemporalGraphBuilder()
        s1 = tb.add_snapshot(_graph_v1(), timestamp=100.0)
        s2 = tb.add_snapshot(_graph_v1(), timestamp=200.0)
        delta = tb.diff(s1, s2)
        assert delta.is_stable

    def test_adjacency_tensor(self):
        tb = TemporalGraphBuilder()
        tb.add_snapshot(_graph_v1(), timestamp=100.0)
        tb.add_snapshot(_graph_v2(), timestamp=200.0)
        tensor, nodes = tb.build_adjacency_tensor()
        assert tensor.shape[0] == 2  # 2 timesteps
        assert tensor.shape[1] == tensor.shape[2]  # N x N
        assert "D" in nodes

    def test_evolution_stats(self):
        tb = TemporalGraphBuilder()
        tb.add_snapshot(_graph_v1(), timestamp=100.0)
        tb.add_snapshot(_graph_v2(), timestamp=200.0)
        stats = tb.evolution_stats()
        assert stats["snapshots"] == 2
        assert stats["total_node_additions"] == 1

    def test_window(self):
        tb = TemporalGraphBuilder()
        for i in range(10):
            tb.add_snapshot(_graph_v1(), timestamp=float(i * 100))
        window = tb.get_window(window_size=3)
        assert len(window) == 3

    def test_max_snapshots(self):
        tb = TemporalGraphBuilder(max_snapshots=5)
        for i in range(10):
            tb.add_snapshot(_graph_v1(), timestamp=float(i))
        assert len(tb.snapshots) == 5


class TestSubgraphExtractor:
    def test_k_hop(self):
        G = nx.path_graph(6)  # 0-1-2-3-4-5
        sub = SubgraphExtractor.k_hop_neighborhood(G, 2, k=1)
        assert set(sub.nodes()) == {1, 2, 3}

    def test_critical_path(self):
        G = nx.path_graph(5)
        sub = SubgraphExtractor.critical_path(G, 0, 4)
        assert set(sub.nodes()) == {0, 1, 2, 3, 4}

    def test_high_centrality(self):
        G = nx.star_graph(5)  # node 0 is center
        sub = SubgraphExtractor.high_centrality(G, top_k=1, metric="betweenness")
        assert 0 in sub.nodes()
