"""Tests for visualization export."""

import pytest
import json
import networkx as nx
from netgraph.viz.export import D3Exporter, MermaidExporter, GrafanaExporter, HTMLExporter


def _sample_graph():
    G = nx.Graph()
    G.add_node("R1", type="router", capabilities=["Router"])
    G.add_node("S1", type="network_device", capabilities=["Bridge"])
    G.add_node("R2", type="router", capabilities=["Router"])
    G.add_edge("R1", "S1", protocol="lldp", local_port="Gi0/1", speed_mbps=1000)
    G.add_edge("S1", "R2", protocol="lldp", remote_port="Gi0/2", speed_mbps=1000)
    return G


class TestD3Exporter:
    def test_to_d3_json(self):
        G = _sample_graph()
        data = D3Exporter.to_d3_json(G)
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == 3
        assert len(data["links"]) == 2

    def test_d3_node_attributes(self):
        G = _sample_graph()
        data = D3Exporter.to_d3_json(G)
        r1 = [n for n in data["nodes"] if n["id"] == "R1"][0]
        assert r1["group"] == "router"

    def test_d3_with_scores(self):
        G = _sample_graph()
        scores = {"R1": 0.8, "S1": 0.2}
        data = D3Exporter.to_d3_json(G, node_scores=scores)
        r1 = [n for n in data["nodes"] if n["id"] == "R1"][0]
        assert r1["anomaly_score"] == 0.8

    def test_save(self, tmp_path):
        G = _sample_graph()
        filepath = str(tmp_path / "test.json")
        D3Exporter.save(filepath, G)
        with open(filepath) as f:
            data = json.load(f)
        assert len(data["nodes"]) == 3


class TestMermaidExporter:
    def test_to_mermaid(self):
        G = _sample_graph()
        output = MermaidExporter.to_mermaid(G)
        assert "graph TB" in output
        assert "R1" in output
        assert "S1" in output

    def test_mermaid_direction(self):
        G = _sample_graph()
        output = MermaidExporter.to_mermaid(G, direction="LR")
        assert "graph LR" in output

    def test_mermaid_title(self):
        G = _sample_graph()
        output = MermaidExporter.to_mermaid(G, title="My Topology")
        assert "My Topology" in output


class TestGrafanaExporter:
    def test_to_grafana(self):
        G = _sample_graph()
        config = GrafanaExporter.to_grafana_dashboard(G, title="Test")
        assert "dashboard" in config
        assert config["dashboard"]["title"] == "Test"
        assert len(config["dashboard"]["panels"]) > 0


class TestHTMLExporter:
    def test_to_html(self):
        G = _sample_graph()
        html_str = HTMLExporter.to_html(G)
        assert "<!DOCTYPE html>" in html_str
        assert "d3.js" in html_str or "d3.v7" in html_str
        assert "R1" in html_str

    def test_html_with_scores(self):
        G = _sample_graph()
        scores = {"R1": 0.9}
        html_str = HTMLExporter.to_html(G, node_scores=scores)
        assert "anomaly_score" in html_str

    def test_save(self, tmp_path):
        G = _sample_graph()
        filepath = str(tmp_path / "test.html")
        HTMLExporter.save(filepath, G)
        with open(filepath) as f:
            content = f.read()
        assert len(content) > 1000
