"""Tests for REST API."""

import pytest
import json
import networkx as nx

try:
    from netgraph.api.routes import NetGraphAPI
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


def _sample_graph():
    G = nx.Graph()
    G.add_node("R1", type="router")
    G.add_node("R2", type="router")
    G.add_node("S1", type="network_device")
    G.add_edge("R1", "R2", protocol="ospf")
    G.add_edge("R2", "S1", protocol="lldp")
    return G


@pytest.fixture
def api_client():
    if not HAS_FLASK:
        pytest.skip("Flask not installed")
    api = NetGraphAPI()
    api.set_graph(_sample_graph())
    app = api.create_app()
    app.config["TESTING"] = True
    return app.test_client()


class TestAPI:
    def test_health(self, api_client):
        resp = api_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["nodes"] == 3

    def test_topology(self, api_client):
        resp = api_client.get("/api/v1/topology")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["nodes"]) == 3
        assert len(data["links"]) == 2

    def test_topology_nodes(self, api_client):
        resp = api_client.get("/api/v1/topology/nodes")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 3

    def test_topology_edges(self, api_client):
        resp = api_client.get("/api/v1/topology/edges")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2

    def test_topology_node_detail(self, api_client):
        resp = api_client.get("/api/v1/topology/node/R1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == "R1"
        assert data["degree"] == 1

    def test_topology_node_not_found(self, api_client):
        resp = api_client.get("/api/v1/topology/node/NONEXISTENT")
        assert resp.status_code == 404

    def test_impact_blast(self, api_client):
        resp = api_client.post(
            "/api/v1/impact/blast",
            data=json.dumps({"node": "R2"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["failed_entity"] == "R2"
