"""Tests for LLDP/CDP parser."""

import pytest
import networkx as nx
from netgraph.ingest.lldp_parser import LLDPParser, LLDPNeighbor


SAMPLE_LLDP = """
--------------------------------------------------------------
Local Port id: GigabitEthernet0/0/1
System Name: switch1
Port id: GigabitEthernet0/0/2
System Description: Cisco IOS Software
Time To Live: 120
System Capabilities: Bridge, Router
Management Addresses: 10.1.1.1

--------------------------------------------------------------
Local Port id: GigabitEthernet0/0/2
System Name: router1
Port id: GigabitEthernet0/0/0
System Description: Cisco IOS XE
Time To Live: 180
System Capabilities: Router
Management Addresses: 10.2.1.1
"""

SAMPLE_CDP = """
-------------------------
Device ID: switch2.example.com
Interface: GigabitEthernet0/1,  Port ID (outgoing port): GigabitEthernet0/2
Platform: cisco WS-C3750X-48PF-S,
Capabilities: Switch IGMP
IP address: 10.1.1.5

-------------------------
Device ID: router2.example.com
Interface: GigabitEthernet0/2,  Port ID (outgoing port): GigabitEthernet0/0/1
Platform: Cisco ISR4451-X/K9,
Capabilities: Router Switch
IP address: 10.2.2.1
"""


class TestLLDPParser:
    def test_parse_lldp_text(self):
        parser = LLDPParser("core-sw1")
        neighbors = parser.parse_lldp_text(SAMPLE_LLDP)
        assert len(neighbors) == 2
        assert neighbors[0].remote_device == "switch1"
        assert neighbors[1].remote_device == "router1"

    def test_parse_lldp_local_port(self):
        parser = LLDPParser("core-sw1")
        neighbors = parser.parse_lldp_text(SAMPLE_LLDP)
        assert neighbors[0].local_port == "GigabitEthernet0/0/1"
        assert neighbors[1].local_port == "GigabitEthernet0/0/2"

    def test_parse_lldp_capabilities(self):
        parser = LLDPParser("core-sw1")
        neighbors = parser.parse_lldp_text(SAMPLE_LLDP)
        assert "Bridge" in neighbors[0].remote_capabilities
        assert "Router" in neighbors[0].remote_capabilities
        assert "Router" in neighbors[1].remote_capabilities

    def test_parse_lldp_ttl(self):
        parser = LLDPParser("core-sw1")
        neighbors = parser.parse_lldp_text(SAMPLE_LLDP)
        assert neighbors[0].ttl == 120
        assert neighbors[1].ttl == 180

    def test_parse_cdp_text(self):
        parser = LLDPParser("core-sw1")
        neighbors = parser.parse_cdp_text(SAMPLE_CDP)
        assert len(neighbors) == 2
        assert neighbors[0].remote_device == "switch2"
        assert neighbors[0].protocol == "cdp"

    def test_parse_cdp_platform(self):
        parser = LLDPParser("core-sw1")
        neighbors = parser.parse_cdp_text(SAMPLE_CDP)
        assert "WS-C3750X" in neighbors[0].remote_system_description

    def test_to_graph(self):
        parser = LLDPParser("core-sw1")
        parser.parse_lldp_text(SAMPLE_LLDP)
        G = parser.to_graph()
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 3  # core-sw1, switch1, router1
        assert G.number_of_edges() == 2

    def test_graph_node_attributes(self):
        parser = LLDPParser("core-sw1")
        parser.parse_lldp_text(SAMPLE_LLDP)
        G = parser.to_graph()
        assert "switch1" in G.nodes()
        assert G.nodes["switch1"]["type"] == "network_device"

    def test_graph_edge_attributes(self):
        parser = LLDPParser("core-sw1")
        parser.parse_lldp_text(SAMPLE_LLDP)
        G = parser.to_graph()
        edge_data = G.edges["core-sw1", "switch1"]
        assert edge_data["local_port"] == "GigabitEthernet0/0/1"
        assert edge_data["protocol"] == "lldp"

    def test_parse_json(self):
        data = {
            "lldp_neighbors": [
                {
                    "local_device": "sw1",
                    "local_port": "Gi0/1",
                    "remote_device": "sw2",
                    "remote_port": "Gi0/2",
                    "protocol": "lldp",
                }
            ]
        }
        parser = LLDPParser("sw1")
        neighbors = parser.parse_json(data)
        assert len(neighbors) == 1
        assert neighbors[0].remote_device == "sw2"

    def test_edge_id_deterministic(self):
        n1 = LLDPNeighbor("A", "p1", "B", "p2")
        n2 = LLDPNeighbor("B", "p2", "A", "p1")
        assert n1.edge_id == n2.edge_id

    def test_empty_input(self):
        parser = LLDPParser("sw1")
        neighbors = parser.parse_lldp_text("")
        assert len(neighbors) == 0
