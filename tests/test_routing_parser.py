"""Tests for routing table parser."""

import pytest
from netgraph.ingest.routing_parser import RoutingTableParser, Route


SAMPLE_IOS = """
Codes: C - connected, S - static, O - OSPF, B - BGP

      10.0.0.0/8 is variably subnetted
C        10.0.0.0/30 is directly connected, GigabitEthernet0/0/0
L        10.0.0.2/32 is directly connected, GigabitEthernet0/0/0
O        10.1.2.0/24 [110/2] via 10.0.0.1, 00:45:12, GigabitEthernet0/0/0
O IA     10.2.0.0/16 [110/20] via 10.0.0.1, 01:12:33, GigabitEthernet0/0/0
B        172.16.0.0/16 [20/0] via 10.2.1.1, 02:30:15
S*       0.0.0.0/0 [1/0] via 10.0.0.1
"""


class TestRoutingParser:
    def test_parse_ios_connected(self):
        parser = RoutingTableParser("router1")
        routes = parser.parse_ios(SAMPLE_IOS)
        connected = [r for r in routes if r.protocol == "connected"]
        assert len(connected) >= 1

    def test_parse_ios_ospf(self):
        parser = RoutingTableParser("router1")
        routes = parser.parse_ios(SAMPLE_IOS)
        ospf = [r for r in routes if "ospf" in r.protocol]
        assert len(ospf) >= 1
        ospf_route = ospf[0]
        assert ospf_route.next_hop == "10.0.0.1"

    def test_parse_ios_bgp(self):
        parser = RoutingTableParser("router1")
        routes = parser.parse_ios(SAMPLE_IOS)
        bgp = [r for r in routes if r.protocol == "bgp"]
        assert len(bgp) >= 1
        assert bgp[0].next_hop == "10.2.1.1"

    def test_route_network(self):
        r = Route(prefix="10.1.2.0", mask_len=24, protocol="ospf")
        assert r.network == "10.1.2.0/24"

    def test_to_graph(self):
        parser = RoutingTableParser("router1")
        parser.parse_ios(SAMPLE_IOS)
        G = parser.to_graph()
        assert "router1" in G.nodes()
        assert G.number_of_edges() > 0

    def test_protocol_summary(self):
        parser = RoutingTableParser("router1")
        parser.parse_ios(SAMPLE_IOS)
        summary = parser.get_protocol_summary()
        assert "connected" in summary or "ospf" in summary

    def test_parse_json(self):
        data = [
            {"prefix": "10.0.0.0/24", "protocol": "ospf", "next_hop": "10.0.0.1"},
            {"prefix": "172.16.0.0/16", "protocol": "bgp", "next_hop": "10.2.1.1"},
        ]
        parser = RoutingTableParser("router1")
        routes = parser.parse_json(data)
        assert len(routes) == 2
        assert routes[0].protocol == "ospf"

    def test_empty_input(self):
        parser = RoutingTableParser("router1")
        routes = parser.parse_ios("")
        assert len(routes) == 0
