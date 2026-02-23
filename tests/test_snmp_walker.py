"""Tests for SNMP walker."""

import pytest
from netgraph.ingest.snmp_walker import SNMPWalker, SNMPDevice


SAMPLE_WALK = """
1.3.6.1.2.1.1.1.0 = STRING: "Cisco NX-OS n9000"
1.3.6.1.2.1.1.5.0 = STRING: "core-sw1"
1.3.6.1.2.1.1.6.0 = STRING: "DC1 Rack A"
1.3.6.1.2.1.2.2.1.2.1 = STRING: "mgmt0"
1.3.6.1.2.1.2.2.1.2.2 = STRING: "Ethernet1/1"
1.3.6.1.2.1.2.2.1.5.1 = Gauge32: 1000000000
1.3.6.1.2.1.2.2.1.5.2 = Gauge32: 10000000000
1.3.6.1.2.1.2.2.1.7.1 = INTEGER: 1
1.3.6.1.2.1.2.2.1.7.2 = INTEGER: 1
1.3.6.1.2.1.2.2.1.8.1 = INTEGER: 1
1.3.6.1.2.1.2.2.1.8.2 = INTEGER: 1
1.3.6.1.2.1.2.2.1.10.1 = Counter32: 1000
1.3.6.1.2.1.2.2.1.10.2 = Counter32: 5000000
1.3.6.1.2.1.2.2.1.14.1 = Counter32: 0
1.3.6.1.2.1.2.2.1.14.2 = Counter32: 5
1.0.8802.1.1.2.1.4.1.1.9.0.1.1 = STRING: "neighbor-sw1"
1.0.8802.1.1.2.1.4.1.1.7.0.1.1 = STRING: "Gi0/1"
"""


class TestSNMPWalker:
    def test_parse_system_info(self):
        walker = SNMPWalker("unknown")
        device = walker.parse_walk(SAMPLE_WALK)
        assert device.hostname == "core-sw1"
        assert "NX-OS" in device.sys_descr
        assert device.sys_location == "DC1 Rack A"

    def test_parse_interfaces(self):
        walker = SNMPWalker("unknown")
        device = walker.parse_walk(SAMPLE_WALK)
        assert len(device.interfaces) >= 2
        assert 1 in device.interfaces
        assert device.interfaces[1].description == "mgmt0"

    def test_interface_status(self):
        walker = SNMPWalker("unknown")
        device = walker.parse_walk(SAMPLE_WALK)
        assert device.interfaces[1].admin_status == "up"
        assert device.interfaces[2].oper_status == "up"

    def test_interface_counters(self):
        walker = SNMPWalker("unknown")
        device = walker.parse_walk(SAMPLE_WALK)
        assert device.interfaces[2].in_octets == 5000000
        assert device.interfaces[2].in_errors == 5

    def test_lldp_neighbors(self):
        walker = SNMPWalker("unknown")
        device = walker.parse_walk(SAMPLE_WALK)
        assert len(device.lldp_neighbors) == 1
        assert device.lldp_neighbors[0]["system_name"] == "neighbor-sw1"

    def test_to_graph(self):
        walker = SNMPWalker("unknown")
        walker.parse_walk(SAMPLE_WALK)
        G = walker.to_graph()
        assert "core-sw1" in G.nodes()
        assert "neighbor-sw1" in G.nodes()
        assert G.has_edge("core-sw1", "neighbor-sw1")

    def test_interface_stats(self):
        walker = SNMPWalker("unknown")
        walker.parse_walk(SAMPLE_WALK)
        stats = walker.get_interface_stats()
        assert len(stats) >= 2
        assert stats[0]["name"] in ("mgmt0", "Ethernet1/1")

    def test_empty_walk(self):
        walker = SNMPWalker("unknown")
        device = walker.parse_walk("")
        assert device.hostname == "unknown"
        assert len(device.interfaces) == 0
