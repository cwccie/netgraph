"""Tests for interface state collector."""

import pytest
from netgraph.ingest.interface_collector import InterfaceCollector, InterfaceState, DeviceSnapshot


SAMPLE_IOS_INTF = """GigabitEthernet0/0/0 is up, line protocol is up
  Description: Uplink to Core
  Internet address is 10.0.0.2/30
  MTU 1500 bytes, BW 1000000 Kbit
  5 minute input rate 450000 bits/sec, 300 packets/sec
  5 minute output rate 380000 bits/sec, 250 packets/sec
    1000000 packets input, 500000000 bytes
    800000 packets output, 400000000 bytes
    5 input errors, 0 CRC, 0 frame
    2 output errors, 0 collisions

GigabitEthernet0/0/1 is administratively down, line protocol is down
  MTU 1500 bytes, BW 100000 Kbit
  5 minute input rate 0 bits/sec
  5 minute output rate 0 bits/sec
    0 packets input
    0 packets output
    0 input errors
    0 output errors
"""


class TestInterfaceCollector:
    def test_parse_ios(self):
        collector = InterfaceCollector("router1")
        snap = collector.parse_ios_interfaces(SAMPLE_IOS_INTF)
        assert len(snap.interfaces) == 2

    def test_interface_status(self):
        collector = InterfaceCollector("router1")
        snap = collector.parse_ios_interfaces(SAMPLE_IOS_INTF)
        gi0 = snap.interfaces.get("GigabitEthernet0/0/0")
        assert gi0 is not None
        assert gi0.admin_status == "up"
        assert gi0.oper_status == "up"
        assert gi0.is_up

    def test_interface_down(self):
        collector = InterfaceCollector("router1")
        snap = collector.parse_ios_interfaces(SAMPLE_IOS_INTF)
        gi1 = snap.interfaces.get("GigabitEthernet0/0/1")
        assert gi1 is not None
        assert gi1.admin_status == "down"
        assert not gi1.is_up

    def test_interface_speed(self):
        collector = InterfaceCollector("router1")
        snap = collector.parse_ios_interfaces(SAMPLE_IOS_INTF)
        gi0 = snap.interfaces["GigabitEthernet0/0/0"]
        assert gi0.speed_mbps == 1000

    def test_up_down_counts(self):
        collector = InterfaceCollector("router1")
        snap = collector.parse_ios_interfaces(SAMPLE_IOS_INTF)
        assert snap.up_count == 1
        assert snap.down_count == 1

    def test_parse_json(self):
        data = [
            {"name": "Gi0/1", "admin_status": "up", "oper_status": "up", "speed_mbps": 1000},
            {"name": "Gi0/2", "admin_status": "up", "oper_status": "down", "speed_mbps": 1000},
        ]
        collector = InterfaceCollector("router1")
        snap = collector.parse_json(data)
        assert len(snap.interfaces) == 2

    def test_compute_deltas(self):
        collector = InterfaceCollector("router1")
        snap1 = DeviceSnapshot(device="router1")
        snap1.interfaces["Gi0/1"] = InterfaceState(
            device="router1", name="Gi0/1", in_bytes=1000, out_bytes=500, in_errors=0
        )
        snap2 = DeviceSnapshot(device="router1")
        snap2.interfaces["Gi0/1"] = InterfaceState(
            device="router1", name="Gi0/1", in_bytes=2000, out_bytes=1500, in_errors=2
        )
        deltas = collector.compute_deltas(snap1, snap2, interval_sec=300)
        assert deltas["Gi0/1"]["in_bytes_delta"] == 1000
        assert deltas["Gi0/1"]["in_errors_delta"] == 2
