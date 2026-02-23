"""
Interface state collector.

Collects and normalizes interface state data from network devices,
including operational status, counters, errors, and utilization metrics.
Supports merging data from multiple collection sources.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx


@dataclass
class InterfaceState:
    """Point-in-time interface state snapshot."""
    device: str
    name: str
    admin_status: str = "up"
    oper_status: str = "up"
    speed_mbps: int = 0
    duplex: str = "full"
    mtu: int = 1500
    ip_address: str = ""
    subnet_mask: str = ""
    mac_address: str = ""
    vlan: int = 0
    description: str = ""
    # Counters
    in_bytes: int = 0
    out_bytes: int = 0
    in_packets: int = 0
    out_packets: int = 0
    in_errors: int = 0
    out_errors: int = 0
    in_discards: int = 0
    out_discards: int = 0
    crc_errors: int = 0
    collisions: int = 0
    # Derived
    input_rate_bps: float = 0.0
    output_rate_bps: float = 0.0
    # Metadata
    timestamp: float = 0.0
    collection_method: str = "cli"

    @property
    def is_up(self) -> bool:
        return self.admin_status == "up" and self.oper_status == "up"

    @property
    def utilization_in(self) -> float:
        if self.speed_mbps <= 0:
            return 0.0
        return min(self.input_rate_bps / (self.speed_mbps * 1_000_000) * 100, 100.0)

    @property
    def utilization_out(self) -> float:
        if self.speed_mbps <= 0:
            return 0.0
        return min(self.output_rate_bps / (self.speed_mbps * 1_000_000) * 100, 100.0)

    @property
    def error_rate(self) -> float:
        total = self.in_packets + self.out_packets
        if total == 0:
            return 0.0
        return (self.in_errors + self.out_errors) / total


@dataclass
class DeviceSnapshot:
    """Complete interface state snapshot for a device."""
    device: str
    timestamp: float = field(default_factory=time.time)
    interfaces: dict[str, InterfaceState] = field(default_factory=dict)

    @property
    def up_count(self) -> int:
        return sum(1 for i in self.interfaces.values() if i.is_up)

    @property
    def down_count(self) -> int:
        return sum(1 for i in self.interfaces.values() if not i.is_up)

    @property
    def total_errors(self) -> int:
        return sum(
            i.in_errors + i.out_errors for i in self.interfaces.values()
        )


# IOS 'show interfaces' patterns
_IOS_INTF_HEADER = re.compile(
    r"^(\S+)\s+is\s+(administratively\s+)?(up|down),\s+"
    r"line\s+protocol\s+is\s+(up|down)"
)
_IOS_SPEED = re.compile(r"BW\s+(\d+)\s+Kbit")
_IOS_MTU = re.compile(r"MTU\s+(\d+)\s+bytes")
_IOS_MAC = re.compile(r"(?:Hardware|address)\s+is\s+([0-9a-fA-F.]+)")
_IOS_IP = re.compile(r"Internet\s+address\s+is\s+(\d+\.\d+\.\d+\.\d+)/(\d+)")
_IOS_INPUT_RATE = re.compile(r"input\s+rate\s+(\d+)\s+bits/sec")
_IOS_OUTPUT_RATE = re.compile(r"output\s+rate\s+(\d+)\s+bits/sec")
_IOS_IN_PACKETS = re.compile(r"(\d+)\s+packets\s+input")
_IOS_OUT_PACKETS = re.compile(r"(\d+)\s+packets\s+output")
_IOS_IN_ERRORS = re.compile(r"(\d+)\s+input\s+errors")
_IOS_OUT_ERRORS = re.compile(r"(\d+)\s+output\s+errors")
_IOS_CRC = re.compile(r"(\d+)\s+CRC")
_IOS_COLLISIONS = re.compile(r"(\d+)\s+collisions")
_IOS_DESCRIPTION = re.compile(r"Description:\s+(.+)")


class InterfaceCollector:
    """
    Collect and normalize interface state from network devices.

    Supports:
      - Cisco IOS 'show interfaces' text output
      - JSON structured interface data
      - Dict-based programmatic input
    """

    def __init__(self, device_name: str = "unknown"):
        self.device_name = device_name
        self.snapshots: list[DeviceSnapshot] = []

    def parse_ios_interfaces(self, text: str) -> DeviceSnapshot:
        """Parse Cisco IOS 'show interfaces' output."""
        snapshot = DeviceSnapshot(device=self.device_name)

        # Split into per-interface blocks
        blocks = re.split(r"(?=^\S+\s+is\s+)", text, flags=re.MULTILINE)

        for block in blocks:
            if not block.strip():
                continue

            m = _IOS_INTF_HEADER.match(block)
            if not m:
                continue

            name = m.group(1)
            admin = "down" if m.group(2) else "up"
            oper = m.group(4)

            iface = InterfaceState(
                device=self.device_name,
                name=name,
                admin_status=admin,
                oper_status=oper,
                timestamp=time.time(),
                collection_method="cli",
            )

            # Extract fields
            self._extract_int(block, _IOS_SPEED, iface, "speed_mbps",
                              transform=lambda x: x // 1000)
            self._extract_int(block, _IOS_MTU, iface, "mtu")
            self._extract_str(block, _IOS_MAC, iface, "mac_address")
            self._extract_str(block, _IOS_DESCRIPTION, iface, "description")
            self._extract_float(block, _IOS_INPUT_RATE, iface, "input_rate_bps")
            self._extract_float(block, _IOS_OUTPUT_RATE, iface, "output_rate_bps")
            self._extract_int(block, _IOS_IN_PACKETS, iface, "in_packets")
            self._extract_int(block, _IOS_OUT_PACKETS, iface, "out_packets")
            self._extract_int(block, _IOS_IN_ERRORS, iface, "in_errors")
            self._extract_int(block, _IOS_OUT_ERRORS, iface, "out_errors")
            self._extract_int(block, _IOS_CRC, iface, "crc_errors")
            self._extract_int(block, _IOS_COLLISIONS, iface, "collisions")

            ip_m = _IOS_IP.search(block)
            if ip_m:
                iface.ip_address = ip_m.group(1)
                iface.subnet_mask = ip_m.group(2)

            snapshot.interfaces[name] = iface

        self.snapshots.append(snapshot)
        return snapshot

    def parse_json(self, data: dict | list | str) -> DeviceSnapshot:
        """Parse JSON interface data."""
        import json as _json
        if isinstance(data, str):
            data = _json.loads(data)

        snapshot = DeviceSnapshot(device=self.device_name)

        entries = data if isinstance(data, list) else data.get("interfaces", [])

        for entry in entries:
            name = entry.get("name", entry.get("interface", ""))
            iface = InterfaceState(
                device=self.device_name,
                name=name,
                admin_status=entry.get("admin_status", "up"),
                oper_status=entry.get("oper_status", "up"),
                speed_mbps=entry.get("speed_mbps", entry.get("speed", 0)),
                mtu=entry.get("mtu", 1500),
                ip_address=entry.get("ip_address", ""),
                mac_address=entry.get("mac_address", ""),
                description=entry.get("description", ""),
                in_bytes=entry.get("in_bytes", 0),
                out_bytes=entry.get("out_bytes", 0),
                in_errors=entry.get("in_errors", 0),
                out_errors=entry.get("out_errors", 0),
                input_rate_bps=entry.get("input_rate_bps", 0),
                output_rate_bps=entry.get("output_rate_bps", 0),
                timestamp=entry.get("timestamp", time.time()),
                collection_method="json",
            )
            snapshot.interfaces[name] = iface

        self.snapshots.append(snapshot)
        return snapshot

    def annotate_graph(
        self, G: nx.Graph, snapshot: Optional[DeviceSnapshot] = None
    ) -> nx.Graph:
        """Add interface state data as edge attributes to an existing graph."""
        if snapshot is None and self.snapshots:
            snapshot = self.snapshots[-1]
        if snapshot is None:
            return G

        for u, v, data in G.edges(data=True):
            local_port = data.get("local_port", "")
            if local_port in snapshot.interfaces:
                iface = snapshot.interfaces[local_port]
                data["oper_status"] = iface.oper_status
                data["speed_mbps"] = iface.speed_mbps
                data["utilization_in"] = iface.utilization_in
                data["utilization_out"] = iface.utilization_out
                data["error_rate"] = iface.error_rate
                data["input_rate_bps"] = iface.input_rate_bps
                data["output_rate_bps"] = iface.output_rate_bps

        return G

    @staticmethod
    def _extract_int(block, pattern, obj, attr, transform=None):
        m = pattern.search(block)
        if m:
            val = int(m.group(1))
            if transform:
                val = transform(val)
            setattr(obj, attr, val)

    @staticmethod
    def _extract_float(block, pattern, obj, attr):
        m = pattern.search(block)
        if m:
            setattr(obj, attr, float(m.group(1)))

    @staticmethod
    def _extract_str(block, pattern, obj, attr):
        m = pattern.search(block)
        if m:
            setattr(obj, attr, m.group(1).strip())

    def compute_deltas(
        self,
        older: DeviceSnapshot,
        newer: DeviceSnapshot,
        interval_sec: float = 300.0,
    ) -> dict[str, dict]:
        """Compute counter deltas between two snapshots."""
        deltas = {}
        for name in newer.interfaces:
            if name not in older.interfaces:
                continue

            old = older.interfaces[name]
            new = newer.interfaces[name]

            delta_in = max(new.in_bytes - old.in_bytes, 0)
            delta_out = max(new.out_bytes - old.out_bytes, 0)

            deltas[name] = {
                "in_bytes_delta": delta_in,
                "out_bytes_delta": delta_out,
                "in_rate_bps": (delta_in * 8) / interval_sec if interval_sec > 0 else 0,
                "out_rate_bps": (delta_out * 8) / interval_sec if interval_sec > 0 else 0,
                "in_errors_delta": max(new.in_errors - old.in_errors, 0),
                "out_errors_delta": max(new.out_errors - old.out_errors, 0),
                "status_changed": old.oper_status != new.oper_status,
            }

        return deltas
