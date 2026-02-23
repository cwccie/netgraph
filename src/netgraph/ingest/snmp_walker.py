"""
SNMP MIB walker for topology discovery.

Parses SNMP walk output to extract device information, interface states,
LLDP-MIB entries, and bridge forwarding tables for topology construction.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx


@dataclass
class SNMPInterface:
    """Represents an SNMP-discovered interface."""
    index: int
    name: str
    description: str = ""
    admin_status: str = "up"
    oper_status: str = "up"
    speed: int = 0  # bps
    mtu: int = 1500
    mac_address: str = ""
    in_octets: int = 0
    out_octets: int = 0
    in_errors: int = 0
    out_errors: int = 0
    in_discards: int = 0
    out_discards: int = 0


@dataclass
class SNMPDevice:
    """Represents an SNMP-discovered device."""
    hostname: str
    sys_descr: str = ""
    sys_object_id: str = ""
    sys_uptime: int = 0
    sys_contact: str = ""
    sys_location: str = ""
    interfaces: dict[int, SNMPInterface] = field(default_factory=dict)
    lldp_neighbors: list[dict] = field(default_factory=list)


# Common SNMP OID prefixes
OID_PREFIXES = {
    "sysDescr": "1.3.6.1.2.1.1.1",
    "sysObjectID": "1.3.6.1.2.1.1.2",
    "sysUpTime": "1.3.6.1.2.1.1.3",
    "sysContact": "1.3.6.1.2.1.1.4",
    "sysName": "1.3.6.1.2.1.1.5",
    "sysLocation": "1.3.6.1.2.1.1.6",
    "ifIndex": "1.3.6.1.2.1.2.2.1.1",
    "ifDescr": "1.3.6.1.2.1.2.2.1.2",
    "ifType": "1.3.6.1.2.1.2.2.1.3",
    "ifMtu": "1.3.6.1.2.1.2.2.1.4",
    "ifSpeed": "1.3.6.1.2.1.2.2.1.5",
    "ifPhysAddress": "1.3.6.1.2.1.2.2.1.6",
    "ifAdminStatus": "1.3.6.1.2.1.2.2.1.7",
    "ifOperStatus": "1.3.6.1.2.1.2.2.1.8",
    "ifInOctets": "1.3.6.1.2.1.2.2.1.10",
    "ifOutOctets": "1.3.6.1.2.1.2.2.1.16",
    "ifInErrors": "1.3.6.1.2.1.2.2.1.14",
    "ifOutErrors": "1.3.6.1.2.1.2.2.1.20",
    "ifInDiscards": "1.3.6.1.2.1.2.2.1.13",
    "ifOutDiscards": "1.3.6.1.2.1.2.2.1.19",
    "ifName": "1.3.6.1.2.1.31.1.1.1.1",
    "ifHighSpeed": "1.3.6.1.2.1.31.1.1.1.15",
    "lldpRemSysName": "1.0.8802.1.1.2.1.4.1.1.9",
    "lldpRemPortId": "1.0.8802.1.1.2.1.4.1.1.7",
    "lldpRemPortDesc": "1.0.8802.1.1.2.1.4.1.1.8",
    "lldpRemSysDesc": "1.0.8802.1.1.2.1.4.1.1.10",
    "lldpRemManAddr": "1.0.8802.1.1.2.1.4.2.1.4",
    "lldpLocPortId": "1.0.8802.1.1.2.1.3.7.1.3",
}

# SNMP walk line pattern: OID = TYPE: VALUE
_WALK_LINE = re.compile(
    r"^([\d.]+)\s*=\s*(?:(\w+):\s*)?(.*)$"
)

_STATUS_MAP = {"1": "up", "2": "down", "3": "testing"}


class SNMPWalker:
    """
    Parse SNMP walk output to extract device and topology information.

    Processes snmpwalk/snmpbulkwalk text output and extracts:
      - System MIB (hostname, description, location, uptime)
      - Interface MIB (names, states, counters, speeds)
      - LLDP-MIB (neighbor relationships)
    """

    def __init__(self, hostname: str = "unknown"):
        self.hostname = hostname
        self.device = SNMPDevice(hostname=hostname)
        self._raw: dict[str, str] = {}

    def parse_walk(self, text: str) -> SNMPDevice:
        """Parse complete SNMP walk text output."""
        # First pass: extract all OID -> value mappings
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            m = _WALK_LINE.match(line)
            if not m:
                continue

            oid = m.group(1)
            value = m.group(3).strip().strip('"')
            self._raw[oid] = value

        # Parse system info
        self._parse_system()
        # Parse interfaces
        self._parse_interfaces()
        # Parse LLDP neighbors
        self._parse_lldp()

        return self.device

    def _parse_system(self):
        """Extract system MIB values."""
        for oid, value in self._raw.items():
            if oid.startswith(OID_PREFIXES["sysDescr"]):
                self.device.sys_descr = value
            elif oid.startswith(OID_PREFIXES["sysName"]):
                self.device.hostname = value
                self.hostname = value
            elif oid.startswith(OID_PREFIXES["sysLocation"]):
                self.device.sys_location = value
            elif oid.startswith(OID_PREFIXES["sysContact"]):
                self.device.sys_contact = value
            elif oid.startswith(OID_PREFIXES["sysUpTime"]):
                try:
                    self.device.sys_uptime = int(value)
                except ValueError:
                    pass
            elif oid.startswith(OID_PREFIXES["sysObjectID"]):
                self.device.sys_object_id = value

    def _parse_interfaces(self):
        """Extract interface MIB values."""
        # Get interface indices from ifDescr
        for oid, value in self._raw.items():
            if oid.startswith(OID_PREFIXES["ifDescr"] + "."):
                idx = int(oid.split(".")[-1])
                if idx not in self.device.interfaces:
                    self.device.interfaces[idx] = SNMPInterface(
                        index=idx, name=value
                    )
                self.device.interfaces[idx].description = value

            elif oid.startswith(OID_PREFIXES["ifName"] + "."):
                idx = int(oid.split(".")[-1])
                if idx not in self.device.interfaces:
                    self.device.interfaces[idx] = SNMPInterface(
                        index=idx, name=value
                    )
                self.device.interfaces[idx].name = value

        # Fill in interface details
        for oid, value in self._raw.items():
            parts = oid.split(".")
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[-1])
            except ValueError:
                continue

            if idx not in self.device.interfaces:
                continue

            iface = self.device.interfaces[idx]

            if oid.startswith(OID_PREFIXES["ifAdminStatus"]):
                iface.admin_status = _STATUS_MAP.get(value, value)
            elif oid.startswith(OID_PREFIXES["ifOperStatus"]):
                iface.oper_status = _STATUS_MAP.get(value, value)
            elif oid.startswith(OID_PREFIXES["ifSpeed"]):
                try:
                    iface.speed = int(value)
                except ValueError:
                    pass
            elif oid.startswith(OID_PREFIXES["ifHighSpeed"]):
                try:
                    iface.speed = int(value) * 1_000_000
                except ValueError:
                    pass
            elif oid.startswith(OID_PREFIXES["ifMtu"]):
                try:
                    iface.mtu = int(value)
                except ValueError:
                    pass
            elif oid.startswith(OID_PREFIXES["ifPhysAddress"]):
                iface.mac_address = value
            elif oid.startswith(OID_PREFIXES["ifInOctets"]):
                try:
                    iface.in_octets = int(value)
                except ValueError:
                    pass
            elif oid.startswith(OID_PREFIXES["ifOutOctets"]):
                try:
                    iface.out_octets = int(value)
                except ValueError:
                    pass
            elif oid.startswith(OID_PREFIXES["ifInErrors"]):
                try:
                    iface.in_errors = int(value)
                except ValueError:
                    pass
            elif oid.startswith(OID_PREFIXES["ifOutErrors"]):
                try:
                    iface.out_errors = int(value)
                except ValueError:
                    pass

    def _parse_lldp(self):
        """Extract LLDP-MIB neighbor entries."""
        # Group LLDP entries by neighbor index
        lldp_data: dict[str, dict] = {}

        for oid, value in self._raw.items():
            if oid.startswith(OID_PREFIXES["lldpRemSysName"]):
                key = oid[len(OID_PREFIXES["lldpRemSysName"]):]
                lldp_data.setdefault(key, {})["system_name"] = value
            elif oid.startswith(OID_PREFIXES["lldpRemPortId"]):
                key = oid[len(OID_PREFIXES["lldpRemPortId"]):]
                lldp_data.setdefault(key, {})["port_id"] = value
            elif oid.startswith(OID_PREFIXES["lldpRemPortDesc"]):
                key = oid[len(OID_PREFIXES["lldpRemPortDesc"]):]
                lldp_data.setdefault(key, {})["port_desc"] = value
            elif oid.startswith(OID_PREFIXES["lldpRemSysDesc"]):
                key = oid[len(OID_PREFIXES["lldpRemSysDesc"]):]
                lldp_data.setdefault(key, {})["system_desc"] = value

        for key, data in lldp_data.items():
            if "system_name" in data:
                self.device.lldp_neighbors.append(data)

    def to_graph(self) -> nx.Graph:
        """Build a NetworkX graph from the parsed SNMP data."""
        G = nx.Graph()

        # Add local device node
        G.add_node(
            self.hostname,
            type="network_device",
            sys_descr=self.device.sys_descr,
            sys_location=self.device.sys_location,
            uptime=self.device.sys_uptime,
            interface_count=len(self.device.interfaces),
        )

        # Add edges from LLDP neighbors
        for nbr in self.device.lldp_neighbors:
            remote = nbr.get("system_name", "unknown")
            G.add_node(remote, type="network_device")
            G.add_edge(
                self.hostname,
                remote,
                remote_port=nbr.get("port_id", ""),
                protocol="lldp-mib",
            )

        return G

    def get_interface_stats(self) -> list[dict]:
        """Return interface statistics as a list of dicts."""
        stats = []
        for idx, iface in sorted(self.device.interfaces.items()):
            stats.append({
                "index": idx,
                "name": iface.name,
                "admin_status": iface.admin_status,
                "oper_status": iface.oper_status,
                "speed": iface.speed,
                "in_octets": iface.in_octets,
                "out_octets": iface.out_octets,
                "in_errors": iface.in_errors,
                "out_errors": iface.out_errors,
                "utilization": _calc_utilization(iface),
            })
        return stats


def _calc_utilization(iface: SNMPInterface) -> float:
    """Estimate interface utilization (0.0-1.0) from counters."""
    if iface.speed <= 0:
        return 0.0
    total_octets = iface.in_octets + iface.out_octets
    total_bits = total_octets * 8
    # This is a snapshot, not rate — callers should diff two snapshots
    return min(total_bits / max(iface.speed, 1), 1.0)


def parse_snmp_walk_file(filepath: str, hostname: str = "unknown") -> SNMPDevice:
    """Convenience: parse an SNMP walk file."""
    walker = SNMPWalker(hostname)
    with open(filepath) as f:
        text = f.read()
    return walker.parse_walk(text)
