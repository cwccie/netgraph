"""
Routing table parser for topology discovery.

Parses routing table output (show ip route) from Cisco IOS/IOS-XE/NX-OS,
Juniper JunOS, and Arista EOS to extract next-hop relationships and
build Layer 3 topology graphs.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx


@dataclass
class Route:
    """Represents a single routing table entry."""
    prefix: str
    mask_len: int
    protocol: str  # O, B, S, C, D, R, etc.
    next_hop: str = ""
    next_hop_interface: str = ""
    metric: int = 0
    admin_distance: int = 0
    age: str = ""
    tag: int = 0
    source_router: str = ""

    @property
    def network(self) -> str:
        return f"{self.prefix}/{self.mask_len}"


# Protocol code mapping
PROTOCOL_CODES = {
    "C": "connected",
    "S": "static",
    "R": "rip",
    "B": "bgp",
    "D": "eigrp",
    "EX": "eigrp-external",
    "O": "ospf",
    "IA": "ospf-inter-area",
    "N1": "ospf-nssa-type1",
    "N2": "ospf-nssa-type2",
    "E1": "ospf-external-type1",
    "E2": "ospf-external-type2",
    "i": "isis",
    "L1": "isis-level1",
    "L2": "isis-level2",
    "ia": "isis-inter-area",
    "L": "local",
}

# IOS route line patterns
_IOS_ROUTE = re.compile(
    r"^([A-Z*\s]{1,5})\s+"     # protocol code
    r"(\d+\.\d+\.\d+\.\d+)"     # prefix
    r"(?:/(\d+))?\s*"            # optional CIDR mask
    r"(?:\[(\d+)/(\d+)\])?\s*"   # [AD/metric]
    r"(?:via\s+(\d+\.\d+\.\d+\.\d+))?\s*"  # next-hop
    r"(?:,\s*(.+))?"             # interface/age
)

# Alternate: subnet under a parent route
_IOS_SUBNET = re.compile(
    r"^\s+(\d+\.\d+\.\d+\.\d+)(?:/(\d+))?\s+"
    r"(?:\[(\d+)/(\d+)\])?\s*"
    r"(?:via\s+(\d+\.\d+\.\d+\.\d+))?\s*"
    r"(?:,\s*(.+))?"
)

# Junos route pattern
_JUNOS_ROUTE = re.compile(
    r"^(\d+\.\d+\.\d+\.\d+/\d+)\s+"
    r"\*?\[(\w+)/(\d+)\]\s+"
    r"(?:\d+[dhms]+\s+ago\s+)?"
    r"(?:>\s*)?(?:to\s+)?(\d+\.\d+\.\d+\.\d+)?\s*"
    r"(?:via\s+(\S+))?"
)


class RoutingTableParser:
    """
    Parse routing table output from network devices.

    Supports:
      - Cisco IOS/IOS-XE 'show ip route' format
      - Cisco NX-OS 'show ip route' format
      - Juniper JunOS 'show route' format
      - JSON structured format
    """

    def __init__(self, device_name: str = "unknown"):
        self.device_name = device_name
        self.routes: list[Route] = []

    def parse_ios(self, text: str) -> list[Route]:
        """Parse Cisco IOS 'show ip route' output."""
        routes = []
        current_proto = ""
        current_mask = 24

        for line in text.splitlines():
            line = line.rstrip()
            if not line or line.startswith("Codes:") or line.startswith("Gateway"):
                continue

            # Try full route line
            m = _IOS_ROUTE.match(line)
            if m:
                proto_code = m.group(1).strip()
                prefix = m.group(2)
                mask = int(m.group(3)) if m.group(3) else current_mask
                ad = int(m.group(4)) if m.group(4) else 0
                metric = int(m.group(5)) if m.group(5) else 0
                next_hop = m.group(6) or ""
                rest = m.group(7) or ""

                current_proto = proto_code
                current_mask = mask

                iface = ""
                if rest:
                    # Extract interface name
                    iface_m = re.search(
                        r"((?:Gi|Te|Fa|Et|Vl|Lo|Po|Hu)\S+)", rest
                    )
                    if iface_m:
                        iface = iface_m.group(1)

                route = Route(
                    prefix=prefix,
                    mask_len=mask,
                    protocol=PROTOCOL_CODES.get(proto_code, proto_code),
                    next_hop=next_hop,
                    next_hop_interface=iface,
                    metric=metric,
                    admin_distance=ad,
                    source_router=self.device_name,
                )
                routes.append(route)
                continue

            # Try subnet line (indented)
            m = _IOS_SUBNET.match(line)
            if m:
                prefix = m.group(1)
                mask = int(m.group(2)) if m.group(2) else current_mask
                ad = int(m.group(3)) if m.group(3) else 0
                metric = int(m.group(4)) if m.group(4) else 0
                next_hop = m.group(5) or ""
                rest = m.group(6) or ""

                iface = ""
                if rest:
                    iface_m = re.search(
                        r"((?:Gi|Te|Fa|Et|Vl|Lo|Po|Hu)\S+)", rest
                    )
                    if iface_m:
                        iface = iface_m.group(1)

                route = Route(
                    prefix=prefix,
                    mask_len=mask,
                    protocol=PROTOCOL_CODES.get(current_proto, current_proto),
                    next_hop=next_hop,
                    next_hop_interface=iface,
                    metric=metric,
                    admin_distance=ad,
                    source_router=self.device_name,
                )
                routes.append(route)

        self.routes.extend(routes)
        return routes

    def parse_junos(self, text: str) -> list[Route]:
        """Parse Juniper JunOS 'show route' output."""
        routes = []

        for line in text.splitlines():
            m = _JUNOS_ROUTE.match(line.strip())
            if not m:
                continue

            network = m.group(1)
            prefix, mask = network.split("/")
            proto = m.group(2)
            metric = int(m.group(3))
            next_hop = m.group(4) or ""
            iface = m.group(5) or ""

            route = Route(
                prefix=prefix,
                mask_len=int(mask),
                protocol=proto.lower(),
                next_hop=next_hop,
                next_hop_interface=iface,
                metric=metric,
                source_router=self.device_name,
            )
            routes.append(route)

        self.routes.extend(routes)
        return routes

    def parse_json(self, data: dict | list | str) -> list[Route]:
        """Parse JSON routing table data."""
        import json as _json
        if isinstance(data, str):
            data = _json.loads(data)

        entries = data if isinstance(data, list) else data.get("routes", [])

        routes = []
        for entry in entries:
            prefix = entry.get("prefix", "")
            mask = entry.get("mask_len", entry.get("prefix_length", 24))

            if "/" in prefix:
                prefix, mask = prefix.split("/")
                mask = int(mask)

            route = Route(
                prefix=prefix,
                mask_len=int(mask),
                protocol=entry.get("protocol", "unknown"),
                next_hop=entry.get("next_hop", entry.get("nexthop", "")),
                next_hop_interface=entry.get("interface", ""),
                metric=entry.get("metric", 0),
                admin_distance=entry.get("admin_distance", entry.get("preference", 0)),
                source_router=entry.get("source", self.device_name),
            )
            routes.append(route)

        self.routes.extend(routes)
        return routes

    def to_graph(self) -> nx.DiGraph:
        """Build a directed graph from routing relationships.

        Nodes are devices (identified by management IPs or names).
        Edges represent next-hop relationships.
        """
        G = nx.DiGraph()

        G.add_node(self.device_name, type="router")

        for route in self.routes:
            if route.next_hop and route.protocol != "connected":
                # Add next-hop as a node (it's another router)
                nh_name = route.next_hop
                G.add_node(nh_name, type="next_hop")

                # Edge from this device toward the next-hop
                G.add_edge(
                    self.device_name,
                    nh_name,
                    prefix=route.network,
                    protocol=route.protocol,
                    metric=route.metric,
                    admin_distance=route.admin_distance,
                    interface=route.next_hop_interface,
                )

        return G

    def get_protocol_summary(self) -> dict[str, int]:
        """Count routes by protocol."""
        summary: dict[str, int] = {}
        for route in self.routes:
            summary[route.protocol] = summary.get(route.protocol, 0) + 1
        return summary


def parse_routing_file(
    filepath: str,
    device_name: str = "unknown",
    format: str = "ios",
) -> list[Route]:
    """Convenience: parse a routing table file."""
    parser = RoutingTableParser(device_name)
    with open(filepath) as f:
        text = f.read()

    if format == "ios":
        return parser.parse_ios(text)
    elif format == "junos":
        return parser.parse_junos(text)
    elif format == "json":
        return parser.parse_json(text)
    else:
        raise ValueError(f"Unsupported format: {format}")
