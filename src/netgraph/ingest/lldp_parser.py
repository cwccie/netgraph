"""
LLDP/CDP topology parser.

Parses LLDP (Link Layer Discovery Protocol) and CDP (Cisco Discovery Protocol)
output from network devices to discover neighbors and build topology edges.
Supports raw CLI output, JSON structured data, and XML formats.
"""

import re
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx


@dataclass
class LLDPNeighbor:
    """Represents a single LLDP/CDP neighbor relationship."""
    local_device: str
    local_port: str
    remote_device: str
    remote_port: str
    remote_system_description: str = ""
    remote_capabilities: list = field(default_factory=list)
    remote_mgmt_address: str = ""
    ttl: int = 120
    protocol: str = "lldp"

    @property
    def edge_id(self) -> str:
        """Deterministic edge identifier."""
        parts = sorted([
            f"{self.local_device}:{self.local_port}",
            f"{self.remote_device}:{self.remote_port}",
        ])
        return hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]


class LLDPParser:
    """
    Parse LLDP/CDP neighbor data from various formats.

    Supports:
      - Raw CLI text output (show lldp neighbors detail)
      - JSON structured output (modern NOS)
      - CDP text output (Cisco IOS)
      - Custom dict format for programmatic use
    """

    # LLDP CLI patterns
    _LLDP_LOCAL_PORT = re.compile(
        r"(?:Local\s+(?:Port|Intf)\s*(?:id)?)\s*:\s*(.+)", re.IGNORECASE
    )
    _LLDP_REMOTE_DEVICE = re.compile(
        r"(?:System\s+Name|SysName)\s*:\s*(.+)", re.IGNORECASE
    )
    _LLDP_REMOTE_PORT = re.compile(
        r"(?:Port\s+(?:id|Description))\s*:\s*(.+)", re.IGNORECASE
    )
    _LLDP_REMOTE_DESC = re.compile(
        r"System\s+Description\s*:\s*(.+)", re.IGNORECASE
    )
    _LLDP_MGMT_ADDR = re.compile(
        r"Management\s+Address(?:es)?\s*:\s*(.+)", re.IGNORECASE
    )
    _LLDP_CAPABILITY = re.compile(
        r"System\s+Capabilities\s*:\s*(.+)", re.IGNORECASE
    )
    _LLDP_TTL = re.compile(r"Time\s+To\s+Live\s*:\s*(\d+)", re.IGNORECASE)

    # CDP CLI patterns
    _CDP_DEVICE_ID = re.compile(
        r"Device\s+ID\s*:\s*(.+)", re.IGNORECASE
    )
    _CDP_LOCAL_PORT = re.compile(
        r"(?:Interface|Local\s+Intrfce)\s*:\s*(\S+)", re.IGNORECASE
    )
    _CDP_REMOTE_PORT = re.compile(
        r"Port\s+ID\s*\(outgoing\s+port\)\s*:\s*(.+)", re.IGNORECASE
    )
    _CDP_PLATFORM = re.compile(
        r"Platform\s*:\s*(.+?)(?:,|$)", re.IGNORECASE
    )
    _CDP_MGMT_ADDR = re.compile(
        r"(?:IP\s+address|Entry\s+address)\s*:\s*(\S+)", re.IGNORECASE
    )
    _CDP_CAPABILITY = re.compile(
        r"Capabilities\s*:\s*(.+)", re.IGNORECASE
    )

    def __init__(self, local_device: str = "unknown"):
        self.local_device = local_device
        self.neighbors: list[LLDPNeighbor] = []

    def parse_lldp_text(self, text: str) -> list[LLDPNeighbor]:
        """Parse 'show lldp neighbors detail' output."""
        neighbors = []
        # Split on separator lines or double newlines indicating new neighbor
        blocks = re.split(r"-{20,}|\n\n(?=Local\s)", text)

        for block in blocks:
            if not block.strip():
                continue

            local_port = self._extract(self._LLDP_LOCAL_PORT, block)
            remote_device = self._extract(self._LLDP_REMOTE_DEVICE, block)
            remote_port = self._extract(self._LLDP_REMOTE_PORT, block)

            if not (local_port and remote_device):
                continue

            neighbor = LLDPNeighbor(
                local_device=self.local_device,
                local_port=local_port,
                remote_device=remote_device,
                remote_port=remote_port or "unknown",
                remote_system_description=self._extract(
                    self._LLDP_REMOTE_DESC, block
                ) or "",
                remote_mgmt_address=self._extract(
                    self._LLDP_MGMT_ADDR, block
                ) or "",
                remote_capabilities=self._parse_capabilities(
                    self._extract(self._LLDP_CAPABILITY, block) or ""
                ),
                ttl=int(self._extract(self._LLDP_TTL, block) or "120"),
                protocol="lldp",
            )
            neighbors.append(neighbor)

        self.neighbors.extend(neighbors)
        return neighbors

    def parse_cdp_text(self, text: str) -> list[LLDPNeighbor]:
        """Parse 'show cdp neighbors detail' output."""
        neighbors = []
        blocks = re.split(r"-{20,}", text)

        for block in blocks:
            if not block.strip():
                continue

            remote_device = self._extract(self._CDP_DEVICE_ID, block)
            local_port = self._extract(self._CDP_LOCAL_PORT, block)
            remote_port = self._extract(self._CDP_REMOTE_PORT, block)

            if not (remote_device and local_port):
                continue

            # Clean device ID (remove domain)
            remote_device = remote_device.split(".")[0].strip()

            neighbor = LLDPNeighbor(
                local_device=self.local_device,
                local_port=local_port.strip().rstrip(","),
                remote_device=remote_device,
                remote_port=remote_port or "unknown",
                remote_system_description=self._extract(
                    self._CDP_PLATFORM, block
                ) or "",
                remote_mgmt_address=self._extract(
                    self._CDP_MGMT_ADDR, block
                ) or "",
                remote_capabilities=self._parse_capabilities(
                    self._extract(self._CDP_CAPABILITY, block) or ""
                ),
                protocol="cdp",
            )
            neighbors.append(neighbor)

        self.neighbors.extend(neighbors)
        return neighbors

    def parse_json(self, data: dict | str) -> list[LLDPNeighbor]:
        """Parse JSON-structured LLDP data (e.g., from NX-OS, EOS, OpenConfig)."""
        if isinstance(data, str):
            data = json.loads(data)

        neighbors = []
        # Support multiple JSON formats
        entries = data.get("lldp_neighbors", data.get("neighbors", []))
        if isinstance(data, list):
            entries = data

        for entry in entries:
            neighbor = LLDPNeighbor(
                local_device=entry.get("local_device", self.local_device),
                local_port=entry.get("local_port", entry.get("local_interface", "")),
                remote_device=entry.get(
                    "remote_device",
                    entry.get("neighbor", entry.get("system_name", "")),
                ),
                remote_port=entry.get(
                    "remote_port", entry.get("neighbor_interface", "")
                ),
                remote_system_description=entry.get("system_description", ""),
                remote_mgmt_address=entry.get("mgmt_address", ""),
                remote_capabilities=entry.get("capabilities", []),
                ttl=entry.get("ttl", 120),
                protocol=entry.get("protocol", "lldp"),
            )
            neighbors.append(neighbor)

        self.neighbors.extend(neighbors)
        return neighbors

    def to_graph(self) -> nx.Graph:
        """Convert all parsed neighbors to a NetworkX graph."""
        G = nx.Graph()

        for n in self.neighbors:
            # Add nodes with attributes
            G.add_node(
                n.local_device,
                type="network_device",
            )
            G.add_node(
                n.remote_device,
                type="network_device",
                description=n.remote_system_description,
                mgmt_address=n.remote_mgmt_address,
                capabilities=n.remote_capabilities,
            )

            # Add edge with interface details
            G.add_edge(
                n.local_device,
                n.remote_device,
                local_port=n.local_port,
                remote_port=n.remote_port,
                protocol=n.protocol,
                edge_id=n.edge_id,
                ttl=n.ttl,
            )

        return G

    @staticmethod
    def _extract(pattern: re.Pattern, text: str) -> Optional[str]:
        m = pattern.search(text)
        return m.group(1).strip() if m else None

    @staticmethod
    def _parse_capabilities(cap_str: str) -> list[str]:
        caps = []
        for cap in ["Bridge", "Router", "Station", "Telephone", "WLAN", "Repeater"]:
            if cap.lower() in cap_str.lower():
                caps.append(cap)
        return caps


def parse_lldp_file(filepath: str, local_device: str = "unknown") -> nx.Graph:
    """Convenience: parse an LLDP text file and return a graph."""
    parser = LLDPParser(local_device)
    with open(filepath) as f:
        text = f.read()
    parser.parse_lldp_text(text)
    return parser.to_graph()


def parse_cdp_file(filepath: str, local_device: str = "unknown") -> nx.Graph:
    """Convenience: parse a CDP text file and return a graph."""
    parser = LLDPParser(local_device)
    with open(filepath) as f:
        text = f.read()
    parser.parse_cdp_text(text)
    return parser.to_graph()
