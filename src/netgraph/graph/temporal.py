"""
Temporal graph builder.

Constructs time-series of graph snapshots for temporal analysis.
Tracks graph evolution, detects structural changes, and supports
time-windowed feature extraction for temporal GNN models.
"""

import time
import copy
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import networkx as nx


@dataclass
class GraphSnapshot:
    """A single timestamped graph snapshot."""
    graph: nx.Graph
    timestamp: float
    label: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()


@dataclass
class GraphDelta:
    """Changes between two consecutive graph snapshots."""
    t_from: float
    t_to: float
    added_nodes: set
    removed_nodes: set
    added_edges: set
    removed_edges: set
    modified_edges: list  # (u, v, changed_attrs)
    modified_nodes: list  # (node, changed_attrs)

    @property
    def is_stable(self) -> bool:
        return (
            len(self.added_nodes) == 0
            and len(self.removed_nodes) == 0
            and len(self.added_edges) == 0
            and len(self.removed_edges) == 0
        )

    @property
    def change_magnitude(self) -> float:
        """Scalar measure of how much the graph changed."""
        return (
            len(self.added_nodes)
            + len(self.removed_nodes)
            + len(self.added_edges) * 2
            + len(self.removed_edges) * 2
            + len(self.modified_edges)
            + len(self.modified_nodes)
        )


class TemporalGraphBuilder:
    """
    Build and manage temporal graph sequences.

    Maintains an ordered list of graph snapshots and provides:
      - Graph diffing between snapshots
      - Sliding window feature extraction
      - Temporal adjacency tensor construction
      - Graph evolution statistics
    """

    def __init__(self, max_snapshots: int = 1000):
        self.snapshots: list[GraphSnapshot] = []
        self.max_snapshots = max_snapshots

    def add_snapshot(
        self,
        G: nx.Graph,
        timestamp: Optional[float] = None,
        label: str = "",
        metadata: Optional[dict] = None,
    ) -> GraphSnapshot:
        """Add a new graph snapshot to the timeline."""
        ts = timestamp or time.time()
        snap = GraphSnapshot(
            graph=copy.deepcopy(G),
            timestamp=ts,
            label=label,
            metadata=metadata or {},
        )
        self.snapshots.append(snap)

        # Enforce max size
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]

        return snap

    def diff(
        self, snap_a: GraphSnapshot, snap_b: GraphSnapshot
    ) -> GraphDelta:
        """Compute the difference between two snapshots."""
        G_a = snap_a.graph
        G_b = snap_b.graph

        nodes_a = set(G_a.nodes())
        nodes_b = set(G_b.nodes())

        edges_a = set(G_a.edges())
        edges_b = set(G_b.edges())

        # Node changes
        added_nodes = nodes_b - nodes_a
        removed_nodes = nodes_a - nodes_b

        # Edge changes
        added_edges = edges_b - edges_a
        removed_edges = edges_a - edges_b

        # Modified edges (same endpoints, different attributes)
        modified_edges = []
        common_edges = edges_a & edges_b
        for u, v in common_edges:
            attrs_a = G_a.edges[u, v]
            attrs_b = G_b.edges[u, v]
            changes = {}
            all_keys = set(attrs_a.keys()) | set(attrs_b.keys())
            for key in all_keys:
                val_a = attrs_a.get(key)
                val_b = attrs_b.get(key)
                if val_a != val_b:
                    changes[key] = (val_a, val_b)
            if changes:
                modified_edges.append((u, v, changes))

        # Modified nodes
        modified_nodes = []
        common_nodes = nodes_a & nodes_b
        for node in common_nodes:
            attrs_a = G_a.nodes[node]
            attrs_b = G_b.nodes[node]
            changes = {}
            all_keys = set(attrs_a.keys()) | set(attrs_b.keys())
            for key in all_keys:
                val_a = attrs_a.get(key)
                val_b = attrs_b.get(key)
                if val_a != val_b:
                    changes[key] = (val_a, val_b)
            if changes:
                modified_nodes.append((node, changes))

        return GraphDelta(
            t_from=snap_a.timestamp,
            t_to=snap_b.timestamp,
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            added_edges=added_edges,
            removed_edges=removed_edges,
            modified_edges=modified_edges,
            modified_nodes=modified_nodes,
        )

    def get_deltas(self) -> list[GraphDelta]:
        """Compute deltas between all consecutive snapshots."""
        deltas = []
        for i in range(1, len(self.snapshots)):
            delta = self.diff(self.snapshots[i - 1], self.snapshots[i])
            deltas.append(delta)
        return deltas

    def get_window(
        self,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        window_size: Optional[int] = None,
    ) -> list[GraphSnapshot]:
        """Get snapshots within a time window or by count."""
        if window_size is not None:
            return self.snapshots[-window_size:]

        result = []
        for snap in self.snapshots:
            if t_start is not None and snap.timestamp < t_start:
                continue
            if t_end is not None and snap.timestamp > t_end:
                continue
            result.append(snap)
        return result

    def build_adjacency_tensor(
        self,
        window_size: Optional[int] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build a 3D adjacency tensor (T x N x N) from snapshots.

        Uses the union of all nodes across snapshots for consistent indexing.

        Returns:
          - tensor: (T, N, N) numpy array
          - node_list: ordered list of node names
        """
        snaps = self.get_window(window_size=window_size)
        if not snaps:
            return np.array([]), []

        # Union of all nodes
        all_nodes = set()
        for snap in snaps:
            all_nodes.update(snap.graph.nodes())
        node_list = sorted(all_nodes)
        node_map = {n: i for i, n in enumerate(node_list)}
        N = len(node_list)
        T = len(snaps)

        tensor = np.zeros((T, N, N), dtype=np.float32)

        for t, snap in enumerate(snaps):
            for u, v in snap.graph.edges():
                i = node_map[u]
                j = node_map[v]
                tensor[t, i, j] = 1.0
                if not snap.graph.is_directed():
                    tensor[t, j, i] = 1.0

        return tensor, node_list

    def evolution_stats(self) -> dict:
        """Compute statistics about graph evolution over time."""
        if len(self.snapshots) < 2:
            return {"snapshots": len(self.snapshots)}

        deltas = self.get_deltas()

        node_counts = [s.node_count for s in self.snapshots]
        edge_counts = [s.edge_count for s in self.snapshots]
        change_magnitudes = [d.change_magnitude for d in deltas]

        stable_count = sum(1 for d in deltas if d.is_stable)

        return {
            "snapshots": len(self.snapshots),
            "time_span": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            "node_count_mean": np.mean(node_counts),
            "node_count_std": np.std(node_counts),
            "edge_count_mean": np.mean(edge_counts),
            "edge_count_std": np.std(edge_counts),
            "avg_change_magnitude": np.mean(change_magnitudes) if change_magnitudes else 0,
            "max_change_magnitude": max(change_magnitudes) if change_magnitudes else 0,
            "stability_ratio": stable_count / len(deltas) if deltas else 1.0,
            "total_node_additions": sum(len(d.added_nodes) for d in deltas),
            "total_node_removals": sum(len(d.removed_nodes) for d in deltas),
            "total_edge_additions": sum(len(d.added_edges) for d in deltas),
            "total_edge_removals": sum(len(d.removed_edges) for d in deltas),
        }


class SubgraphExtractor:
    """Extract relevant subgraphs from larger topologies."""

    @staticmethod
    def k_hop_neighborhood(
        G: nx.Graph, center: str, k: int = 2
    ) -> nx.Graph:
        """Extract k-hop neighborhood subgraph around a node."""
        if center not in G:
            raise ValueError(f"Node {center} not in graph")

        nodes = {center}
        frontier = {center}
        for _ in range(k):
            new_frontier = set()
            for node in frontier:
                new_frontier.update(G.neighbors(node))
            nodes.update(new_frontier)
            frontier = new_frontier

        return G.subgraph(nodes).copy()

    @staticmethod
    def by_attribute(
        G: nx.Graph, attr: str, value: object
    ) -> nx.Graph:
        """Extract subgraph of nodes matching an attribute value."""
        nodes = [
            n for n, d in G.nodes(data=True) if d.get(attr) == value
        ]
        return G.subgraph(nodes).copy()

    @staticmethod
    def critical_path(
        G: nx.Graph, source: str, target: str
    ) -> nx.Graph:
        """Extract subgraph along all shortest paths between two nodes."""
        try:
            all_paths = list(nx.all_shortest_paths(G, source, target))
        except nx.NetworkXNoPath:
            return nx.Graph()

        nodes = set()
        for path in all_paths:
            nodes.update(path)

        return G.subgraph(nodes).copy()

    @staticmethod
    def high_centrality(
        G: nx.Graph, top_k: int = 10, metric: str = "betweenness"
    ) -> nx.Graph:
        """Extract subgraph of top-k nodes by centrality metric."""
        if metric == "betweenness":
            scores = nx.betweenness_centrality(G)
        elif metric == "closeness":
            scores = nx.closeness_centrality(G)
        elif metric == "degree":
            scores = dict(nx.degree_centrality(G))
        elif metric == "pagerank":
            scores = nx.pagerank(G)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        top_nodes = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return G.subgraph(top_nodes).copy()
