"""
Node and edge feature engineering for GNN input.

Converts NetworkX graph attributes into numerical feature matrices
suitable for Graph Neural Network processing. Handles categorical
encoding, normalization, and feature vector construction.
"""

import numpy as np
import networkx as nx
from typing import Optional


# Default node feature schema
DEFAULT_NODE_FEATURES = [
    "degree",
    "betweenness",
    "closeness",
    "clustering",
    "pagerank",
    "is_router",
    "is_switch",
    "interface_count",
    "error_rate",
    "utilization",
]

# Default edge feature schema
DEFAULT_EDGE_FEATURES = [
    "speed_mbps",
    "utilization_in",
    "utilization_out",
    "error_rate",
    "is_ospf",
    "is_bgp",
    "is_static",
    "is_eigrp",
    "weight",
]


class FeatureEngineering:
    """
    Transform graph attributes into numerical feature matrices.

    Produces:
      - X: Node feature matrix (N x F_node)
      - E: Edge feature matrix (M x F_edge)
      - A: Adjacency matrix (N x N)
      - node_map: node name -> index mapping
    """

    def __init__(
        self,
        node_features: Optional[list[str]] = None,
        edge_features: Optional[list[str]] = None,
        normalize: bool = True,
    ):
        self.node_feature_names = node_features or DEFAULT_NODE_FEATURES
        self.edge_feature_names = edge_features or DEFAULT_EDGE_FEATURES
        self.normalize = normalize
        self.node_map: dict[str, int] = {}
        self.node_stats: dict[str, tuple[float, float]] = {}  # mean, std
        self.edge_stats: dict[str, tuple[float, float]] = {}

    def extract(self, G: nx.Graph) -> dict:
        """
        Extract all features from graph.

        Returns dict with keys:
          - X: node feature matrix (N, F_node)
          - E: edge feature matrix (M, F_edge)
          - A: adjacency matrix (N, N)
          - A_hat: normalized adjacency (N, N)
          - D: degree matrix (N, N)
          - node_map: {name: index}
          - edge_list: list of (src, dst) index pairs
        """
        # Build node map
        nodes = sorted(G.nodes())
        self.node_map = {n: i for i, n in enumerate(nodes)}
        N = len(nodes)

        # Compute graph metrics
        metrics = self._compute_graph_metrics(G)

        # Build node feature matrix
        X = self._build_node_features(G, nodes, metrics)

        # Build adjacency matrix
        A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float32)

        # Build edge features
        E, edge_list = self._build_edge_features(G)

        # Normalized adjacency: D^(-1/2) A D^(-1/2)
        A_hat = self._normalize_adjacency(A)

        # Degree matrix
        D = np.diag(A.sum(axis=1))

        if self.normalize:
            X = self._normalize_matrix(X, "node")
            if E.shape[0] > 0:
                E = self._normalize_matrix(E, "edge")

        return {
            "X": X,
            "E": E,
            "A": A,
            "A_hat": A_hat,
            "D": D,
            "node_map": self.node_map,
            "edge_list": edge_list,
        }

    def _compute_graph_metrics(self, G: nx.Graph) -> dict:
        """Compute centrality and other graph-level metrics."""
        # Use undirected for metrics that need it
        uG = G.to_undirected() if G.is_directed() else G

        metrics = {}
        metrics["degree"] = dict(uG.degree())
        try:
            metrics["betweenness"] = nx.betweenness_centrality(uG)
        except Exception:
            metrics["betweenness"] = {n: 0.0 for n in uG.nodes()}
        try:
            metrics["closeness"] = nx.closeness_centrality(uG)
        except Exception:
            metrics["closeness"] = {n: 0.0 for n in uG.nodes()}
        try:
            metrics["clustering"] = nx.clustering(uG)
        except Exception:
            metrics["clustering"] = {n: 0.0 for n in uG.nodes()}
        try:
            metrics["pagerank"] = nx.pagerank(uG, max_iter=100)
        except Exception:
            metrics["pagerank"] = {n: 1.0 / len(uG) for n in uG.nodes()}

        return metrics

    def _build_node_features(
        self, G: nx.Graph, nodes: list, metrics: dict
    ) -> np.ndarray:
        """Build node feature matrix."""
        N = len(nodes)
        F = len(self.node_feature_names)
        X = np.zeros((N, F), dtype=np.float32)

        for i, node in enumerate(nodes):
            attrs = G.nodes[node]
            for j, feat in enumerate(self.node_feature_names):
                if feat in metrics:
                    X[i, j] = float(metrics[feat].get(node, 0.0))
                elif feat == "is_router":
                    X[i, j] = 1.0 if attrs.get("type") == "router" else 0.0
                elif feat == "is_switch":
                    caps = attrs.get("capabilities", [])
                    X[i, j] = 1.0 if "Bridge" in caps else 0.0
                elif feat == "interface_count":
                    X[i, j] = float(attrs.get("interface_count", 0))
                elif feat == "error_rate":
                    X[i, j] = float(attrs.get("error_rate", 0.0))
                elif feat == "utilization":
                    X[i, j] = float(attrs.get("utilization", 0.0))
                elif feat in attrs:
                    try:
                        X[i, j] = float(attrs[feat])
                    except (ValueError, TypeError):
                        X[i, j] = 0.0

        return X

    def _build_edge_features(self, G: nx.Graph) -> tuple[np.ndarray, list]:
        """Build edge feature matrix."""
        edges = list(G.edges(data=True))
        M = len(edges)
        F = len(self.edge_feature_names)
        E = np.zeros((M, F), dtype=np.float32)
        edge_list = []

        protocol_map = {
            "is_ospf": ["ospf", "ospf-inter-area", "ospf-external-type1",
                        "ospf-external-type2"],
            "is_bgp": ["bgp"],
            "is_static": ["static"],
            "is_eigrp": ["eigrp", "eigrp-external"],
        }

        for k, (u, v, data) in enumerate(edges):
            src = self.node_map.get(u, 0)
            dst = self.node_map.get(v, 0)
            edge_list.append((src, dst))

            proto = data.get("protocol", "").lower()

            for j, feat in enumerate(self.edge_feature_names):
                if feat in protocol_map:
                    E[k, j] = 1.0 if proto in protocol_map[feat] else 0.0
                elif feat == "weight":
                    E[k, j] = float(data.get("weight", 1.0))
                elif feat in data:
                    try:
                        E[k, j] = float(data[feat])
                    except (ValueError, TypeError):
                        E[k, j] = 0.0

        return E, edge_list

    def _normalize_adjacency(self, A: np.ndarray) -> np.ndarray:
        """Compute normalized adjacency: A_hat = D^(-1/2) (A + I) D^(-1/2)."""
        N = A.shape[0]
        A_tilde = A + np.eye(N, dtype=np.float32)  # Add self-loops
        D = np.diag(A_tilde.sum(axis=1))

        # D^(-1/2)
        D_inv_sqrt = np.zeros_like(D)
        diag = np.diag(D)
        nonzero = diag > 0
        D_inv_sqrt[np.arange(N)[nonzero], np.arange(N)[nonzero]] = (
            1.0 / np.sqrt(diag[nonzero])
        )

        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def _normalize_matrix(
        self, M: np.ndarray, which: str
    ) -> np.ndarray:
        """Z-score normalize columns."""
        stats = {}
        result = M.copy()

        for j in range(M.shape[1]):
            col = M[:, j]
            mean = col.mean()
            std = col.std()
            if std < 1e-8:
                std = 1.0
            result[:, j] = (col - mean) / std
            stats[j] = (float(mean), float(std))

        if which == "node":
            self.node_stats = stats
        else:
            self.edge_stats = stats

        return result


def graph_to_matrices(G: nx.Graph, normalize: bool = True) -> dict:
    """Convenience: extract features from a graph with default settings."""
    fe = FeatureEngineering(normalize=normalize)
    return fe.extract(G)
