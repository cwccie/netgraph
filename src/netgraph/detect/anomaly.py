"""
Anomaly detection pipeline for network topologies.

Provides graph-level, node-level, edge-level, and temporal anomaly
detection using GNN-based methods and statistical baselines.
"""

import numpy as np
import networkx as nx
from typing import Optional
from dataclasses import dataclass, field

from ..models.autoencoder import GraphAutoencoder
from ..models.training import train_autoencoder
from ..graph.features import FeatureEngineering
from ..graph.temporal import TemporalGraphBuilder, GraphDelta


@dataclass
class AnomalyReport:
    """Complete anomaly detection report."""
    timestamp: float = 0.0
    graph_score: float = 0.0
    node_scores: dict[str, float] = field(default_factory=dict)
    edge_scores: dict[tuple, float] = field(default_factory=dict)
    anomalous_nodes: list[str] = field(default_factory=list)
    anomalous_edges: list[tuple] = field(default_factory=list)
    temporal_anomalies: list[dict] = field(default_factory=list)
    details: dict = field(default_factory=dict)


class AnomalyDetector:
    """
    Multi-level anomaly detection for network graphs.

    Combines:
      1. Graph autoencoder reconstruction error
      2. Statistical deviation from historical baselines
      3. Temporal change detection
      4. Structural anomaly indicators
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        threshold_percentile: float = 95.0,
        temporal_window: int = 10,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        self.temporal_window = temporal_window
        self.seed = seed

        self.model: Optional[GraphAutoencoder] = None
        self.feature_eng = FeatureEngineering(normalize=True)
        self.temporal = TemporalGraphBuilder()
        self.baseline_stats: dict = {}
        self._trained = False

    def fit(
        self,
        G: nx.Graph,
        epochs: int = 100,
        lr: float = 0.001,
        verbose: bool = False,
    ) -> "AnomalyDetector":
        """Train the anomaly detector on a baseline (normal) graph."""
        features = self.feature_eng.extract(G)
        X = features["X"]
        A = features["A"]
        A_hat = features["A_hat"]

        input_dim = X.shape[1]

        self.model = GraphAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            variational=True,
            seed=self.seed,
        )

        train_autoencoder(
            self.model, A_hat, A, X,
            epochs=epochs, lr=lr, verbose=verbose,
        )

        # Compute baseline scores for thresholding
        scores = self.model.anomaly_scores(A_hat, A, X)
        self.baseline_stats = {
            "mean_node_score": float(scores["node_scores"].mean()),
            "std_node_score": float(scores["node_scores"].std()),
            "node_threshold": float(np.percentile(
                scores["node_scores"], self.threshold_percentile
            )),
            "graph_score": scores["graph_score"],
        }

        # Compute structural baselines
        self.baseline_stats.update(self._structural_stats(G))

        self._trained = True
        self.temporal.add_snapshot(G, label="baseline")

        return self

    def detect(self, G: nx.Graph) -> AnomalyReport:
        """Run full anomaly detection on a graph."""
        report = AnomalyReport()

        # Add snapshot for temporal tracking
        self.temporal.add_snapshot(G)

        # 1. GNN-based detection (if trained)
        if self._trained and self.model is not None:
            self._detect_gnn(G, report)

        # 2. Statistical detection
        self._detect_statistical(G, report)

        # 3. Structural anomalies
        self._detect_structural(G, report)

        # 4. Temporal anomalies
        if len(self.temporal.snapshots) >= 2:
            self._detect_temporal(report)

        # Determine anomalous nodes/edges
        if report.node_scores:
            threshold = self.baseline_stats.get(
                "node_threshold",
                np.percentile(list(report.node_scores.values()), self.threshold_percentile),
            )
            report.anomalous_nodes = [
                n for n, s in report.node_scores.items() if s > threshold
            ]

        return report

    def _detect_gnn(self, G: nx.Graph, report: AnomalyReport):
        """GNN-based anomaly detection using autoencoder."""
        features = self.feature_eng.extract(G)
        X = features["X"]
        A = features["A"]
        A_hat = features["A_hat"]
        node_map = features["node_map"]
        inv_map = {v: k for k, v in node_map.items()}

        scores = self.model.anomaly_scores(A_hat, A, X)

        report.graph_score = scores["graph_score"]

        for idx, score in enumerate(scores["node_scores"]):
            node_name = inv_map.get(idx, str(idx))
            report.node_scores[node_name] = float(score)

        for (i, j), score in scores["edge_scores"].items():
            u = inv_map.get(i, str(i))
            v = inv_map.get(j, str(j))
            report.edge_scores[(u, v)] = score

    def _detect_statistical(self, G: nx.Graph, report: AnomalyReport):
        """Statistical anomaly detection based on graph metrics."""
        metrics = {
            "degree": dict(G.degree()),
            "clustering": nx.clustering(G) if not G.is_directed() else {},
        }

        # Flag nodes with unusual degree
        degrees = list(metrics["degree"].values())
        if degrees:
            mean_deg = np.mean(degrees)
            std_deg = np.std(degrees)
            if std_deg > 0:
                for node, deg in metrics["degree"].items():
                    z_score = abs(deg - mean_deg) / std_deg
                    if z_score > 2.0:
                        current = report.node_scores.get(node, 0.0)
                        report.node_scores[node] = current + z_score * 0.1

        report.details["statistical"] = {
            "mean_degree": float(np.mean(degrees)) if degrees else 0,
            "std_degree": float(np.std(degrees)) if degrees else 0,
        }

    def _detect_structural(self, G: nx.Graph, report: AnomalyReport):
        """Detect structural anomalies (single points of failure, etc.)."""
        uG = G.to_undirected() if G.is_directed() else G

        # Articulation points (single points of failure)
        try:
            cut_vertices = list(nx.articulation_points(uG))
        except Exception:
            cut_vertices = []

        # Bridge edges
        try:
            bridges = list(nx.bridges(uG))
        except Exception:
            bridges = []

        # Isolated nodes
        isolated = list(nx.isolates(G))

        report.details["structural"] = {
            "articulation_points": cut_vertices,
            "bridges": bridges,
            "isolated_nodes": isolated,
            "connected_components": nx.number_connected_components(uG),
        }

        # Boost anomaly score for structural issues
        for node in cut_vertices:
            current = report.node_scores.get(node, 0.0)
            report.node_scores[node] = current + 0.3

        for node in isolated:
            report.node_scores[node] = report.node_scores.get(node, 0.0) + 0.5

    def _detect_temporal(self, report: AnomalyReport):
        """Detect anomalies in graph evolution."""
        snaps = self.temporal.get_window(window_size=self.temporal_window)
        if len(snaps) < 2:
            return

        deltas = []
        for i in range(1, len(snaps)):
            delta = self.temporal.diff(snaps[i - 1], snaps[i])
            deltas.append(delta)

        latest_delta = deltas[-1]

        # Compute change magnitudes
        magnitudes = [d.change_magnitude for d in deltas]
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)

        latest_mag = latest_delta.change_magnitude

        if std_mag > 0:
            z_score = (latest_mag - mean_mag) / std_mag
        else:
            z_score = 0.0

        if z_score > 2.0:
            report.temporal_anomalies.append({
                "type": "sudden_change",
                "magnitude": latest_mag,
                "z_score": z_score,
                "added_nodes": list(latest_delta.added_nodes),
                "removed_nodes": list(latest_delta.removed_nodes),
                "added_edges": [list(e) for e in latest_delta.added_edges],
                "removed_edges": [list(e) for e in latest_delta.removed_edges],
            })

        # Check for disappearing nodes
        if latest_delta.removed_nodes:
            report.temporal_anomalies.append({
                "type": "node_disappearance",
                "nodes": list(latest_delta.removed_nodes),
            })

        # Check for new unexpected nodes
        if latest_delta.added_nodes:
            report.temporal_anomalies.append({
                "type": "new_nodes",
                "nodes": list(latest_delta.added_nodes),
            })

        report.details["temporal"] = {
            "change_magnitude": latest_mag,
            "z_score": z_score,
            "stable": latest_delta.is_stable,
        }

    def _structural_stats(self, G: nx.Graph) -> dict:
        """Compute structural baseline statistics."""
        degrees = [d for _, d in G.degree()]
        return {
            "baseline_nodes": G.number_of_nodes(),
            "baseline_edges": G.number_of_edges(),
            "baseline_mean_degree": float(np.mean(degrees)) if degrees else 0,
            "baseline_density": nx.density(G),
        }
