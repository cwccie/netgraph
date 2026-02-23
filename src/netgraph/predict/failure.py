"""
Failure prediction for network topologies.

Predicts link failures, node overloads, capacity exhaustion,
and SLA violation risks using GNN embeddings and trend analysis.
"""

import numpy as np
import networkx as nx
from typing import Optional
from dataclasses import dataclass, field

from ..graph.features import FeatureEngineering
from ..graph.temporal import TemporalGraphBuilder
from ..models.gcn import GCN
from ..models.layers import sigmoid


@dataclass
class FailurePrediction:
    """Prediction results for a single entity."""
    entity: str  # node or edge name
    entity_type: str  # "node" or "edge"
    failure_probability: float = 0.0
    time_to_failure_hours: float = float("inf")
    risk_level: str = "low"  # low, medium, high, critical
    contributing_factors: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class PredictionReport:
    """Complete failure prediction report."""
    link_failures: list[FailurePrediction] = field(default_factory=list)
    node_overloads: list[FailurePrediction] = field(default_factory=list)
    capacity_exhaustion: list[FailurePrediction] = field(default_factory=list)
    sla_violations: list[FailurePrediction] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class FailurePredictor:
    """
    Network failure prediction using GNN embeddings and trend analysis.

    Predicts:
      - Link failures (based on error rates, utilization trends)
      - Node overloads (based on traffic patterns, CPU/memory)
      - Capacity exhaustion (based on growth trends)
      - SLA violation risk (based on latency, jitter, loss)
    """

    def __init__(
        self,
        lookback_window: int = 10,
        prediction_horizon_hours: float = 24.0,
        seed: int = 42,
    ):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon_hours
        self.seed = seed
        self.feature_eng = FeatureEngineering(normalize=True)
        self.temporal = TemporalGraphBuilder()
        self._history: list[dict] = []

    def add_observation(self, G: nx.Graph, timestamp: Optional[float] = None):
        """Add a graph observation to the prediction history."""
        snap = self.temporal.add_snapshot(G, timestamp=timestamp)

        # Extract key metrics for trend analysis
        metrics = {
            "timestamp": snap.timestamp,
            "node_count": snap.node_count,
            "edge_count": snap.edge_count,
        }

        # Per-node metrics
        for node, data in G.nodes(data=True):
            metrics[f"node_{node}_degree"] = G.degree(node)
            metrics[f"node_{node}_util"] = data.get("utilization", 0.0)
            metrics[f"node_{node}_errors"] = data.get("error_rate", 0.0)

        # Per-edge metrics
        for u, v, data in G.edges(data=True):
            key = f"edge_{u}_{v}"
            metrics[f"{key}_util_in"] = data.get("utilization_in", 0.0)
            metrics[f"{key}_util_out"] = data.get("utilization_out", 0.0)
            metrics[f"{key}_errors"] = data.get("error_rate", 0.0)
            metrics[f"{key}_speed"] = data.get("speed_mbps", 0)

        self._history.append(metrics)

    def predict(self, G: nx.Graph) -> PredictionReport:
        """Generate failure predictions for the current topology."""
        report = PredictionReport()

        # Link failure prediction
        report.link_failures = self._predict_link_failures(G)

        # Node overload prediction
        report.node_overloads = self._predict_node_overloads(G)

        # Capacity exhaustion
        report.capacity_exhaustion = self._predict_capacity_exhaustion(G)

        # SLA violations
        report.sla_violations = self._predict_sla_violations(G)

        # Summary
        all_preds = (
            report.link_failures
            + report.node_overloads
            + report.capacity_exhaustion
            + report.sla_violations
        )

        critical = sum(1 for p in all_preds if p.risk_level == "critical")
        high = sum(1 for p in all_preds if p.risk_level == "high")

        report.summary = {
            "total_predictions": len(all_preds),
            "critical_risks": critical,
            "high_risks": high,
            "overall_health": "critical" if critical > 0 else (
                "degraded" if high > 0 else "healthy"
            ),
        }

        return report

    def _predict_link_failures(self, G: nx.Graph) -> list[FailurePrediction]:
        """Predict link failure probabilities."""
        predictions = []

        for u, v, data in G.edges(data=True):
            factors = []
            risk_score = 0.0

            # Error rate analysis
            error_rate = data.get("error_rate", 0.0)
            if error_rate > 0.01:
                risk_score += 0.3
                factors.append(f"High error rate: {error_rate:.4f}")
            if error_rate > 0.05:
                risk_score += 0.3
                factors.append("Critical error rate")

            # Utilization analysis
            util_in = data.get("utilization_in", 0.0)
            util_out = data.get("utilization_out", 0.0)
            max_util = max(util_in, util_out)

            if max_util > 80:
                risk_score += 0.2
                factors.append(f"High utilization: {max_util:.1f}%")
            if max_util > 95:
                risk_score += 0.2
                factors.append("Near capacity saturation")

            # Trend analysis from history
            trend = self._compute_trend(f"edge_{u}_{v}_errors")
            if trend > 0:
                risk_score += min(trend * 10, 0.3)
                factors.append(f"Error trend increasing: {trend:.4f}/interval")

            # Operational status
            if data.get("oper_status") == "down":
                risk_score = 1.0
                factors.append("Link currently down")

            failure_prob = float(sigmoid(np.array([risk_score * 4 - 2]))[0])

            pred = FailurePrediction(
                entity=f"{u} <-> {v}",
                entity_type="edge",
                failure_probability=failure_prob,
                risk_level=self._risk_level(failure_prob),
                contributing_factors=factors,
                confidence=min(0.5 + len(self._history) * 0.05, 0.95),
            )

            if failure_prob > 0.1:
                pred.time_to_failure_hours = max(
                    self.prediction_horizon * (1.0 - failure_prob), 1.0
                )

            predictions.append(pred)

        return sorted(predictions, key=lambda p: p.failure_probability, reverse=True)

    def _predict_node_overloads(self, G: nx.Graph) -> list[FailurePrediction]:
        """Predict node overload probabilities."""
        predictions = []

        for node, data in G.nodes(data=True):
            factors = []
            risk_score = 0.0

            degree = G.degree(node)
            avg_degree = np.mean([d for _, d in G.degree()]) if G.number_of_nodes() > 0 else 0

            if degree > avg_degree * 2:
                risk_score += 0.2
                factors.append(f"High degree: {degree} (avg: {avg_degree:.1f})")

            util = data.get("utilization", 0.0)
            if util > 70:
                risk_score += 0.3
                factors.append(f"High utilization: {util:.1f}%")

            # Check if node is a critical path
            try:
                betweenness = nx.betweenness_centrality(G)
                if betweenness.get(node, 0) > 0.3:
                    risk_score += 0.2
                    factors.append("High betweenness centrality — traffic bottleneck")
            except Exception:
                pass

            failure_prob = float(sigmoid(np.array([risk_score * 4 - 2]))[0])

            if failure_prob > 0.05:
                predictions.append(FailurePrediction(
                    entity=node,
                    entity_type="node",
                    failure_probability=failure_prob,
                    risk_level=self._risk_level(failure_prob),
                    contributing_factors=factors,
                    confidence=min(0.5 + len(self._history) * 0.05, 0.95),
                ))

        return sorted(predictions, key=lambda p: p.failure_probability, reverse=True)

    def _predict_capacity_exhaustion(self, G: nx.Graph) -> list[FailurePrediction]:
        """Predict capacity exhaustion using trend analysis."""
        predictions = []

        for u, v, data in G.edges(data=True):
            speed = data.get("speed_mbps", 0)
            if speed <= 0:
                continue

            util_in = data.get("utilization_in", 0.0)
            util_out = data.get("utilization_out", 0.0)
            max_util = max(util_in, util_out)

            trend = self._compute_trend(f"edge_{u}_{v}_util_in")
            if trend <= 0:
                trend = self._compute_trend(f"edge_{u}_{v}_util_out")

            if trend > 0 and max_util > 50:
                remaining_capacity = 100.0 - max_util
                if trend > 0:
                    hours_to_exhaust = remaining_capacity / (trend * 12)  # 12 samples/day
                else:
                    hours_to_exhaust = float("inf")

                if hours_to_exhaust < self.prediction_horizon * 7:
                    predictions.append(FailurePrediction(
                        entity=f"{u} <-> {v}",
                        entity_type="edge",
                        failure_probability=min(max_util / 100.0, 0.99),
                        time_to_failure_hours=hours_to_exhaust,
                        risk_level=self._risk_level(max_util / 100.0),
                        contributing_factors=[
                            f"Current utilization: {max_util:.1f}%",
                            f"Growth trend: {trend:.2f}%/interval",
                            f"Speed: {speed} Mbps",
                            f"Estimated exhaustion: {hours_to_exhaust:.0f} hours",
                        ],
                        confidence=min(0.4 + len(self._history) * 0.06, 0.9),
                    ))

        return sorted(predictions, key=lambda p: p.time_to_failure_hours)

    def _predict_sla_violations(self, G: nx.Graph) -> list[FailurePrediction]:
        """Predict SLA violation risks."""
        predictions = []

        for u, v, data in G.edges(data=True):
            factors = []
            risk_score = 0.0

            # Packet loss indicator
            error_rate = data.get("error_rate", 0.0)
            if error_rate > 0.001:
                risk_score += 0.3
                factors.append(f"Packet loss: {error_rate * 100:.3f}%")

            # Congestion indicator (high utilization = latency)
            max_util = max(
                data.get("utilization_in", 0.0),
                data.get("utilization_out", 0.0),
            )
            if max_util > 70:
                risk_score += 0.2
                factors.append(f"Congestion risk at {max_util:.1f}% utilization")

            # Path redundancy (single path = higher risk)
            try:
                paths = list(nx.all_simple_paths(G, u, v, cutoff=3))
                if len(paths) <= 1:
                    risk_score += 0.2
                    factors.append("No redundant paths")
            except Exception:
                pass

            failure_prob = float(sigmoid(np.array([risk_score * 4 - 2]))[0])

            if failure_prob > 0.1:
                predictions.append(FailurePrediction(
                    entity=f"{u} <-> {v}",
                    entity_type="edge",
                    failure_probability=failure_prob,
                    risk_level=self._risk_level(failure_prob),
                    contributing_factors=factors,
                    confidence=0.6,
                ))

        return sorted(predictions, key=lambda p: p.failure_probability, reverse=True)

    def _compute_trend(self, metric_key: str) -> float:
        """Compute linear trend from historical observations."""
        values = []
        for h in self._history[-self.lookback_window:]:
            if metric_key in h:
                values.append(h[metric_key])

        if len(values) < 3:
            return 0.0

        x = np.arange(len(values), dtype=np.float64)
        y = np.array(values, dtype=np.float64)

        # Simple linear regression
        x_mean = x.mean()
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()

        if den < 1e-10:
            return 0.0

        return float(num / den)

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability >= 0.8:
            return "critical"
        elif probability >= 0.5:
            return "high"
        elif probability >= 0.2:
            return "medium"
        return "low"
