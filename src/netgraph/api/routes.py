"""
REST API for NetGraph.

Provides endpoints for topology query, anomaly detection,
failure prediction, and impact analysis.
"""

import json
import time
from typing import Optional

try:
    from flask import Flask, Blueprint, jsonify, request, abort
except ImportError:
    Flask = None
    Blueprint = None

import networkx as nx

from ..detect.anomaly import AnomalyDetector, AnomalyReport
from ..predict.failure import FailurePredictor, PredictionReport
from ..impact.analysis import ImpactAnalyzer, ImpactReport
from ..viz.export import D3Exporter
from ..graph.features import FeatureEngineering


class NetGraphAPI:
    """
    REST API server for NetGraph topology intelligence.

    Endpoints:
      GET  /api/v1/topology          — Current topology as D3 JSON
      GET  /api/v1/topology/nodes    — List all nodes with attributes
      GET  /api/v1/topology/edges    — List all edges with attributes
      GET  /api/v1/topology/node/<n> — Single node detail
      GET  /api/v1/anomaly           — Current anomaly report
      GET  /api/v1/anomaly/nodes     — Node anomaly scores
      GET  /api/v1/predict           — Failure predictions
      POST /api/v1/impact/blast      — Blast radius for a node
      POST /api/v1/impact/whatif     — What-if failure simulation
      GET  /api/v1/health            — API health check
    """

    def __init__(self):
        self.graph: Optional[nx.Graph] = None
        self.detector = AnomalyDetector()
        self.predictor = FailurePredictor()
        self.analyzer = ImpactAnalyzer()
        self._anomaly_report: Optional[AnomalyReport] = None
        self._prediction_report: Optional[PredictionReport] = None
        self._last_update: float = 0

    def set_graph(self, G: nx.Graph):
        """Update the current topology graph."""
        self.graph = G
        self._last_update = time.time()

    def create_app(self) -> "Flask":
        """Create and configure the Flask application."""
        if Flask is None:
            raise ImportError("Flask is required: pip install flask")

        app = Flask(__name__)
        api = Blueprint("api", __name__, url_prefix="/api/v1")

        @api.route("/health")
        def health():
            return jsonify({
                "status": "ok",
                "graph_loaded": self.graph is not None,
                "nodes": self.graph.number_of_nodes() if self.graph else 0,
                "edges": self.graph.number_of_edges() if self.graph else 0,
                "last_update": self._last_update,
            })

        @api.route("/topology")
        def topology():
            if self.graph is None:
                abort(404, "No topology loaded")
            scores = None
            if self._anomaly_report:
                scores = self._anomaly_report.node_scores
            data = D3Exporter.to_d3_json(self.graph, node_scores=scores)
            return jsonify(data)

        @api.route("/topology/nodes")
        def topology_nodes():
            if self.graph is None:
                abort(404, "No topology loaded")
            nodes = []
            for node, data in self.graph.nodes(data=True):
                entry = {"id": node, "degree": self.graph.degree(node)}
                entry.update({k: _serialize(v) for k, v in data.items()})
                nodes.append(entry)
            return jsonify({"nodes": nodes, "count": len(nodes)})

        @api.route("/topology/edges")
        def topology_edges():
            if self.graph is None:
                abort(404, "No topology loaded")
            edges = []
            for u, v, data in self.graph.edges(data=True):
                entry = {"source": u, "target": v}
                entry.update({k: _serialize(v_) for k, v_ in data.items()})
                edges.append(entry)
            return jsonify({"edges": edges, "count": len(edges)})

        @api.route("/topology/node/<node_id>")
        def topology_node(node_id):
            if self.graph is None:
                abort(404, "No topology loaded")
            if node_id not in self.graph:
                abort(404, f"Node {node_id} not found")
            data = dict(self.graph.nodes[node_id])
            neighbors = list(self.graph.neighbors(node_id))
            return jsonify({
                "id": node_id,
                "attributes": {k: _serialize(v) for k, v in data.items()},
                "degree": self.graph.degree(node_id),
                "neighbors": neighbors,
            })

        @api.route("/anomaly")
        def anomaly():
            if self.graph is None:
                abort(404, "No topology loaded")
            report = self.detector.detect(self.graph)
            self._anomaly_report = report
            return jsonify({
                "graph_score": report.graph_score,
                "anomalous_nodes": report.anomalous_nodes,
                "anomalous_edges": [
                    list(e) for e in report.anomalous_edges
                ],
                "temporal_anomalies": report.temporal_anomalies,
                "node_count": len(report.node_scores),
                "details": report.details,
            })

        @api.route("/anomaly/nodes")
        def anomaly_nodes():
            if self._anomaly_report is None:
                if self.graph is None:
                    abort(404, "No topology loaded")
                self._anomaly_report = self.detector.detect(self.graph)
            return jsonify({
                "scores": self._anomaly_report.node_scores,
                "anomalous": self._anomaly_report.anomalous_nodes,
            })

        @api.route("/predict")
        def predict():
            if self.graph is None:
                abort(404, "No topology loaded")
            self.predictor.add_observation(self.graph)
            report = self.predictor.predict(self.graph)
            self._prediction_report = report
            return jsonify({
                "summary": report.summary,
                "link_failures": [
                    {
                        "entity": p.entity,
                        "probability": p.failure_probability,
                        "risk_level": p.risk_level,
                        "factors": p.contributing_factors,
                    }
                    for p in report.link_failures[:10]
                ],
                "node_overloads": [
                    {
                        "entity": p.entity,
                        "probability": p.failure_probability,
                        "risk_level": p.risk_level,
                    }
                    for p in report.node_overloads[:10]
                ],
                "capacity_exhaustion": [
                    {
                        "entity": p.entity,
                        "time_to_failure_hours": p.time_to_failure_hours,
                        "risk_level": p.risk_level,
                    }
                    for p in report.capacity_exhaustion[:10]
                ],
            })

        @api.route("/impact/blast", methods=["POST"])
        def impact_blast():
            if self.graph is None:
                abort(404, "No topology loaded")
            data = request.get_json()
            if not data or "node" not in data:
                abort(400, "Request must include 'node' field")

            node = data["node"]
            br = self.analyzer.blast_radius_node(self.graph, node)
            return jsonify({
                "failed_entity": br.failed_entity,
                "severity": br.severity,
                "directly_affected": br.directly_affected,
                "indirectly_affected": br.indirectly_affected,
                "isolated_nodes": br.isolated_nodes,
                "connectivity_loss_pct": br.connectivity_loss_pct,
                "total_affected": br.total_affected,
            })

        @api.route("/impact/whatif", methods=["POST"])
        def impact_whatif():
            if self.graph is None:
                abort(404, "No topology loaded")
            data = request.get_json()
            if not data or "failures" not in data:
                abort(400, "Request must include 'failures' list")

            failures = data["failures"]
            results = self.analyzer.what_if(self.graph, failures)
            return jsonify({"steps": results})

        app.register_blueprint(api)
        return app


def _serialize(value):
    """Make a value JSON-serializable."""
    if isinstance(value, (list, dict, str, int, float, bool, type(None))):
        return value
    return str(value)
