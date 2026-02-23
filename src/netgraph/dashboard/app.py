"""
Flask web dashboard for NetGraph.

Provides interactive topology visualization, anomaly heatmap,
prediction timeline, and impact simulator.
"""

import json
import os
from typing import Optional

try:
    from flask import Flask, render_template, jsonify, request
except ImportError:
    Flask = None

import networkx as nx

from ..api.routes import NetGraphAPI
from ..viz.export import D3Exporter, HTMLExporter
from ..detect.anomaly import AnomalyDetector
from ..predict.failure import FailurePredictor
from ..impact.analysis import ImpactAnalyzer


def create_dashboard(
    G: Optional[nx.Graph] = None,
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
) -> "Flask":
    """Create the NetGraph dashboard Flask application."""
    if Flask is None:
        raise ImportError("Flask is required: pip install flask")

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
    )

    # Initialize API backend
    api_backend = NetGraphAPI()
    if G is not None:
        api_backend.set_graph(G)

    # Register API routes
    api_app = api_backend.create_app()
    for rule in api_app.url_map.iter_rules():
        if rule.endpoint != "static":
            view_func = api_app.view_functions.get(rule.endpoint)
            if view_func:
                app.add_url_rule(rule.rule, rule.endpoint, view_func, methods=rule.methods)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/topology")
    def topology_view():
        return render_template("topology.html")

    @app.route("/anomaly")
    def anomaly_view():
        return render_template("anomaly.html")

    @app.route("/predict")
    def predict_view():
        return render_template("predict.html")

    @app.route("/impact")
    def impact_view():
        return render_template("impact.html")

    @app.route("/set_graph", methods=["POST"])
    def set_graph():
        """Upload topology data."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        G = nx.node_link_graph(data)
        api_backend.set_graph(G)
        return jsonify({"status": "ok", "nodes": G.number_of_nodes()})

    return app


def run_dashboard(
    G: Optional[nx.Graph] = None,
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
):
    """Start the dashboard server."""
    app = create_dashboard(G, host, port, debug)
    app.run(host=host, port=port, debug=debug)
