"""
NetGraph CLI — Command-line interface for the NetGraph platform.

Commands:
  ingest    — Parse topology data from files
  train     — Train GNN models on topology data
  detect    — Run anomaly detection
  predict   — Generate failure predictions
  impact    — Run impact analysis
  viz       — Export visualizations
  dashboard — Start the web dashboard
  demo      — Run a demo with sample data
"""

import json
import sys
import os
import time

try:
    import click
except ImportError:
    print("Click is required: pip install click")
    sys.exit(1)

import networkx as nx


@click.group()
@click.version_option(version="0.1.0", prog_name="netgraph")
def cli():
    """NetGraph — GNN-based Network Topology Intelligence Platform."""
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--format", "-f", type=click.Choice(["lldp", "cdp", "snmp", "routing", "json"]), default="lldp")
@click.option("--device", "-d", default="unknown", help="Local device name")
@click.option("--output", "-o", default=None, help="Output file (JSON graph)")
def ingest(file, format, device, output):
    """Ingest topology data from a file."""
    from .ingest.lldp_parser import LLDPParser
    from .ingest.snmp_walker import SNMPWalker
    from .ingest.routing_parser import RoutingTableParser

    click.echo(f"Ingesting {format} data from {file}...")

    with open(file) as f:
        text = f.read()

    if format == "lldp":
        parser = LLDPParser(device)
        parser.parse_lldp_text(text)
        G = parser.to_graph()
    elif format == "cdp":
        parser = LLDPParser(device)
        parser.parse_cdp_text(text)
        G = parser.to_graph()
    elif format == "snmp":
        walker = SNMPWalker(device)
        walker.parse_walk(text)
        G = walker.to_graph()
    elif format == "routing":
        rparser = RoutingTableParser(device)
        rparser.parse_ios(text)
        G = rparser.to_graph()
    elif format == "json":
        data = json.loads(text)
        if "nodes" in data and "links" in data:
            G = nx.node_link_graph(data)
        else:
            parser = LLDPParser(device)
            parser.parse_json(data)
            G = parser.to_graph()
    else:
        click.echo(f"Unknown format: {format}", err=True)
        return

    click.echo(f"  Nodes: {G.number_of_nodes()}")
    click.echo(f"  Edges: {G.number_of_edges()}")

    if output:
        data = nx.node_link_data(G)
        with open(output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        click.echo(f"  Saved to {output}")

    click.echo("Done.")


@cli.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Choice(["gcn", "gat", "sage", "autoencoder"]), default="gcn")
@click.option("--epochs", "-e", default=100, help="Training epochs")
@click.option("--lr", default=0.01, help="Learning rate")
@click.option("--hidden", default=32, help="Hidden dimension")
@click.option("--output", "-o", default=None, help="Save model to file")
def train(graph_file, model, epochs, lr, hidden, output):
    """Train a GNN model on topology data."""
    from .graph.features import FeatureEngineering
    from .models.gcn import GCN
    from .models.gat import GAT
    from .models.sage import GraphSAGE
    from .models.autoencoder import GraphAutoencoder
    from .models.training import train_autoencoder

    click.echo(f"Loading graph from {graph_file}...")
    with open(graph_file) as f:
        data = json.load(f)
    G = nx.node_link_graph(data)

    click.echo(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    click.echo(f"Training {model} model for {epochs} epochs...")

    fe = FeatureEngineering()
    features = fe.extract(G)
    X = features["X"]
    A = features["A"]
    A_hat = features["A_hat"]

    if model == "autoencoder":
        gnn = GraphAutoencoder(
            input_dim=X.shape[1],
            hidden_dim=hidden,
            latent_dim=hidden // 2,
            variational=True,
        )
        history = train_autoencoder(gnn, A_hat, A, X, epochs=epochs, lr=lr, verbose=True)
    else:
        click.echo(f"Training {model} (node classification mode)...")
        # For demo: use random labels
        import numpy as np
        labels = np.random.randint(0, 3, size=G.number_of_nodes())
        train_mask = np.ones(G.number_of_nodes(), dtype=bool)

        if model == "gcn":
            gnn = GCN(X.shape[1], [hidden], 3)
        elif model == "gat":
            gnn = GAT(X.shape[1], hidden, 3, num_heads=4)
        elif model == "sage":
            gnn = GraphSAGE(X.shape[1], [hidden], 3)

        from .models.training import train_node_classifier
        history = train_node_classifier(
            gnn, A_hat, X, labels, train_mask,
            epochs=epochs, lr=lr, verbose=True,
        )

    click.echo(f"Final loss: {history.train_loss[-1]:.4f}")

    if output:
        if hasattr(gnn, "save"):
            gnn.save(output)
            click.echo(f"Model saved to {output}")


@cli.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.option("--threshold", "-t", default=95.0, help="Anomaly threshold percentile")
@click.option("--output", "-o", default=None, help="Save report to JSON")
def detect(graph_file, threshold, output):
    """Run anomaly detection on a topology."""
    from .detect.anomaly import AnomalyDetector

    click.echo(f"Loading graph from {graph_file}...")
    with open(graph_file) as f:
        data = json.load(f)
    G = nx.node_link_graph(data)

    click.echo("Training anomaly detector...")
    detector = AnomalyDetector(threshold_percentile=threshold)
    detector.fit(G, epochs=50, verbose=False)

    click.echo("Running detection...")
    report = detector.detect(G)

    click.echo(f"\n--- Anomaly Report ---")
    click.echo(f"Graph Score: {report.graph_score:.4f}")
    click.echo(f"Anomalous Nodes ({len(report.anomalous_nodes)}):")
    for node in report.anomalous_nodes:
        score = report.node_scores.get(node, 0)
        click.echo(f"  {node}: {score:.4f}")

    if report.temporal_anomalies:
        click.echo(f"\nTemporal Anomalies ({len(report.temporal_anomalies)}):")
        for ta in report.temporal_anomalies:
            click.echo(f"  {ta['type']}")

    structural = report.details.get("structural", {})
    if structural.get("articulation_points"):
        click.echo(f"\nSingle Points of Failure: {structural['articulation_points']}")
    if structural.get("bridges"):
        click.echo(f"Bridge Links: {structural['bridges']}")

    if output:
        out_data = {
            "graph_score": report.graph_score,
            "anomalous_nodes": report.anomalous_nodes,
            "node_scores": report.node_scores,
            "details": report.details,
        }
        with open(output, "w") as f:
            json.dump(out_data, f, indent=2, default=str)
        click.echo(f"\nReport saved to {output}")


@cli.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Save predictions to JSON")
def predict(graph_file, output):
    """Generate failure predictions."""
    from .predict.failure import FailurePredictor

    click.echo(f"Loading graph from {graph_file}...")
    with open(graph_file) as f:
        data = json.load(f)
    G = nx.node_link_graph(data)

    predictor = FailurePredictor()
    predictor.add_observation(G)
    report = predictor.predict(G)

    click.echo(f"\n--- Failure Prediction Report ---")
    click.echo(f"Overall Health: {report.summary.get('overall_health', 'unknown')}")
    click.echo(f"Critical Risks: {report.summary.get('critical_risks', 0)}")
    click.echo(f"High Risks: {report.summary.get('high_risks', 0)}")

    if report.link_failures:
        click.echo(f"\nLink Failure Risks (top 5):")
        for pred in report.link_failures[:5]:
            click.echo(f"  {pred.entity}: {pred.failure_probability:.2%} ({pred.risk_level})")

    if report.node_overloads:
        click.echo(f"\nNode Overload Risks:")
        for pred in report.node_overloads[:5]:
            click.echo(f"  {pred.entity}: {pred.failure_probability:.2%} ({pred.risk_level})")

    if output:
        with open(output, "w") as f:
            json.dump({
                "summary": report.summary,
                "link_failures": [
                    {"entity": p.entity, "probability": p.failure_probability, "risk": p.risk_level}
                    for p in report.link_failures
                ],
            }, f, indent=2)


@cli.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.option("--node", "-n", default=None, help="Specific node to analyze")
@click.option("--output", "-o", default=None, help="Save report to JSON")
def impact(graph_file, node, output):
    """Run impact analysis on a topology."""
    from .impact.analysis import ImpactAnalyzer

    click.echo(f"Loading graph from {graph_file}...")
    with open(graph_file) as f:
        data = json.load(f)
    G = nx.node_link_graph(data)

    analyzer = ImpactAnalyzer()

    if node:
        br = analyzer.blast_radius_node(G, node)
        click.echo(f"\n--- Blast Radius: {node} ---")
        click.echo(f"Severity: {br.severity}")
        click.echo(f"Directly Affected: {br.directly_affected}")
        click.echo(f"Indirectly Affected: {br.indirectly_affected}")
        click.echo(f"Connectivity Loss: {br.connectivity_loss_pct:.1f}%")
    else:
        report = analyzer.analyze(G)
        click.echo(f"\n--- Impact Analysis Report ---")
        click.echo(f"Resilience Score: {report.overall_resilience:.2f}")
        click.echo(f"Resilience Grade: {report.summary.get('resilience_grade', '?')}")
        click.echo(f"Single Points of Failure: {report.summary.get('single_points_of_failure', [])}")
        click.echo(f"Critical Nodes: {report.summary.get('critical_nodes', [])}")

        if report.critical_paths:
            click.echo(f"\nMost Critical Links:")
            for cp in report.critical_paths[:5]:
                click.echo(f"  {cp['edge']}: betweenness={cp['betweenness_centrality']:.4f}, bridge={cp['is_bridge']}")

    if output:
        with open(output, "w") as f:
            json.dump(report.summary if not node else {
                "node": node, "severity": br.severity,
                "directly_affected": br.directly_affected,
                "connectivity_loss_pct": br.connectivity_loss_pct,
            }, f, indent=2, default=str)


@cli.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.option("--format", "-f", type=click.Choice(["html", "d3", "mermaid", "grafana"]), default="html")
@click.option("--output", "-o", default="topology", help="Output filename (without extension)")
def viz(graph_file, format, output):
    """Export topology visualizations."""
    from .viz.export import D3Exporter, MermaidExporter, GrafanaExporter, HTMLExporter

    click.echo(f"Loading graph from {graph_file}...")
    with open(graph_file) as f:
        data = json.load(f)
    G = nx.node_link_graph(data)

    if format == "html":
        filepath = f"{output}.html"
        HTMLExporter.save(filepath, G, title="NetGraph Topology")
    elif format == "d3":
        filepath = f"{output}.json"
        D3Exporter.save(filepath, G)
    elif format == "mermaid":
        filepath = f"{output}.md"
        MermaidExporter.save(filepath, G, title="Network Topology")
    elif format == "grafana":
        filepath = f"{output}-grafana.json"
        GrafanaExporter.save(filepath, G)

    click.echo(f"Exported to {filepath}")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Dashboard host")
@click.option("--port", "-p", default=5000, help="Dashboard port")
@click.option("--graph", "-g", default=None, help="Pre-load graph file")
def dashboard(host, port, graph):
    """Start the web dashboard."""
    from .dashboard.app import run_dashboard

    G = None
    if graph:
        with open(graph) as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
        click.echo(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    click.echo(f"Starting NetGraph dashboard on {host}:{port}")
    run_dashboard(G, host=host, port=port, debug=True)


@cli.command()
def demo():
    """Run a demo with sample topology data."""
    from .graph.features import FeatureEngineering
    from .detect.anomaly import AnomalyDetector
    from .predict.failure import FailurePredictor
    from .impact.analysis import ImpactAnalyzer
    from .viz.export import HTMLExporter, MermaidExporter

    click.echo("=" * 60)
    click.echo("  NetGraph Demo — GNN Network Topology Intelligence")
    click.echo("=" * 60)

    # Build a sample enterprise network topology
    click.echo("\n[1/6] Building sample enterprise topology...")
    G = _build_demo_topology()
    click.echo(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Feature extraction
    click.echo("\n[2/6] Extracting graph features for GNN...")
    fe = FeatureEngineering()
    features = fe.extract(G)
    click.echo(f"  Node features: {features['X'].shape}")
    click.echo(f"  Adjacency: {features['A'].shape}")

    # Anomaly detection
    click.echo("\n[3/6] Training anomaly detector (VGAE)...")
    detector = AnomalyDetector(hidden_dim=16, latent_dim=8)
    detector.fit(G, epochs=50, verbose=False)
    report = detector.detect(G)
    click.echo(f"  Graph anomaly score: {report.graph_score:.4f}")
    click.echo(f"  Structural issues: {report.details.get('structural', {}).get('articulation_points', [])}")

    # Failure prediction
    click.echo("\n[4/6] Running failure prediction...")
    predictor = FailurePredictor()
    predictor.add_observation(G)
    pred_report = predictor.predict(G)
    click.echo(f"  Overall health: {pred_report.summary.get('overall_health', 'unknown')}")
    if pred_report.link_failures:
        top = pred_report.link_failures[0]
        click.echo(f"  Highest risk: {top.entity} ({top.failure_probability:.1%})")

    # Impact analysis
    click.echo("\n[5/6] Running impact analysis...")
    analyzer = ImpactAnalyzer()
    impact_report = analyzer.analyze(G)
    click.echo(f"  Resilience: {impact_report.overall_resilience:.2f} (grade: {impact_report.summary['resilience_grade']})")
    click.echo(f"  SPOFs: {impact_report.summary['single_points_of_failure']}")

    # What-if simulation
    core_node = "core-sw1"
    if core_node in G:
        br = analyzer.blast_radius_node(G, core_node)
        click.echo(f"  If {core_node} fails: {br.total_affected} nodes affected ({br.severity})")

    # Visualization
    click.echo("\n[6/6] Generating visualizations...")
    HTMLExporter.save("/tmp/netgraph-demo.html", G,
                     title="NetGraph Demo", node_scores=report.node_scores)
    click.echo(f"  HTML topology: /tmp/netgraph-demo.html")

    mermaid = MermaidExporter.to_mermaid(G, title="Demo Topology")
    click.echo(f"  Mermaid diagram: {len(mermaid)} chars")

    click.echo("\n" + "=" * 60)
    click.echo("  Demo complete! Open /tmp/netgraph-demo.html in a browser.")
    click.echo("=" * 60)


def _build_demo_topology() -> nx.Graph:
    """Build a realistic enterprise network topology for demos."""
    G = nx.Graph()

    # Core layer
    for i in range(1, 3):
        G.add_node(f"core-sw{i}", type="router", capabilities=["Router", "Bridge"],
                   utilization=45.0 + i * 5, interface_count=48)

    # Distribution layer
    for i in range(1, 5):
        G.add_node(f"dist-sw{i}", type="router", capabilities=["Router", "Bridge"],
                   utilization=35.0 + i * 3, interface_count=24)

    # Access layer
    for i in range(1, 9):
        G.add_node(f"access-sw{i}", type="network_device", capabilities=["Bridge"],
                   utilization=20.0 + i * 4, interface_count=48)

    # WAN routers
    for i in range(1, 3):
        G.add_node(f"wan-rtr{i}", type="router", capabilities=["Router"],
                   utilization=55.0 + i * 5, interface_count=8)

    # Firewall
    G.add_node("fw1", type="router", capabilities=["Router"],
               utilization=60.0, interface_count=8)

    # Core interconnects (full mesh)
    G.add_edge("core-sw1", "core-sw2", protocol="lldp",
               local_port="Te1/0/1", remote_port="Te1/0/1",
               speed_mbps=10000, utilization_in=40.0, utilization_out=38.0,
               error_rate=0.0001)

    # Core to distribution
    for i in range(1, 5):
        for c in range(1, 3):
            G.add_edge(f"core-sw{c}", f"dist-sw{i}", protocol="lldp",
                      local_port=f"Te1/0/{i+1}", remote_port=f"Te1/0/{c}",
                      speed_mbps=10000, utilization_in=30.0, utilization_out=28.0,
                      error_rate=0.0)

    # Distribution to access
    access_map = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8]}
    for dist, access_list in access_map.items():
        for acc in access_list:
            G.add_edge(f"dist-sw{dist}", f"access-sw{acc}", protocol="lldp",
                      local_port=f"Gi1/0/{acc}", remote_port="Gi1/0/1",
                      speed_mbps=1000, utilization_in=45.0, utilization_out=42.0,
                      error_rate=0.0002)

    # WAN connections
    G.add_edge("core-sw1", "wan-rtr1", protocol="lldp",
               speed_mbps=10000, utilization_in=55.0, error_rate=0.001)
    G.add_edge("core-sw2", "wan-rtr2", protocol="lldp",
               speed_mbps=10000, utilization_in=50.0, error_rate=0.0005)
    G.add_edge("wan-rtr1", "wan-rtr2", protocol="lldp",
               speed_mbps=1000, utilization_in=20.0, error_rate=0.0)

    # Firewall
    G.add_edge("core-sw1", "fw1", protocol="lldp",
               speed_mbps=10000, utilization_in=60.0, error_rate=0.0)

    return G


if __name__ == "__main__":
    cli()
