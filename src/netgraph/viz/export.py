"""
Visualization export for network topologies.

Exports graphs to D3.js JSON, Grafana dashboard configs,
Mermaid diagram syntax, and interactive HTML topology maps.
"""

import json
import html
import hashlib
from typing import Optional
import networkx as nx


class D3Exporter:
    """Export NetworkX graphs to D3.js force-directed JSON format."""

    @staticmethod
    def to_d3_json(
        G: nx.Graph,
        node_scores: Optional[dict] = None,
        edge_scores: Optional[dict] = None,
    ) -> dict:
        """
        Convert graph to D3.js force-directed layout JSON.

        Returns dict with 'nodes' and 'links' arrays.
        """
        node_map = {n: i for i, n in enumerate(G.nodes())}

        nodes = []
        for node, data in G.nodes(data=True):
            d = {
                "id": node,
                "index": node_map[node],
                "type": data.get("type", "device"),
                "group": _device_group(data),
            }
            if node_scores and node in node_scores:
                d["anomaly_score"] = node_scores[node]
            for key in ("description", "mgmt_address", "sys_location"):
                if key in data:
                    d[key] = data[key]
            nodes.append(d)

        links = []
        for u, v, data in G.edges(data=True):
            link = {
                "source": node_map[u],
                "target": node_map[v],
                "protocol": data.get("protocol", ""),
                "local_port": data.get("local_port", ""),
                "remote_port": data.get("remote_port", ""),
            }
            if edge_scores and (u, v) in edge_scores:
                link["anomaly_score"] = edge_scores[(u, v)]
            for key in ("speed_mbps", "utilization_in", "utilization_out"):
                if key in data:
                    link[key] = data[key]
            links.append(link)

        return {"nodes": nodes, "links": links}

    @staticmethod
    def save(filepath: str, G: nx.Graph, **kwargs):
        """Save D3.js JSON to file."""
        data = D3Exporter.to_d3_json(G, **kwargs)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)


class MermaidExporter:
    """Export NetworkX graphs to Mermaid diagram syntax."""

    @staticmethod
    def to_mermaid(
        G: nx.Graph,
        direction: str = "TB",
        title: str = "",
    ) -> str:
        """Convert graph to Mermaid diagram syntax."""
        lines = [f"graph {direction}"]
        if title:
            lines.insert(0, f"---\ntitle: {title}\n---")

        # Sanitize node names for Mermaid
        safe = {}
        for node in G.nodes():
            safe[node] = _mermaid_safe(node)

        # Add nodes with styling based on type
        for node, data in G.nodes(data=True):
            name = safe[node]
            label = node
            device_type = data.get("type", "device")

            if device_type == "router":
                lines.append(f"    {name}{{{label}}}")
            elif device_type == "switch" or "Bridge" in data.get("capabilities", []):
                lines.append(f"    {name}[{label}]")
            else:
                lines.append(f"    {name}({label})")

        # Add edges
        for u, v, data in G.edges(data=True):
            u_safe = safe[u]
            v_safe = safe[v]
            protocol = data.get("protocol", "")
            local_port = data.get("local_port", "")
            remote_port = data.get("remote_port", "")

            if local_port or remote_port:
                label = f"{local_port} --- {remote_port}"
                lines.append(f"    {u_safe} -- \"{label}\" --- {v_safe}")
            elif protocol:
                lines.append(f"    {u_safe} -- \"{protocol}\" --- {v_safe}")
            else:
                lines.append(f"    {u_safe} --- {v_safe}")

        return "\n".join(lines)

    @staticmethod
    def save(filepath: str, G: nx.Graph, **kwargs):
        """Save Mermaid diagram to file."""
        content = MermaidExporter.to_mermaid(G, **kwargs)
        with open(filepath, "w") as f:
            f.write(content)


class GrafanaExporter:
    """Export dashboard configuration for Grafana."""

    @staticmethod
    def to_grafana_dashboard(
        G: nx.Graph,
        title: str = "NetGraph Topology",
        datasource: str = "Prometheus",
    ) -> dict:
        """Generate a Grafana dashboard JSON config."""
        panels = []
        panel_id = 1

        # Topology overview panel (node graph)
        panels.append({
            "id": panel_id,
            "title": "Network Topology",
            "type": "nodeGraph",
            "gridPos": {"h": 12, "w": 24, "x": 0, "y": 0},
            "targets": [{
                "datasource": datasource,
                "expr": "netgraph_topology",
            }],
        })
        panel_id += 1

        # Node metrics panels
        for node in list(G.nodes())[:6]:  # Top 6 nodes
            panels.append({
                "id": panel_id,
                "title": f"Node: {node}",
                "type": "timeseries",
                "gridPos": {"h": 6, "w": 8, "x": ((panel_id - 2) % 3) * 8, "y": 12 + ((panel_id - 2) // 3) * 6},
                "targets": [
                    {
                        "datasource": datasource,
                        "expr": f'netgraph_node_anomaly_score{{node="{node}"}}',
                        "legendFormat": "Anomaly Score",
                    },
                    {
                        "datasource": datasource,
                        "expr": f'netgraph_node_utilization{{node="{node}"}}',
                        "legendFormat": "Utilization",
                    },
                ],
            })
            panel_id += 1

        # Anomaly heatmap
        panels.append({
            "id": panel_id,
            "title": "Anomaly Heatmap",
            "type": "heatmap",
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 30},
            "targets": [{
                "datasource": datasource,
                "expr": "netgraph_anomaly_score",
            }],
        })

        return {
            "dashboard": {
                "title": title,
                "tags": ["netgraph", "network", "topology"],
                "timezone": "browser",
                "panels": panels,
                "time": {"from": "now-6h", "to": "now"},
                "refresh": "30s",
            },
        }

    @staticmethod
    def save(filepath: str, G: nx.Graph, **kwargs):
        """Save Grafana dashboard JSON."""
        config = GrafanaExporter.to_grafana_dashboard(G, **kwargs)
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)


class HTMLExporter:
    """Generate interactive HTML topology maps using D3.js."""

    @staticmethod
    def to_html(
        G: nx.Graph,
        title: str = "NetGraph Topology",
        width: int = 1200,
        height: int = 800,
        node_scores: Optional[dict] = None,
        edge_scores: Optional[dict] = None,
    ) -> str:
        """Generate a standalone HTML page with interactive D3.js topology."""
        d3_data = D3Exporter.to_d3_json(G, node_scores, edge_scores)
        data_json = json.dumps(d3_data, default=str)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; background: #1a1a2e; color: #e0e0e0; font-family: sans-serif; }}
        h1 {{ text-align: center; padding: 10px; margin: 0; background: #16213e; }}
        #graph {{ width: 100%; height: calc(100vh - 50px); }}
        .node {{ cursor: pointer; }}
        .node text {{ font-size: 11px; fill: #e0e0e0; }}
        .link {{ stroke-opacity: 0.6; }}
        .tooltip {{
            position: absolute; background: #16213e; border: 1px solid #0f3460;
            padding: 8px 12px; border-radius: 4px; font-size: 12px;
            pointer-events: none; display: none;
        }}
        .legend {{ position: absolute; bottom: 20px; right: 20px; background: #16213e;
            padding: 10px; border-radius: 4px; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>{html.escape(title)}</h1>
    <div id="graph"></div>
    <div class="tooltip" id="tooltip"></div>
    <div class="legend">
        <strong>Node Types</strong><br>
        <span style="color: #e94560;">&#9679;</span> Router &nbsp;
        <span style="color: #0f3460;">&#9679;</span> Switch &nbsp;
        <span style="color: #533483;">&#9679;</span> Other
    </div>
    <script>
    const data = {data_json};

    const width = {width};
    const height = {height};

    const colorScale = d3.scaleOrdinal()
        .domain(["router", "switch", "other"])
        .range(["#e94560", "#0f3460", "#533483"]);

    const svg = d3.select("#graph").append("svg")
        .attr("width", "100%")
        .attr("height", "100%")
        .attr("viewBox", [0, 0, width, height]);

    const g = svg.append("g");

    svg.call(d3.zoom().on("zoom", (event) => {{
        g.attr("transform", event.transform);
    }}));

    const simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links).id(d => d.index).distance(100))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(30));

    const link = g.append("g").selectAll("line")
        .data(data.links).enter().append("line")
        .attr("class", "link")
        .attr("stroke", d => d.anomaly_score > 0.5 ? "#e94560" : "#4a4a6a")
        .attr("stroke-width", d => Math.max(1, (d.speed_mbps || 100) / 1000));

    const node = g.append("g").selectAll("g")
        .data(data.nodes).enter().append("g")
        .attr("class", "node")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("circle")
        .attr("r", d => 8 + (d.anomaly_score || 0) * 10)
        .attr("fill", d => colorScale(d.group))
        .attr("stroke", d => (d.anomaly_score || 0) > 0.5 ? "#ff0000" : "#fff")
        .attr("stroke-width", 1.5);

    node.append("text")
        .attr("dx", 12).attr("dy", 4)
        .text(d => d.id);

    const tooltip = d3.select("#tooltip");

    node.on("mouseover", (event, d) => {{
        tooltip.style("display", "block")
            .html(`<strong>${{d.id}}</strong><br>Type: ${{d.type}}<br>` +
                  (d.anomaly_score ? `Anomaly: ${{d.anomaly_score.toFixed(3)}}<br>` : "") +
                  (d.description ? `${{d.description}}<br>` : ""));
    }}).on("mousemove", (event) => {{
        tooltip.style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px");
    }}).on("mouseout", () => {{ tooltip.style("display", "none"); }});

    simulation.on("tick", () => {{
        link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
    }});

    function dragstarted(event) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }}
    function dragged(event) {{
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }}
    function dragended(event) {{
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }}
    </script>
</body>
</html>"""

    @staticmethod
    def save(filepath: str, G: nx.Graph, **kwargs):
        """Save interactive HTML topology."""
        content = HTMLExporter.to_html(G, **kwargs)
        with open(filepath, "w") as f:
            f.write(content)


def _device_group(attrs: dict) -> str:
    """Determine device group for visualization."""
    dtype = attrs.get("type", "")
    if dtype == "router":
        return "router"
    caps = attrs.get("capabilities", [])
    if "Router" in caps:
        return "router"
    if "Bridge" in caps:
        return "switch"
    return "other"


def _mermaid_safe(name: str) -> str:
    """Make a node name safe for Mermaid diagram syntax."""
    safe = name.replace("-", "_").replace(".", "_").replace("/", "_")
    safe = safe.replace(" ", "_").replace(":", "_")
    if safe[0].isdigit():
        safe = "n" + safe
    return safe
