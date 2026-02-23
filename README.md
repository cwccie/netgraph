# NetGraph

**GNN-based Network Topology Intelligence Platform**

[![CI](https://github.com/cwccie/netgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/cwccie/netgraph/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

NetGraph ingests network topology from LLDP/CDP/SNMP/routing tables, builds temporal graph representations, and applies Graph Neural Networks to perform **anomaly detection**, **failure prediction**, and **impact analysis** — all in pure NumPy, no PyTorch required.

Built by a network engineer (CCIE #14124, 25 years) who wanted GNN intelligence without the ML framework tax.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NetGraph Pipeline                         │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│  Ingest  │  Graph   │  Models  │  Detect  │ Predict  │  Impact  │
│          │          │          │          │          │          │
│ LLDP/CDP │ Feature  │ GCN      │ Graph    │ Link     │ Blast    │
│ SNMP MIB │ Engineer │ GAT      │ Anomaly  │ Failure  │ Radius   │
│ Routing  │ Temporal │ SAGE     │ Node     │ Overload │ Redund.  │
│ Intf.    │ Graph    │ Auto-    │ Edge     │ Capacity │ Critical │
│ State    │ Diff     │ encoder  │ Temporal │ SLA Risk │ What-If  │
├──────────┴──────────┴──────────┴──────────┴──────────┴──────────┤
│              Visualization / API / Dashboard                     │
│  D3.js JSON │ Grafana │ Mermaid │ HTML │ REST API │ Flask UI    │
└─────────────────────────────────────────────────────────────────┘
```

## Why GNNs for Network Topology?

Network topologies **are** graphs. Traditional monitoring treats devices as independent entities, missing the structural relationships that determine how failures propagate. GNNs operate directly on graph structure:

| Approach | What It Sees | What It Misses |
|----------|-------------|----------------|
| SNMP Polling | Individual device metrics | Structural dependencies |
| Syslog/Traps | Events after they happen | Predictive patterns |
| **NetGraph GNN** | **Full topology structure + metrics** | **Nothing — it sees the graph** |

**Graph Convolutional Networks (GCN)** aggregate information from neighboring nodes, learning representations that encode both local metrics AND topological context. A router with 5 healthy neighbors behaves differently than one with 5 failing neighbors — GCNs capture this automatically.

**Graph Attention (GAT)** learns *which* neighbors matter most, automatically weighting the core switch connection higher than the printer uplink.

**Graph Autoencoders** learn to compress and reconstruct the topology. When reconstruction fails for a node or edge, that's an anomaly — the structure doesn't match what the model learned as "normal."

## Quick Start

### Install

```bash
pip install -e ".[all]"
```

### Run the Demo

```bash
netgraph demo
```

This builds a sample enterprise topology (core/distribution/access), trains a VGAE anomaly detector, runs failure prediction and impact analysis, and generates an interactive HTML visualization.

### Ingest Real Topology

```bash
# From LLDP neighbor output
netgraph ingest lldp_output.txt --format lldp --device core-sw1 -o topology.json

# From SNMP walk
netgraph ingest snmp_walk.txt --format snmp --device core-sw1 -o topology.json

# From routing table
netgraph ingest routing_table.txt --format routing --device router1 -o topology.json
```

### Detect Anomalies

```bash
netgraph detect topology.json --threshold 95
```

### Predict Failures

```bash
netgraph predict topology.json
```

### Impact Analysis

```bash
# Full analysis
netgraph impact topology.json

# What happens if core-sw1 fails?
netgraph impact topology.json --node core-sw1
```

### Visualize

```bash
# Interactive HTML topology map
netgraph viz topology.json --format html -o network

# Mermaid diagram for documentation
netgraph viz topology.json --format mermaid -o network

# Grafana dashboard config
netgraph viz topology.json --format grafana -o network
```

### Web Dashboard

```bash
netgraph dashboard --graph topology.json --port 5000
```

### REST API

```bash
# Start the dashboard (includes API)
netgraph dashboard --graph topology.json

# Query topology
curl http://localhost:5000/api/v1/topology

# Get anomaly report
curl http://localhost:5000/api/v1/anomaly

# Get failure predictions
curl http://localhost:5000/api/v1/predict

# Blast radius analysis
curl -X POST http://localhost:5000/api/v1/impact/blast \
  -H "Content-Type: application/json" \
  -d '{"node": "core-sw1"}'

# What-if simulation
curl -X POST http://localhost:5000/api/v1/impact/whatif \
  -H "Content-Type: application/json" \
  -d '{"failures": [{"type": "node", "target": "core-sw1"}]}'
```

## GNN Models

All models are implemented in **pure NumPy** — functional forward passes, backpropagation, and training with Adam optimizer. No PyTorch, TensorFlow, or PyG required.

### Graph Convolutional Network (GCN)

```
H' = σ(D̂⁻¹/² Â D̂⁻¹/² H W)
```

- Aggregates features from k-hop neighborhoods
- Multi-layer architecture with dropout
- Supports node-level and graph-level tasks

### Graph Attention Network (GAT)

```
αᵢⱼ = softmax(LeakyReLU(aᵀ[Whᵢ ‖ Whⱼ]))
```

- Multi-head attention mechanism
- Learns which neighbors are most important
- Automatically weights critical links higher

### GraphSAGE

```
h_v = σ(W · CONCAT(h_v, AGG({h_u : u ∈ N(v)})))
```

- Inductive learning — works on unseen topologies
- Mean, max, and sum aggregation strategies
- Neighborhood sampling for scalability

### Variational Graph Autoencoder (VGAE)

```
Encoder: Z = GCN(A, X)
Decoder: Â = σ(Z Zᵀ)
```

- Learns latent topology representation
- Reconstruction error = anomaly score
- KL divergence regularization for smooth latent space

## Dashboard

The web dashboard provides:

- **Topology View** — Interactive force-directed graph with zoom, drag, and node inspection
- **Anomaly Heatmap** — Color-coded nodes by anomaly score, structural issues highlighted
- **Prediction Timeline** — Failure predictions ranked by probability and urgency
- **Impact Simulator** — Click-to-fail nodes and see blast radius in real time

## Project Structure

```
netgraph/
├── src/netgraph/
│   ├── ingest/           # Topology ingestion (LLDP, CDP, SNMP, routing)
│   ├── graph/            # Graph construction, features, temporal
│   ├── models/           # GNN models (GCN, GAT, SAGE, autoencoder)
│   ├── detect/           # Anomaly detection pipeline
│   ├── predict/          # Failure prediction
│   ├── impact/           # Impact analysis, blast radius, what-if
│   ├── viz/              # Visualization (D3, Mermaid, Grafana, HTML)
│   ├── api/              # REST API
│   ├── dashboard/        # Flask web dashboard
│   └── cli.py            # Click CLI
├── tests/                # 50+ tests
├── sample_data/          # Sample LLDP, routing, SNMP data
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Docker

```bash
# Run demo
docker compose run --rm netgraph-demo

# Start dashboard
docker compose up -d netgraph
# → http://localhost:5000
```

## Requirements

- Python 3.10+
- NumPy >= 1.24
- NetworkX >= 3.0
- Click >= 8.0
- Flask >= 3.0 (optional, for web dashboard)

**No PyTorch. No TensorFlow. No CUDA. Just NumPy.**

## Use Cases

- **Network Operations Center** — Continuous anomaly monitoring with structural awareness
- **Change Management** — Predict impact before making changes (what-if simulation)
- **Capacity Planning** — Forecast link exhaustion and bottlenecks
- **Incident Response** — Instant blast radius calculation during outages
- **Network Auditing** — Identify single points of failure and redundancy gaps
- **Documentation** — Auto-generate topology diagrams in Mermaid/D3

## Author

**Corey A. Wade**
- CCIE #14124 (25 years network engineering)
- PhD Candidate, AI + Security
- GitHub: [@cwccie](https://github.com/cwccie)

## License

MIT — see [LICENSE](LICENSE).
