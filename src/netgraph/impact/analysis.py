"""
Impact analysis for network topologies.

Calculates blast radius, redundancy scoring, critical path
identification, and what-if failure simulation.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional
import copy


@dataclass
class BlastRadius:
    """Impact of a single node or link failure."""
    failed_entity: str
    entity_type: str  # "node" or "edge"
    directly_affected: list[str] = field(default_factory=list)
    indirectly_affected: list[str] = field(default_factory=list)
    isolated_nodes: list[str] = field(default_factory=list)
    disconnected_pairs: int = 0
    severity: str = "low"  # low, medium, high, critical
    connectivity_loss_pct: float = 0.0
    affected_traffic_pct: float = 0.0

    @property
    def total_affected(self) -> int:
        return len(self.directly_affected) + len(self.indirectly_affected)


@dataclass
class RedundancyScore:
    """Redundancy assessment for a node or path."""
    entity: str
    score: float = 0.0  # 0.0 = no redundancy, 1.0 = fully redundant
    alternate_paths: int = 0
    min_path_length: int = 0
    is_single_point_of_failure: bool = False
    details: str = ""


@dataclass
class ImpactReport:
    """Complete impact analysis report."""
    blast_radii: list[BlastRadius] = field(default_factory=list)
    redundancy_scores: list[RedundancyScore] = field(default_factory=list)
    critical_paths: list[dict] = field(default_factory=list)
    what_if_results: list[dict] = field(default_factory=list)
    overall_resilience: float = 0.0
    summary: dict = field(default_factory=dict)


class ImpactAnalyzer:
    """
    Network impact analysis engine.

    Provides:
      - Blast radius calculation (what breaks if X fails)
      - Redundancy scoring (how resilient is each component)
      - Critical path identification (most important paths)
      - What-if simulation (model failures before they happen)
    """

    def __init__(self):
        self._original_graph: Optional[nx.Graph] = None

    def analyze(self, G: nx.Graph) -> ImpactReport:
        """Run complete impact analysis on a topology."""
        self._original_graph = G
        report = ImpactReport()

        # Blast radius for every node
        for node in G.nodes():
            br = self.blast_radius_node(G, node)
            report.blast_radii.append(br)

        # Redundancy scoring
        for node in G.nodes():
            rs = self.redundancy_score(G, node)
            report.redundancy_scores.append(rs)

        # Critical paths
        report.critical_paths = self.find_critical_paths(G)

        # Overall resilience
        report.overall_resilience = self.compute_resilience(G)

        # Summary
        critical_nodes = [
            br.failed_entity for br in report.blast_radii
            if br.severity in ("high", "critical")
        ]
        spof_nodes = [
            rs.entity for rs in report.redundancy_scores
            if rs.is_single_point_of_failure
        ]

        report.summary = {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "critical_nodes": critical_nodes,
            "single_points_of_failure": spof_nodes,
            "overall_resilience": report.overall_resilience,
            "resilience_grade": self._grade(report.overall_resilience),
        }

        return report

    def blast_radius_node(self, G: nx.Graph, node: str) -> BlastRadius:
        """Calculate blast radius if a specific node fails."""
        if node not in G:
            return BlastRadius(failed_entity=node, entity_type="node")

        uG = G.to_undirected() if G.is_directed() else G

        # Directly affected: immediate neighbors
        directly_affected = list(G.neighbors(node))

        # Simulate removal
        G_sim = copy.deepcopy(uG)
        G_sim.remove_node(node)

        # Find isolated nodes
        remaining_nodes = set(G_sim.nodes())
        isolated = list(nx.isolates(G_sim))

        # Count disconnected pairs
        original_components = nx.number_connected_components(uG)
        new_components = nx.number_connected_components(G_sim)
        component_increase = new_components - original_components

        # Indirectly affected: nodes that lose connectivity
        indirectly_affected = []
        if component_increase > 0:
            # Find nodes in newly created components
            components = list(nx.connected_components(G_sim))
            if len(components) > 1:
                largest = max(components, key=len)
                for comp in components:
                    if comp != largest:
                        indirectly_affected.extend(comp)

        # Connectivity loss
        N = G.number_of_nodes()
        if N > 1:
            original_pairs = N * (N - 1) / 2
            remaining_pairs = sum(
                len(c) * (len(c) - 1) / 2
                for c in nx.connected_components(G_sim)
            )
            connectivity_loss = (original_pairs - remaining_pairs) / original_pairs * 100
        else:
            connectivity_loss = 0.0

        total_affected = len(directly_affected) + len(indirectly_affected)
        severity = self._severity(total_affected, N, connectivity_loss)

        return BlastRadius(
            failed_entity=node,
            entity_type="node",
            directly_affected=directly_affected,
            indirectly_affected=indirectly_affected,
            isolated_nodes=isolated,
            disconnected_pairs=component_increase,
            severity=severity,
            connectivity_loss_pct=connectivity_loss,
        )

    def blast_radius_edge(
        self, G: nx.Graph, u: str, v: str
    ) -> BlastRadius:
        """Calculate blast radius if a specific link fails."""
        uG = G.to_undirected() if G.is_directed() else G

        G_sim = copy.deepcopy(uG)
        if G_sim.has_edge(u, v):
            G_sim.remove_edge(u, v)

        original_components = nx.number_connected_components(uG)
        new_components = nx.number_connected_components(G_sim)

        isolated = list(nx.isolates(G_sim))
        indirectly_affected = []
        if new_components > original_components:
            components = list(nx.connected_components(G_sim))
            if len(components) > 1:
                largest = max(components, key=len)
                for comp in components:
                    if comp != largest:
                        indirectly_affected.extend(comp)

        N = G.number_of_nodes()
        total = len(indirectly_affected)
        severity = self._severity(total, N, 0)

        return BlastRadius(
            failed_entity=f"{u} <-> {v}",
            entity_type="edge",
            directly_affected=[u, v],
            indirectly_affected=indirectly_affected,
            isolated_nodes=isolated,
            disconnected_pairs=new_components - original_components,
            severity=severity,
        )

    def redundancy_score(self, G: nx.Graph, node: str) -> RedundancyScore:
        """Calculate redundancy score for a node."""
        if node not in G:
            return RedundancyScore(entity=node, score=0.0)

        uG = G.to_undirected() if G.is_directed() else G

        neighbors = list(G.neighbors(node))
        if not neighbors:
            return RedundancyScore(
                entity=node, score=0.0,
                is_single_point_of_failure=False,
                details="Isolated node",
            )

        # Check if node is an articulation point
        try:
            is_cut = node in nx.articulation_points(uG)
        except Exception:
            is_cut = False

        # Count alternate paths between neighbors (bypassing this node)
        alt_paths = 0
        total_pairs = 0
        G_without = copy.deepcopy(uG)
        G_without.remove_node(node)

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                total_pairs += 1
                n1, n2 = neighbors[i], neighbors[j]
                if n1 in G_without and n2 in G_without:
                    if nx.has_path(G_without, n1, n2):
                        alt_paths += 1

        score = alt_paths / max(total_pairs, 1)
        min_path = 0

        if alt_paths > 0 and total_pairs > 0:
            try:
                lengths = []
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        n1, n2 = neighbors[i], neighbors[j]
                        if n1 in G_without and n2 in G_without:
                            try:
                                lengths.append(
                                    nx.shortest_path_length(G_without, n1, n2)
                                )
                            except nx.NetworkXNoPath:
                                pass
                if lengths:
                    min_path = min(lengths)
            except Exception:
                pass

        return RedundancyScore(
            entity=node,
            score=score,
            alternate_paths=alt_paths,
            min_path_length=min_path,
            is_single_point_of_failure=is_cut,
            details=f"{'SPOF' if is_cut else 'Redundant'} — "
                    f"{alt_paths}/{total_pairs} neighbor pairs connected without node",
        )

    def find_critical_paths(self, G: nx.Graph, top_k: int = 10) -> list[dict]:
        """Identify the most critical paths in the network."""
        uG = G.to_undirected() if G.is_directed() else G
        critical = []

        # Edge betweenness centrality identifies critical links
        try:
            edge_bc = nx.edge_betweenness_centrality(uG)
        except Exception:
            return []

        sorted_edges = sorted(edge_bc.items(), key=lambda x: x[1], reverse=True)

        for (u, v), centrality in sorted_edges[:top_k]:
            # Check if removing this edge disconnects the graph
            br = self.blast_radius_edge(G, u, v)

            critical.append({
                "edge": (u, v),
                "betweenness_centrality": centrality,
                "is_bridge": br.disconnected_pairs > 0,
                "blast_radius": br.total_affected,
                "severity": br.severity,
            })

        return critical

    def what_if(
        self,
        G: nx.Graph,
        failures: list[dict],
    ) -> list[dict]:
        """
        Simulate multiple simultaneous failures.

        Args:
            G: Current topology graph
            failures: List of failure specs:
                [{"type": "node", "target": "router1"},
                 {"type": "edge", "target": ("router1", "router2")}]

        Returns:
            List of impact descriptions per scenario step.
        """
        results = []
        G_sim = copy.deepcopy(G.to_undirected() if G.is_directed() else G)

        original_nodes = set(G_sim.nodes())
        original_edges = set(G_sim.edges())
        original_components = nx.number_connected_components(G_sim)

        for i, failure in enumerate(failures):
            step_result = {
                "step": i + 1,
                "failure": failure,
                "before_nodes": G_sim.number_of_nodes(),
                "before_edges": G_sim.number_of_edges(),
            }

            if failure["type"] == "node":
                target = failure["target"]
                if target in G_sim:
                    G_sim.remove_node(target)
                    step_result["removed"] = target
            elif failure["type"] == "edge":
                u, v = failure["target"]
                if G_sim.has_edge(u, v):
                    G_sim.remove_edge(u, v)
                    step_result["removed"] = f"{u} <-> {v}"

            new_components = nx.number_connected_components(G_sim)
            isolated = list(nx.isolates(G_sim))

            step_result.update({
                "after_nodes": G_sim.number_of_nodes(),
                "after_edges": G_sim.number_of_edges(),
                "components": new_components,
                "isolated_nodes": isolated,
                "network_partitioned": new_components > 1,
            })

            results.append(step_result)

        # Final summary
        remaining_nodes = set(G_sim.nodes())
        lost_nodes = original_nodes - remaining_nodes

        return results

    def compute_resilience(self, G: nx.Graph) -> float:
        """
        Compute overall network resilience score (0.0-1.0).

        Based on:
          - Algebraic connectivity (Fiedler value)
          - Average node redundancy
          - Bridge/articulation point ratio
          - Average degree
        """
        uG = G.to_undirected() if G.is_directed() else G
        N = uG.number_of_nodes()

        if N < 2:
            return 0.0

        scores = []

        # 1. Connectivity: higher = more resilient
        try:
            if nx.is_connected(uG):
                fiedler = nx.algebraic_connectivity(uG)
                scores.append(min(fiedler / 2.0, 1.0))
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)

        # 2. Low ratio of articulation points = more resilient
        try:
            cuts = len(list(nx.articulation_points(uG)))
            scores.append(1.0 - cuts / max(N, 1))
        except Exception:
            scores.append(0.5)

        # 3. Low ratio of bridges = more resilient
        try:
            bridge_count = len(list(nx.bridges(uG)))
            total_edges = uG.number_of_edges()
            scores.append(
                1.0 - bridge_count / max(total_edges, 1)
            )
        except Exception:
            scores.append(0.5)

        # 4. Average degree (higher = more paths)
        avg_degree = np.mean([d for _, d in uG.degree()])
        scores.append(min(avg_degree / 4.0, 1.0))  # 4+ avg degree is good

        return float(np.mean(scores))

    @staticmethod
    def _severity(affected: int, total: int, connectivity_loss: float) -> str:
        if total == 0:
            return "low"
        ratio = affected / total
        if ratio > 0.5 or connectivity_loss > 50:
            return "critical"
        elif ratio > 0.25 or connectivity_loss > 25:
            return "high"
        elif ratio > 0.1 or connectivity_loss > 10:
            return "medium"
        return "low"

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.6:
            return "C"
        elif score >= 0.4:
            return "D"
        return "F"
