"""
Network Analysis Module for UN Voting Records
Treats nations as nodes and voting similarity as edges
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
from community import community_louvain
from datetime import datetime

from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    TEMPORAL_DECAY_FACTOR,
)
from src.similarity_utils import compute_cosine_similarity_matrix

logger = logging.getLogger(__name__)


class VotingNetwork:
    """
    Builds and analyzes network graphs of UN voting patterns.

    Nodes: Countries
    Edges: Voting similarity (cosine similarity > threshold)
    Edge weights: Similarity scores with temporal weighting
    """

    def __init__(
        self,
        vote_matrix: pd.DataFrame,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        apply_temporal_weighting: bool = False,
        years: Optional[np.ndarray] = None,
    ):
        """
        Initialize the voting network.

        Args:
            vote_matrix: DataFrame with countries as index, resolutions as columns
            similarity_threshold: Minimum similarity to create an edge
            apply_temporal_weighting: Whether to weight recent votes more heavily
            years: Array of years for each resolution (for temporal weighting)
        """
        self.vote_matrix = vote_matrix
        self.similarity_threshold = similarity_threshold
        self.apply_temporal_weighting = apply_temporal_weighting
        self.years = years
        self.graph = None
        self.similarity_matrix = None
        self.communities = None

        logger.info(f"Initialized VotingNetwork with {len(vote_matrix)} countries")

    def build_graph(self) -> nx.Graph:
        """
        Build a network graph from voting patterns.

        Returns:
            NetworkX Graph object
        """
        logger.info("Building similarity matrix...")

        # Apply temporal weighting if requested
        weighted_matrix = self.vote_matrix.copy()
        if self.apply_temporal_weighting and self.years is not None:
            weighted_matrix = self._apply_temporal_weights(weighted_matrix, self.years)

        self.similarity_matrix = compute_cosine_similarity_matrix(
            weighted_matrix,
            min_norm=1e-8,
            drop_zero_rows=True,
        )

        logger.info("Building network graph...")

        # Create graph
        self.graph = nx.Graph()

        # Add nodes (countries)
        for country in self.similarity_matrix.index:
            self.graph.add_node(country)

        if self.similarity_matrix.empty:
            logger.info("No valid vote vectors available; returning empty graph")
            return self.graph

        # Add edges (similarities above threshold)
        # Add edges (similarities above threshold)
        # Vectorized approach: Find indices where similarity >= threshold
        # This avoids O(N^2) Python loops

        # Get the upper triangle indices (excluding diagonal) to avoid duplicates
        # and self-loops
        mask = np.triu(np.ones(self.similarity_matrix.shape), k=1).astype(bool)

        # Filter by threshold
        threshold_mask = (
            self.similarity_matrix.values >= self.similarity_threshold
        ) & mask

        # Get indices of edges
        rows, cols = np.where(threshold_mask)

        # Get weights
        weights = self.similarity_matrix.values[rows, cols]

        # Get country names
        countries = self.similarity_matrix.index

        # Bulk add edges
        edges_to_add = [
            (countries[r], countries[c], {"weight": w})
            for r, c, w in zip(rows, cols, weights)
        ]

        self.graph.add_edges_from(edges_to_add)
        edge_count = len(edges_to_add)

        logger.info(
            f"Created graph with {self.graph.number_of_nodes()} nodes and {edge_count} edges"
        )

        return self.graph

    def _apply_temporal_weights(
        self, matrix: pd.DataFrame, years: np.ndarray
    ) -> pd.DataFrame:
        """
        Apply exponential decay to older votes.

        Args:
            matrix: Vote matrix
            years: Array of years for each resolution column

        Returns:
            Weighted vote matrix
        """
        current_year = datetime.now().year
        weights = np.power(TEMPORAL_DECAY_FACTOR, current_year - years)
        weighted = matrix * weights
        logger.info(f"Applied temporal weighting (decay={TEMPORAL_DECAY_FACTOR})")
        return weighted

    def calculate_centrality_metrics(self) -> Dict[str, pd.Series]:
        """
        Calculate various centrality metrics for influence/power analysis.

        Returns:
            Dictionary of metric_name -> Series of scores by country
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first")

        logger.info("Calculating centrality metrics...")

        metrics = {}

        # PageRank: Overall influence
        pagerank = nx.pagerank(self.graph, weight="weight")
        metrics["pagerank"] = pd.Series(pagerank).sort_values(ascending=False)

        # Betweenness: Broker/mediator power
        betweenness = nx.betweenness_centrality(self.graph, weight="weight")
        metrics["betweenness"] = pd.Series(betweenness).sort_values(ascending=False)

        # Eigenvector: Influence based on powerful connections
        try:
            eigenvector = nx.eigenvector_centrality(
                self.graph, weight="weight", max_iter=1000
            )
            metrics["eigenvector"] = pd.Series(eigenvector).sort_values(ascending=False)
        except nx.PowerIterationFailedConvergence:
            logger.warning(
                "Eigenvector centrality failed to converge, using approximation"
            )
            eigenvector = nx.eigenvector_centrality_numpy(self.graph, weight="weight")
            metrics["eigenvector"] = pd.Series(eigenvector).sort_values(ascending=False)

        # Degree centrality: Number of connections  (normalized)
        degree = nx.degree_centrality(self.graph)
        metrics["degree"] = pd.Series(degree).sort_values(ascending=False)

        # Closeness: Average distance to all other nodes
        closeness = nx.closeness_centrality(self.graph)
        metrics["closeness"] = pd.Series(closeness).sort_values(ascending=False)

        logger.info("Centrality metrics calculated successfully")

        return metrics

    def detect_communities(self, algorithm="louvain") -> Dict[int, List[str]]:
        """
        Detect voting blocs/communities using community detection algorithms.

        Args:
            algorithm: Community detection algorithm ('louvain', 'girvan_newman', 'label_propagation')

        Returns:
            Dictionary mapping community_id -> list of countries
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first")

        logger.info(f"Detecting communities using {algorithm} algorithm...")

        if algorithm == "louvain":
            # Louvain method (best for modularity optimization)
            partition = community_louvain.best_partition(self.graph, weight="weight")
        elif algorithm == "label_propagation":
            # Label propagation (fast but may vary)
            communities_gen = nx.algorithms.community.label_propagation_communities(
                self.graph
            )
            partition = {}
            for comm_id, community in enumerate(communities_gen):
                for node in community:
                    partition[node] = comm_id
        elif algorithm == "girvan_newman":
            # Girvan-Newman (slow but accurate)
            communities_gen = nx.algorithms.community.girvan_newman(self.graph)
            first_level = next(communities_gen)
            partition = {}
            for comm_id, community in enumerate(first_level):
                for node in community:
                    partition[node] = comm_id
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Convert to dictionary of communities
        self.communities = {}
        for node, comm_id in partition.items():
            if comm_id not in self.communities:
                self.communities[comm_id] = []
            self.communities[comm_id].append(node)

        logger.info(f"Detected {len(self.communities)} communities")
        for comm_id, members in self.communities.items():
            logger.info(f"  Community {comm_id}: {len(members)} members")

        return self.communities

    def get_network_stats(self) -> Dict[str, float]:
        """
        Calculate overall network statistics.

        Returns:
            Dictionary of statistics
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first")

        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_clustering": nx.average_clustering(self.graph, weight="weight"),
        }

        # Connected components
        if nx.is_connected(self.graph):
            stats["is_connected"] = True
            stats["diameter"] = nx.diameter(self.graph)
            stats["avg_shortest_path"] = nx.average_shortest_path_length(
                self.graph, weight="weight"
            )
        else:
            stats["is_connected"] = False
            stats["num_components"] = nx.number_connected_components(self.graph)

        return stats


def build_network_over_time(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    window: str = "1Y",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[Tuple[str, VotingNetwork]]:
    """
    Build multiple networks over time for animation/temporal analysis.

    Args:
        df: Full voting dataframe
        start_year: Beginning of analysis
        end_year: End of analysis
        window: Time window for each network ('1Y', '2Y', '5Y')
        similarity_threshold: Minimum similarity for edges

    Returns:
        List of (time_label, VotingNetwork) tuples
    """
    networks = []

    # Parse window
    if window == "1Y":
        step = 1
    elif window == "2Y":
        step = 2
    elif window == "5Y":
        step = 5
    elif window == "10Y":
        step = 10
    else:
        raise ValueError(f"Invalid window: {window}")

    for year in range(start_year, end_year + 1, step):
        window_end = min(year + step - 1, end_year)

        # Filter data for this window
        df_window = df[(df["year"] >= year) & (df["year"] <= window_end)]

        if df_window.empty:
            continue

        # Create vote matrix
        vote_matrix = df_window.pivot_table(
            index="country_identifier", columns="rcid", values="vote", aggfunc="first"
        )

        # Build network
        network = VotingNetwork(vote_matrix, similarity_threshold)
        network.build_graph()

        time_label = f"{year}-{window_end}" if step > 1 else str(year)
        networks.append((time_label, network))

        logger.info(
            f"Built network for {time_label}: {network.graph.number_of_nodes()} nodes"
        )

    return networks
