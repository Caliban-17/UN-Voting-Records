"""
Soft Power Calculation Module for UN Voting Records
Calculates influence and power metrics based on network position and voting patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from src.config import (
    PAGERANK_WEIGHT,
    BETWEENNESS_WEIGHT,
    EIGENVECTOR_WEIGHT,
    VOTE_SWAYING_WEIGHT,
)

logger = logging.getLogger(__name__)


class SoftPowerCalculator:
    """
    Calculates soft power scores combining multiple influence metrics.

    Soft Power = weighted combination of:
    - PageRank (overall network influence)
    - Betweenness (broker/mediator power)
    - Eigenvector (power through powerful connections)
    - Vote Swaying Rate (how often others follow)
    """

    def __init__(
        self,
        network,
        vote_data: pd.DataFrame,
        centrality_metrics: Optional[Dict[str, pd.Series]] = None,
    ):
        """
        Initialize soft power calculator.

        Args:
            network: VotingNetwork object with built graph
            vote_data: Full voting dataframe
            centrality_metrics: Pre-calculated centrality metrics (optional)
        """
        self.network = network
        self.vote_data = vote_data
        self.centrality_metrics = centrality_metrics

        if self.centrality_metrics is None:
            logger.info("Centrality metrics not provided, calculating...")
            self.centrality_metrics = network.calculate_centrality_metrics()

    def calculate_vote_swaying_rate(
        self, window: str = "1Y", min_votes: int = 10
    ) -> pd.Series:
        """
        Calculate how often other countries vote similarly after this country votes.
        Measures the ability to influence others' voting behavior.

        Optimized implementation using vectorized operations.

        Args:
            window: Time window for analysis ('1Y', '2Y', '5Y')
            min_votes: Minimum votes required for inclusion

        Returns:
            Series of swaying rates by country (0.0 to 1.0)
        """
        logger.info(f"Calculating vote swaying rates with window={window} (Vectorized)")

        # Filter for countries with enough votes
        vote_counts = self.vote_data["country_identifier"].value_counts()
        valid_countries = vote_counts[vote_counts >= min_votes].index

        if len(valid_countries) == 0:
            return pd.Series()

        # Pivot: rows=rcid, cols=country, values=vote
        # We use all countries for comparison, but only calculate scores for valid_countries
        try:
            pivot = self.vote_data.pivot(
                index="rcid", columns="country_identifier", values="vote"
            )
        except ValueError:
            # Handle duplicates if any (though preprocessing should have removed them)
            pivot = self.vote_data.pivot_table(
                index="rcid",
                columns="country_identifier",
                values="vote",
                aggfunc="first",
            )

        values = pivot.values  # shape (n_res, n_all_countries)
        countries = pivot.columns
        swaying_scores = {}

        for country in valid_countries:
            if country not in countries:
                continue

            # Get this country's votes
            c_idx = pivot.columns.get_loc(country)
            c_votes = values[:, c_idx]  # shape (n_res,)

            # Mask where country voted (ignore NaNs)
            mask = ~pd.isna(c_votes)

            if mask.sum() < min_votes:
                swaying_scores[country] = 0.0
                continue

            # Filter values to where country voted
            valid_c_votes = c_votes[mask]  # shape (n_valid_res,)
            other_votes = values[mask, :]  # shape (n_valid_res, n_all_countries)

            # Compare (broadcasting)
            # valid_c_votes[:, None] is (n_valid_res, 1)
            # other_votes is (n_valid_res, n_all_countries)
            # This creates a boolean matrix of matches
            matches = other_votes == valid_c_votes[:, None]

            # Count valid votes by other countries (not NaN)
            not_nan_others = ~pd.isna(other_votes)

            # Total matches and total valid votes per column (country)
            total_matches_by_others = matches.sum(axis=0)
            total_votes_by_others = not_nan_others.sum(axis=0)

            # Exclude self from calculation
            other_indices = np.arange(len(countries)) != c_idx

            # Sum up matches and total votes across all OTHER countries
            total_matches = total_matches_by_others[other_indices].sum()
            total_possible = total_votes_by_others[other_indices].sum()

            if total_possible > 0:
                swaying_scores[country] = total_matches / total_possible
            else:
                swaying_scores[country] = 0.0

        return pd.Series(swaying_scores).sort_values(ascending=False)

    def aggregate_soft_power_score(
        self, custom_weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Calculate aggregate soft power score from all metrics.

        Args:
            custom_weights: Custom weights for metrics (optional)
                          Keys: 'pagerank', 'betweenness', 'eigenvector', 'vote_swaying'

        Returns:
            Series of soft power scores (0.0 to 1.0, normalized)
        """
        logger.info("Calculating aggregate soft power scores...")

        # Use custom weights or defaults
        weights = {
            "pagerank": PAGERANK_WEIGHT,
            "betweenness": BETWEENNESS_WEIGHT,
            "eigenvector": EIGENVECTOR_WEIGHT,
            "vote_swaying": VOTE_SWAYING_WEIGHT,
        }

        if custom_weights:
            weights.update(custom_weights)
            # Normalize weights to sum to 1.0
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        # Get all countries from network
        all_countries = set(self.network.graph.nodes())

        # Initialize scores
        soft_power = pd.Series(0.0, index=list(all_countries))

        # Normalize each metric to 0-1 range
        def normalize(series: pd.Series) -> pd.Series:
            if series.max() == series.min():
                return pd.Series(0.0, index=series.index)
            return (series - series.min()) / (series.max() - series.min())

        # Add weighted PageRank
        pagerank_norm = normalize(self.centrality_metrics["pagerank"])
        soft_power = soft_power.add(pagerank_norm * weights["pagerank"], fill_value=0)

        # Add weighted Betweenness
        betweenness_norm = normalize(self.centrality_metrics["betweenness"])
        soft_power = soft_power.add(
            betweenness_norm * weights["betweenness"], fill_value=0
        )

        # Add weighted Eigenvector
        eigenvector_norm = normalize(self.centrality_metrics["eigenvector"])
        soft_power = soft_power.add(
            eigenvector_norm * weights["eigenvector"], fill_value=0
        )

        # Calculate and add vote swaying (if weight > 0)
        if weights["vote_swaying"] > 0:
            swaying = self.calculate_vote_swaying_rate()
            swaying_norm = normalize(swaying)
            soft_power = soft_power.add(
                swaying_norm * weights["vote_swaying"], fill_value=0
            )

        # Final normalization
        soft_power = normalize(soft_power)

        logger.info(f"Calculated soft power for {len(soft_power)} countries")
        return soft_power.sort_values(ascending=False)

    def get_metric_breakdown(self, country: str) -> Dict[str, float]:
        """
        Get detailed breakdown of soft power components for a specific country.

        Args:
            country: Country identifier

        Returns:
            Dictionary with all metric scores
        """
        breakdown = {
            "pagerank": self.centrality_metrics["pagerank"].get(country, 0.0),
            "betweenness": self.centrality_metrics["betweenness"].get(country, 0.0),
            "eigenvector": self.centrality_metrics["eigenvector"].get(country, 0.0),
            "degree": self.centrality_metrics["degree"].get(country, 0.0),
            "closeness": self.centrality_metrics["closeness"].get(country, 0.0),
        }

        # Add vote swaying if available
        swaying = self.calculate_vote_swaying_rate()
        breakdown["vote_swaying"] = swaying.get(country, 0.0)

        return breakdown


def track_soft_power_over_time(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    frequency: str = "1Y",
    similarity_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Calculate soft power scores over time for trend analysis.

    Args:
        df: Full voting dataframe
        start_year: Start year
        end_year: End year
        frequency: Calculation frequency ('1Y', '2Y', '5Y')
        similarity_threshold: Threshold for network building

    Returns:
        DataFrame with columns: year, country, soft_power_score, pagerank, betweenness, eigenvector
    """
    from src.network_analysis import VotingNetwork

    results = []

    # Parse frequency
    if frequency == "1Y":
        step = 1
    elif frequency == "2Y":
        step = 2
    elif frequency == "5Y":
        step = 5
    else:
        raise ValueError(f"Invalid frequency: {frequency}")

    for year in range(start_year, end_year + 1, step):
        logger.info(f"Calculating soft power for {year}...")

        # Filter data for this year (and apply temporal weighting to historical data)
        df_year = df[df["year"] <= year]

        if df_year.empty:
            continue

        # Create vote matrix
        vote_matrix = df_year.pivot_table(
            index="country_identifier", columns="rcid", values="vote", aggfunc="first"
        )

        # Build network
        network = VotingNetwork(
            vote_matrix, similarity_threshold, apply_temporal_weighting=True
        )
        network.build_graph()

        # Calculate soft power
        calculator = SoftPowerCalculator(network, df_year)
        soft_power_scores = calculator.aggregate_soft_power_score()

        # Store results
        for country, score in soft_power_scores.items():
            results.append(
                {
                    "year": year,
                    "country": country,
                    "soft_power_score": score,
                    "pagerank": calculator.centrality_metrics["pagerank"].get(
                        country, 0.0
                    ),
                    "betweenness": calculator.centrality_metrics["betweenness"].get(
                        country, 0.0
                    ),
                    "eigenvector": calculator.centrality_metrics["eigenvector"].get(
                        country, 0.0
                    ),
                }
            )

    return pd.DataFrame(results)
