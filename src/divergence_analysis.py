"""
Divergence Analysis Module for UN Voting Records
Identifies when and why voting alliances break or shift
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from src.similarity_utils import compute_cosine_similarity_matrix

logger = logging.getLogger(__name__)


class DivergenceDetector:
    """
    Detects voting pattern divergences between countries and within clusters.
    Identifies specific votes/issues that caused alliance breaks.
    """

    def __init__(self, vote_data: pd.DataFrame, similarity_matrix: pd.DataFrame):
        """
        Initialize divergence detector.

        Args:
            vote_data: Full voting dataframe
            similarity_matrix: Country-to-country similarity matrix
        """
        self.vote_data = vote_data
        self.similarity_matrix = similarity_matrix

    def detect_vote_anomalies(
        self, country_a: str, country_b: str, threshold: float = 0.3
    ) -> List[Dict]:
        """
        Find specific votes where two countries diverged significantly.

        Args:
            country_a: First country
            country_b: Second country
            threshold: Minimum similarity drop to consider anomalous

        Returns:
            List of dicts with: {rcid, issue, date, vote_a, vote_b, similarity_delta, descr}
        """
        logger.info(f"Detecting vote anomalies between {country_a} and {country_b}")

        # Get votes for both countries
        votes_a = self.vote_data[self.vote_data["country_identifier"] == country_a]
        votes_b = self.vote_data[self.vote_data["country_identifier"] == country_b]

        # Merge on rcid to find common votes
        merged = votes_a.merge(votes_b, on="rcid", suffixes=("_a", "_b"))

        anomalies = []

        for _, row in merged.iterrows():
            vote_a = row["vote_a"]
            vote_b = row["vote_b"]

            # Calculate local similarity for this vote
            # (1, 1), (-1, -1), (0, 0) = high similarity
            # (1, -1) = low similarity
            if pd.isna(vote_a) or pd.isna(vote_b):
                local_sim = 0.0
            elif vote_a == vote_b:
                local_sim = 1.0
            elif (vote_a == 1 and vote_b == -1) or (vote_a == -1 and vote_b == 1):
                local_sim = 0.0  # Opposite votes
            else:
                local_sim = 0.5  # One abstained

            # Compare to overall similarity
            overall_sim = self.similarity_matrix.loc[country_a, country_b]
            similarity_delta = overall_sim - local_sim

            if similarity_delta >= threshold:
                anomalies.append(
                    {
                        "rcid": row["rcid"],
                        "issue": row.get("issue_a", "Unknown"),
                        "date": row.get("date_a", None),
                        "vote_a": vote_a,
                        "vote_b": vote_b,
                        "similarity_delta": similarity_delta,
                        "descr": row.get("descr_a", ""),
                    }
                )

        # Sort by similarity delta (most divergent first)
        anomalies.sort(key=lambda x: x["similarity_delta"], reverse=True)

        logger.info(f"Found {len(anomalies)} anomalous votes")
        return anomalies

    def identify_divergence_issues(
        self,
        cluster: List[str],
        time_period: Tuple[int, int],
        top_n: int = 10,
        topic: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Identify issues that caused splits within a cluster.

        Args:
            cluster: List of country identifiers in the cluster
            time_period: (start_year, end_year) tuple
            top_n: Number of top divisive issues to return

        Returns:
            DataFrame with columns: issue, vote_variance, countries_split, avg_similarity
        """
        start_year, end_year = time_period
        logger.info(
            f"Analyzing divergence issues for cluster of {len(cluster)} countries ({start_year}-{end_year})"
        )

        # Filter votes by cluster and time period
        mask = (
            (self.vote_data["country_identifier"].isin(cluster))
            & (self.vote_data["year"] >= start_year)
            & (self.vote_data["year"] <= end_year)
        )
        if topic and "primary_topic" in self.vote_data.columns:
            mask &= self.vote_data["primary_topic"] == topic

        cluster_votes = self.vote_data[mask]

        # Group by issue
        issue_stats = []

        for issue in cluster_votes["issue"].unique():
            issue_votes = cluster_votes[cluster_votes["issue"] == issue]

            # Skip if too few countries voted
            if issue_votes["country_identifier"].nunique() < 2:
                continue

            # Calculate variance in votes for this issue
            vote_counts = issue_votes["vote"].value_counts()
            total_votes = len(issue_votes)

            # Higher variance = more split
            vote_variance = vote_counts.std() if len(vote_counts) > 1 else 0

            # Calculate how many distinct vote positions
            countries_split = vote_counts.count()

            # Calculate average pairwise similarity on this issue
            votes_matrix = issue_votes.pivot_table(
                index="country_identifier",
                columns="rcid",
                values="vote",
                aggfunc="first",
            )

            if len(votes_matrix) >= 2:
                sim_df = compute_cosine_similarity_matrix(
                    votes_matrix,
                    min_norm=1e-8,
                    drop_zero_rows=True,
                )

                if len(sim_df) >= 2:
                    sim = sim_df.to_numpy()
                    avg_similarity = np.mean(sim[np.triu_indices_from(sim, k=1)])
                else:
                    avg_similarity = 1.0
            else:
                avg_similarity = 1.0

            issue_stats.append(
                {
                    "issue": issue,
                    "vote_variance": vote_variance,
                    "countries_split": countries_split,
                    "avg_similarity": avg_similarity,
                    "num_votes": total_votes,
                }
            )

        # Create DataFrame and sort by how divisive (low similarity, high variance)
        df = pd.DataFrame(issue_stats)
        if not df.empty:
            df["divisiveness_score"] = (1 - df["avg_similarity"]) * df["vote_variance"]
            df = df.sort_values("divisiveness_score", ascending=False).head(top_n)

        logger.info(f"Identified {len(df)} divisive issues")
        return df

    def calculate_similarity_delta(
        self,
        before_period: Tuple[int, int],
        after_period: Tuple[int, int],
        resolution_id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compare similarity matrices before and after a specific vote or time period.

        Args:
            before_period: (start_year, end_year) for before period
            after_period: (start_year, end_year) for after period
            resolution_id: Specific resolution to analyze (optional)

        Returns:
            DataFrame with columns: country_a, country_b, sim_before, sim_after, delta
        """
        logger.info(f"Calculating similarity delta: {before_period} vs {after_period}")

        def build_similarity_matrix(period):
            """Helper to build similarity matrix for a time period"""
            start, end = period
            mask = (self.vote_data["year"] >= start) & (self.vote_data["year"] <= end)
            if resolution_id:
                mask &= self.vote_data["rcid"] == resolution_id
            if topic and "primary_topic" in self.vote_data.columns:
                mask &= self.vote_data["primary_topic"] == topic

            df_period = self.vote_data[mask]

            if df_period.empty:
                return pd.DataFrame()

            vote_matrix = df_period.pivot_table(
                index="country_identifier",
                columns="rcid",
                values="vote",
                aggfunc="first",
            )

            sim_df = compute_cosine_similarity_matrix(
                vote_matrix,
                min_norm=1e-8,
                drop_zero_rows=True,
            )

            if len(sim_df) == 0:
                return pd.DataFrame()

            return sim_df

        sim_before = build_similarity_matrix(before_period)
        sim_after = build_similarity_matrix(after_period)

        # Get common countries
        common = list(set(sim_before.index) & set(sim_after.index))

        # Calculate deltas
        deltas = []
        for i, country_a in enumerate(common):
            for j, country_b in enumerate(common):
                if i < j:  # Avoid duplicates
                    before = sim_before.loc[country_a, country_b]
                    after = sim_after.loc[country_a, country_b]
                    delta = after - before

                    deltas.append(
                        {
                            "country_a": country_a,
                            "country_b": country_b,
                            "sim_before": before,
                            "sim_after": after,
                            "delta": delta,
                        }
                    )

        df = pd.DataFrame(deltas)
        df = df.sort_values(
            "delta", ascending=True
        )  # Most negative = biggest divergence

        logger.info(f"Calculated {len(df)} country pair deltas")
        return df

    def generate_divergence_report(
        self,
        cluster: List[str],
        year: int,
        window: int = 2,
        topic: Optional[str] = None,
    ) -> Dict:
        """
        Generate comprehensive divergence report for a cluster.

        Args:
            cluster: List of countries in cluster
            year: Year to analyze
            window: Years before/after to compare

        Returns:
            Dictionary with report sections
        """
        logger.info(f"Generating divergence report for cluster in {year}")

        before_period = (year - window, year - 1)
        after_period = (year, year + window)

        report = {"cluster_size": len(cluster), "analysis_year": year, "window": window}

        # Top divisive issues
        divisive_issues = self.identify_divergence_issues(
            cluster, (year - window, year + window), top_n=5, topic=topic
        )
        report["top_divisive_issues"] = (
            divisive_issues.to_dict("records") if not divisive_issues.empty else []
        )

        # Similarity changes
        similarity_deltas = self.calculate_similarity_delta(
            before_period, after_period, topic=topic
        )

        # Find countries that diverged most
        if not similarity_deltas.empty:
            diverged_pairs = similarity_deltas[
                similarity_deltas["country_a"].isin(cluster)
                & similarity_deltas["country_b"].isin(cluster)
            ].head(5)
            report["most_diverged_pairs"] = diverged_pairs.to_dict("records")
        else:
            report["most_diverged_pairs"] = []

        # Countries that left the cluster (if similarity dropped significantly)
        potential_leavers = []
        for country in cluster:
            # Calculate average similarity to rest of cluster
            similarities = []
            for other in cluster:
                if country != other and other in self.similarity_matrix.columns:
                    similarities.append(self.similarity_matrix.loc[country, other])

            if similarities:
                avg_sim = np.mean(similarities)
                if avg_sim < 0.5:  # Arbitrary threshold
                    potential_leavers.append(
                        {"country": country, "avg_similarity_to_cluster": avg_sim}
                    )

        report["potential_leavers"] = potential_leavers

        return report
