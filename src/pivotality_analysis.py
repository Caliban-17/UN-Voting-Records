import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def calculate_pivotality_index(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    margin_threshold: float = 0.15,
    min_votes: int = 10,
) -> Dict[str, Any]:
    """
    Calculate the Pivotality Index for countries.
    A country has a high pivotality index if it frequently votes with the majority
    on tightly contested resolutions (swing votes).

    Args:
        df: The voting DataFrame.
        start_year: Start year of analysis.
        end_year: End year of analysis.
        margin_threshold: The maximum difference between YES and NO votes (as % of total)
                          for a resolution to be considered 'contested'. Default 15%.
        min_votes: Minimum number of votes cast for a resolution to be considered.

    Returns:
        Dict with top swing countries and contested resolution stats.
    """
    df_filtered = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    if df_filtered.empty:
        return {"error": "No data for specified period"}

    # Group by resolution to find contested ones
    # Vote values: 1 (Yes), -1 (No), 0 (Abstain)
    res_grouped = df_filtered.groupby("rcid")

    # Calculate YES, NO, ABSTAIN counts per resolution
    res_stats = res_grouped.agg(
        total_votes=("vote", "count"),
        yes_votes=("vote", lambda x: (x == 1).sum()),
        no_votes=("vote", lambda x: (x == -1).sum()),
        abstain_votes=("vote", lambda x: (x == 0).sum()),
    )

    # Filter out resolutions with very few votes to avoid noise
    res_stats = res_stats[res_stats["total_votes"] >= min_votes]

    # Calculate margin: |YES - NO| / (YES + NO)
    # Using only definitive votes for the margin calculation
    res_stats["definitive_votes"] = res_stats["yes_votes"] + res_stats["no_votes"]

    # Avoid division by zero
    res_stats = res_stats[res_stats["definitive_votes"] > 0]
    res_stats["margin"] = (
        abs(res_stats["yes_votes"] - res_stats["no_votes"])
        / res_stats["definitive_votes"]
    )

    # Identify closely contested resolutions
    contested_res = res_stats[res_stats["margin"] <= margin_threshold].copy()

    if contested_res.empty:
        return {
            "contested_count": 0,
            "pivotality_scores": {},
            "message": f"No tightly contested resolutions found within a {margin_threshold*100}% margin.",
        }

    # Determine the winning side for each contested resolution
    contested_res["winning_side"] = np.where(
        contested_res["yes_votes"] > contested_res["no_votes"], 1, -1
    )

    # Map winning side back to the votes dataframe
    contested_votes = df_filtered[df_filtered["rcid"].isin(contested_res.index)].copy()
    contested_votes = contested_votes.merge(
        contested_res[["winning_side", "margin"]], left_on="rcid", right_index=True
    )

    # A country's vote is a "swing" if they cast a definitive vote on the winning side
    contested_votes["is_swing"] = (
        contested_votes["vote"] == contested_votes["winning_side"]
    ).astype(int)

    # Calculate pivotality score per country
    country_pivotality = contested_votes.groupby("country_identifier").agg(
        contested_participations=("rcid", "count"), swing_votes=("is_swing", "sum")
    )

    # Pivotality index: Swing Votes / Total Contested Resolutions in Period
    # (or per participation)
    # We will use raw swing vote count as the primary metric, since raw count = aggregate power
    country_pivotality["pivotality_index"] = country_pivotality["swing_votes"]

    # Sort and take top 50
    country_pivotality = country_pivotality.sort_values(
        "pivotality_index", ascending=False
    )

    top_pivots = country_pivotality.head(50).to_dict(orient="index")

    return {
        "contested_count": len(contested_res),
        "total_resolutions": len(res_stats),
        "pivotality_scores": top_pivots,
    }
