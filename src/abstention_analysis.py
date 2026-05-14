import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_abstention_rates(
    df: pd.DataFrame, start_year: int, end_year: int, min_votes: int = 10
) -> Dict[str, Any]:
    """
    Calculate abstention rates per country and primary topic.
    Abstention rate is the number of Abstain (0) votes / Total votes.

    Args:
        df: Processed voting data.
        start_year: Start year of period.
        end_year: End year of period.
        min_votes: Minimum votes required to be included.

    Returns:
        Dictionary containing abstention statistics by country and topic.
    """
    df_filtered = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

    if df_filtered.empty:
        return {"error": "No data for specified period"}

    # We want to count total votes and abstain votes (vote == 0)
    df_filtered["is_abstain"] = (df_filtered["vote"] == 0).astype(int)

    # Global abstention rate per country
    country_grouped = df_filtered.groupby("country_identifier")
    country_stats = country_grouped.agg(
        total_votes=("vote", "count"), abstain_count=("is_abstain", "sum")
    )
    country_stats["abstention_rate"] = (
        country_stats["abstain_count"] / country_stats["total_votes"]
    )

    # Filter by minimum votes
    country_stats = country_stats[country_stats["total_votes"] >= min_votes]
    country_stats = country_stats.sort_values("abstention_rate", ascending=False)

    # Topic level abstention rate per country
    topic_col = "primary_topic" if "primary_topic" in df_filtered.columns else "issue"

    topic_grouped = df_filtered.groupby(["country_identifier", topic_col])
    topic_stats = topic_grouped.agg(
        total_votes=("vote", "count"), abstain_count=("is_abstain", "sum")
    ).reset_index()

    topic_stats["abstention_rate"] = (
        topic_stats["abstain_count"] / topic_stats["total_votes"]
    )

    # Filter topic_stats for countries that meet min_votes
    topic_stats = topic_stats[topic_stats["total_votes"] >= (min_votes // 2)]

    # Pivot to create a matrix of Country x Topic abstention rates
    try:
        topic_matrix = topic_stats.pivot_table(
            index="country_identifier",
            columns=topic_col,
            values="abstention_rate",
            aggfunc="mean",
        ).fillna(0)
    except Exception:
        topic_matrix = pd.DataFrame()

    # Sort topics by highest frequency in abstention
    overall_topic_aggs = df_filtered.groupby(topic_col).agg(
        total_votes=("vote", "count"), abstain_count=("is_abstain", "sum")
    )
    overall_topic_aggs["abstention_rate"] = (
        overall_topic_aggs["abstain_count"] / overall_topic_aggs["total_votes"]
    )
    overall_topic_aggs = overall_topic_aggs[
        overall_topic_aggs["total_votes"] >= min_votes
    ]
    overall_topic_aggs = overall_topic_aggs.sort_values(
        "abstention_rate", ascending=False
    )

    # Limit returned payload
    global_rates_dict = (
        country_stats[["abstention_rate", "total_votes"]]
        .head(30)
        .to_dict(orient="index")
    )
    top_strategic_topics_dict = (
        overall_topic_aggs[["abstention_rate", "total_votes"]]
        .head(20)
        .to_dict(orient="index")
    )

    top_countries = list(global_rates_dict.keys())
    if not topic_matrix.empty:
        # Filter topic matrix for only top abstainers
        topic_matrix = topic_matrix.reindex(top_countries).dropna(how="all").fillna(0)
        country_topic_matrix_dict = topic_matrix.to_dict(orient="index")
    else:
        country_topic_matrix_dict = {}

    return {
        "global_rates": global_rates_dict,
        "top_strategic_topics": top_strategic_topics_dict,
        "country_topic_matrix": country_topic_matrix_dict,
    }
