import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from src.main import calculate_similarity, perform_clustering

logger = logging.getLogger(__name__)


def build_sankey_timeline(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    window: int = 5,
    num_clusters: int = 4,
) -> Dict[str, Any]:
    """
    Build a Sankey diagram data structure showing how countries move between voting blocs over time.

    Args:
        df: Voting DataFrame
        start_year: Start year of analysis
        end_year: End year of analysis
        window: Window size in years
        num_clusters: Number of clusters per window

    Returns:
        Dict ready for Plotly Sankey rendering or JSON consumption.
    """
    logger.info(
        f"Building Sankey timeline from {start_year} to {end_year} with {window}Y windows"
    )

    # Generate windows
    windows = []
    for y in range(start_year, end_year + 1, window):
        w_end = min(y + window - 1, end_year)
        windows.append((y, w_end))

    if not windows:
        return {"error": "Invalid time period for windows"}

    # Phase 1: Cluster each window
    window_clusters = []

    for w_start, w_end in windows:
        df_w = df[(df["year"] >= w_start) & (df["year"] <= w_end)]
        if df_w.empty:
            continue

        vote_matrix = df_w.pivot_table(
            index="country_identifier", columns="rcid", values="vote", aggfunc="first"
        )

        sim_matrix = calculate_similarity(vote_matrix)
        if sim_matrix is None or sim_matrix.empty:
            continue

        countries = sim_matrix.index.tolist()
        effective_clusters = min(num_clusters, len(countries))

        if effective_clusters < 2:
            continue

        _, _, labels = perform_clustering(sim_matrix, effective_clusters, countries)

        if labels is None:
            continue

        # Map country to its cluster label for this window
        label_map = {c: int(l) for c, l in zip(countries, labels, strict=False)}
        window_clusters.append({"label": f"{w_start}-{w_end}", "clusters": label_map})

    if len(window_clusters) < 2:
        return {"error": "Not enough valid windows to trace bloc movement"}

    # Phase 2: Build nodes and flows
    nodes = []
    node_indices = {}

    # Create a node for each cluster in each valid window
    # Node names will be e.g. "1980-1984: Bloc 0"
    idx = 0
    for i, w_data in enumerate(window_clusters):
        w_label = w_data["label"]
        unique_blocs = set(w_data["clusters"].values())
        for bloc in unique_blocs:
            node_name = f"{w_label} Bloc {bloc}"
            nodes.append(node_name)
            node_indices[(i, bloc)] = idx
            idx += 1

    # Calculate flows
    sources = []
    targets = []
    values = []

    for i in range(len(window_clusters) - 1):
        prev_w = window_clusters[i]
        curr_w = window_clusters[i + 1]

        # We find countries present in both windows
        common_countries = set(prev_w["clusters"].keys()).intersection(
            set(curr_w["clusters"].keys())
        )

        # Track flows from (prev_bloc -> curr_bloc)
        flow_counts = {}
        for c in common_countries:
            p_bloc = prev_w["clusters"][c]
            c_bloc = curr_w["clusters"][c]

            p_idx = node_indices[(i, p_bloc)]
            c_idx = node_indices[(i + 1, c_bloc)]

            flow_key = (p_idx, c_idx)
            flow_counts[flow_key] = flow_counts.get(flow_key, 0) + 1

        for (src, tgt), count in flow_counts.items():
            sources.append(src)
            targets.append(tgt)
            values.append(count)

    return {
        "nodes": nodes,
        "links": {"source": sources, "target": targets, "value": values},
        "windows": [w["label"] for w in window_clusters],
    }
