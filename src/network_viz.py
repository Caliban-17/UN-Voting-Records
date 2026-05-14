"""
Interactive Plotly Network Visualization Functions
"""

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def plot_network_interactive(
    graph: nx.Graph,
    layout: str = "force",
    color_by: str = "cluster",
    communities: Optional[Dict[int, List[str]]] = None,
    soft_power_scores: Optional[pd.Series] = None,
    title: str = "UN Voting Network",
    show_labels: bool = False,
    max_labels: int = 18,
) -> go.Figure:
    """
    Create an interactive Plotly network graph.

    Args:
        graph: NetworkX graph object
        layout: Layout algorithm ('force', 'circular', 'hierarchical', 'kamada_kawai', 'spring')
        color_by: How to color nodes ('cluster', 'soft_power', 'degree')
        communities: Dictionary mapping community_id -> list of countries
        soft_power_scores: Series of soft power scores by country
        title: Graph title

    Returns:
        Plotly Figure object
    """
    logger.info(f"Creating interactive network graph with {layout} layout")

    # Calculate node positions based on layout
    if layout == "force" or layout == "spring":
        pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout == "hierarchical":
        # Use built-in hierarchical layout (requires additional setup)
        pos = nx.spring_layout(graph, seed=42)
    else:
        pos = nx.spring_layout(graph, seed=42)

    # Extract node positions
    node_x = []
    node_y = []
    node_hover_text = []
    node_label_text = []
    node_size = []
    node_color = []
    degree_map = dict(graph.degree())
    label_nodes: set[str] = set()
    if show_labels:
        ranked = sorted(degree_map.items(), key=lambda item: item[1], reverse=True)
        label_nodes = {node for node, _ in ranked[: max(1, max_labels)]}

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Node size based on degree
        degree = degree_map.get(node, 0)
        node_size.append(float(np.clip(8 + np.sqrt(degree) * 3.2, 8, 30)))

        # Node color based on color_by parameter
        if color_by == "cluster" and communities:
            # Find which community this node belongs to
            node_community = None
            for comm_id, members in communities.items():
                if node in members:
                    node_community = comm_id
                    break
            node_color.append(node_community if node_community is not None else 0)
        elif color_by == "soft_power" and soft_power_scores is not None:
            node_color.append(soft_power_scores.get(node, 0))
        elif color_by == "degree":
            node_color.append(degree)
        else:
            node_color.append(0)

        # Hover text
        hover_parts = [f"<b>{node}</b>"]
        hover_parts.append(f"Connections: {degree}")
        if soft_power_scores is not None:
            hover_parts.append(f"Soft Power: {soft_power_scores.get(node, 0):.3f}")
        node_hover_text.append("<br>".join(hover_parts))
        node_label_text.append(node if node in label_nodes else "")

    # Extract edge positions
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = graph[edge[0]][edge[1]].get("weight", 0.5)
        edge_weights.append(weight)

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.4, color="rgba(101,120,139,0.26)"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if show_labels else "markers",
        hoverinfo="text",
        text=node_label_text if show_labels else None,
        hovertext=node_hover_text,
        textposition="top center",  # Position labels
        textfont=dict(size=9, color="#4e5b69"),  # Style labels
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            size=node_size,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title=color_by.replace("_", " ").title(),
                xanchor="left",
                titleside="right",
            ),
            line_width=1,
            line_color="rgba(255,255,255,0.55)",
            opacity=0.86,
        ),
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
        ),
    )

    return fig


def plot_pca_interactive(
    pca_df: pd.DataFrame,
    explained_variance: np.ndarray,
    clusters: Optional[Dict[int, List[str]]] = None,
    title: str = "Voting Pattern PCA Projection",
) -> go.Figure:
    """
    Create an interactive PCA scatter plot.

    Args:
        pca_df: DataFrame with columns: PC1, PC2, country
        explained_variance: Array of explained variance ratios
        clusters: Dictionary mapping cluster_id -> list of countries
        title: Chart title

    Returns:
        Plotly Figure
    """
    logger.info("Creating interactive PCA plot")

    # Add cluster info if available
    if clusters:
        country_to_cluster = {}
        for cid, members in clusters.items():
            for country in members:
                country_to_cluster[country] = f"Cluster {cid+1}"

        pca_df["Cluster"] = pca_df["country"].map(country_to_cluster).fillna("Unknown")
        color_col = "Cluster"
    else:
        color_col = None

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        text="country",
        color=color_col,
        title=title,
        labels={
            "PC1": f"PC1 ({explained_variance[0]:.1%} variance)",
            "PC2": f"PC2 ({explained_variance[1]:.1%} variance)",
        },
        hover_data={"country": True, "PC1": ":.3f", "PC2": ":.3f"},
    )

    fig.update_traces(textposition="top center")

    fig.update_layout(
        height=600,
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    return fig


def plot_network_animation(
    networks_by_time: List[Tuple[str, nx.Graph]],
    layout: str = "force",
    title: str = "UN Voting Network Evolution",
) -> go.Figure:
    """
    Create an animated network graph showing evolution over time.

    Args:
        networks_by_time: List of (time_label, graph) tuples
        layout: Layout algorithm
        title: Graph title

    Returns:
        Plotly Figure with animation
    """
    logger.info(f"Creating network animation with {len(networks_by_time)} frames")

    # First, calculate a consistent layout using the last (most complete) graph
    if not networks_by_time:
        return go.Figure()

    _, last_graph = networks_by_time[-1]
    if layout == "spring" or layout == "force":
        pos = nx.spring_layout(last_graph, k=1, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(last_graph, seed=42)

    # Create frames for each time period
    frames = []

    for time_label, graph in networks_by_time:
        # Node positions (use consistent layout)
        node_x = []
        node_y = []
        node_text = []
        node_size = []

        for node in graph.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                degree = graph.degree(node)
                node_size.append(10 + degree * 2)
                node_text.append(
                    f"<b>{node}</b><br>Period: {time_label}<br>Connections: {degree}"
                )

        # Edge positions
        edge_x = []
        edge_y = []

        for edge in graph.edges():
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        # Create traces for this frame
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(size=node_size, color="#1f77b4", line_width=2),
        )

        frames.append(go.Frame(data=[edge_trace, node_trace], name=time_label))

    # Initial frame
    initial_data = frames[0].data if frames else []

    # Create figure with animation controls
    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title=title,
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(
                            args=[
                                [f.name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            method="animate",
                            label=f.name,
                        )
                        for f in frames
                    ],
                    active=0,
                    y=0,
                    len=0.9,
                    x=0.1,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
        ),
        frames=frames,
    )

    return fig


def plot_soft_power_trends(
    soft_power_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Soft Power Trends Over Time",
) -> go.Figure:
    """
    Create an interactive line chart showing soft power trends.

    Args:
        soft_power_df: DataFrame with columns: year, country, soft_power_score
        top_n: Number of top countries to show
        title: Chart title

    Returns:
        Plotly Figure
    """
    logger.info(f"Plotting soft power trends for top {top_n} countries")

    # Get top N countries by average soft power score
    avg_scores = soft_power_df.groupby("country")["soft_power_score"].mean()
    top_countries = avg_scores.nlargest(top_n).index.tolist()

    # Filter data
    df_filtered = soft_power_df[soft_power_df["country"].isin(top_countries)]

    # Create line chart
    fig = px.line(
        df_filtered,
        x="year",
        y="soft_power_score",
        color="country",
        title=title,
        labels={
            "soft_power_score": "Soft Power Score",
            "year": "Year",
            "country": "Country",
        },
        hover_data={"country": True, "year": True, "soft_power_score": ":.3f"},
    )

    fig.update_layout(
        height=600,
        hovermode="x unified",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    return fig


def plot_divergence_timeline(
    divergence_df: pd.DataFrame,
    country_a: str,
    country_b: str,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create a timeline showing when two countries diverged on specific votes.

    Args:
        divergence_df: DataFrame with columns: date, issue, similarity_delta
        country_a: First country
        country_b: Second country
        title: Chart title

    Returns:
        Plotly Figure
    """
    if title is None:
        title = f"Voting Divergence: {country_a} vs {country_b}"

    # Create scatter plot with issues as points
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=divergence_df["date"],
            y=divergence_df["similarity_delta"],
            mode="markers",
            marker=dict(
                size=10,
                color=divergence_df["similarity_delta"],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Divergence<br>Score"),
            ),
            text=divergence_df["issue"],
            hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Divergence: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Similarity Delta",
        height=500,
        hovermode="closest",
    )

    return fig


def plot_soft_power_breakdown(
    metric_breakdown: Dict[str, float], country: str
) -> go.Figure:
    """
    Create a bar chart showing soft power metric breakdown for a specific country.

    Args:
        metric_breakdown: Dictionary of metric -> score
        country: Country name

    Returns:
        Plotly Figure
    """
    metrics = list(metric_breakdown.keys())
    scores = list(metric_breakdown.values())

    fig = go.Figure(data=[go.Bar(x=metrics, y=scores, marker_color="indianred")])

    fig.update_layout(
        title=f"Soft Power Components: {country}",
        xaxis_title="Metric",
        yaxis_title="Score",
        height=400,
    )

    return fig


def plot_bloc_sankey(
    sankey_data: Dict[str, Any], title: str = "Voting Bloc Evolution"
) -> go.Figure:
    """
    Create a Sankey diagram showing how countries move between blocs.

    Args:
        sankey_data: Dict with 'nodes' and 'links' (source, target, value)
        title: Chart title

    Returns:
        Plotly Figure
    """
    logger.info("Creating Sankey bloc timeline plot")

    if "error" in sankey_data:
        return go.Figure()

    nodes = sankey_data["nodes"]
    links = sankey_data["links"]

    # Optional: color nodes by something (e.g. blue for bloc 0, red for bloc 1)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                ),
                link=dict(
                    source=links["source"], target=links["target"], value=links["value"]
                ),
            )
        ]
    )

    fig.update_layout(title_text=title, font_size=10, height=600)
    return fig
