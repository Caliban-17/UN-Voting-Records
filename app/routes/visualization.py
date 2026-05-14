"""Visualization routes: network, PCA, issue-timeline, soft-power-trends."""

from __future__ import annotations

import json
import logging

import pandas as pd
from flask import Blueprint, jsonify, request

from app.middleware import with_analysis_slot
from app.routes.core import make_error, make_server_error
from app.services import (
    get_df,
    validate_year_range,
    get_method_metadata,
    compute_soft_power_trends_payload,
)
from src.cache_utils import cached_api
from src.main import preprocess_for_similarity, calculate_similarity

logger = logging.getLogger(__name__)
bp = Blueprint("visualization", __name__)


@bp.route("/network", methods=["POST"])
@cached_api
@with_analysis_slot
def get_network_graph():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)

        start_year, end_year = validate_year_range(data)
        try:
            layout_algo = data.get("layout", "force")
            threshold = float(data.get("threshold", 0.65))
            show_labels = bool(data.get("show_labels", False))
            animate = bool(data.get("animate", False))
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("Threshold must be between 0.0 and 1.0")
        except (ValueError, TypeError):
            return make_error("Invalid threshold or layout", 400)

        from src.network_analysis import VotingNetwork
        from src.network_viz import plot_network_interactive, plot_network_animation

        vote_matrix, _, _ = preprocess_for_similarity(get_df(), start_year, end_year)
        if vote_matrix is None or vote_matrix.empty:
            return make_error("No data for specified period", 400)

        # Animation: build one network per 5-year window
        if animate:
            step = max(1, (end_year - start_year) // 8)
            frames = []
            for yr in range(start_year, end_year + 1, step):
                w_end = min(yr + step - 1, end_year)
                vm, _, _ = preprocess_for_similarity(get_df(), yr, w_end)
                if vm is None or vm.empty:
                    continue
                net = VotingNetwork(vm, similarity_threshold=threshold)
                net.build_graph()
                frames.append((f"{yr}–{w_end}", net.graph))
            if not frames:
                return make_error("Not enough data for animation", 400)
            fig = plot_network_animation(
                frames,
                layout=layout_algo,
                title=f"Network Evolution {start_year}–{end_year}",
            )
            payload = json.loads(fig.to_json())
            payload["meta"] = get_method_metadata(
                start_year, end_year, context={"analysis": "network_animation"}
            )
            return jsonify(payload)

        network = VotingNetwork(vote_matrix, similarity_threshold=threshold)
        network.build_graph()
        if network.graph.number_of_nodes() == 0:
            return make_error("Network has no nodes for selected period", 400)

        communities = (
            network.detect_communities() if network.graph.number_of_edges() > 0 else {}
        )
        fig = plot_network_interactive(
            network.graph,
            layout=layout_algo,
            communities=communities,
            show_labels=show_labels,
        )
        payload = json.loads(fig.to_json())
        payload["meta"] = get_method_metadata(
            start_year,
            end_year,
            context={
                "analysis": "network",
                "threshold": threshold,
                "layout": layout_algo,
            },
        )
        return jsonify(payload)

    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Network graph error", exc)


@bp.route("/pca", methods=["POST"])
@cached_api
@with_analysis_slot
def get_pca_plot():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)

        start_year, end_year = validate_year_range(data)
        projection = data.get("projection", "pca").lower()  # "pca" or "umap"
        if projection not in {"pca", "umap"}:
            projection = "pca"

        vote_matrix, _, _ = preprocess_for_similarity(get_df(), start_year, end_year)
        if vote_matrix is None or vote_matrix.empty:
            return make_error("No data for specified period", 400)
        if min(vote_matrix.shape) < 2:
            return make_error("Not enough data points for projection", 400)

        import plotly.express as px

        countries = vote_matrix.index.tolist()

        fallback = None
        if projection == "umap":
            try:
                import umap  # optional dependency

                reducer = umap.UMAP(n_components=2, random_state=42)
                components = reducer.fit_transform(vote_matrix.values)
                x_label, y_label = "UMAP-1", "UMAP-2"
                title = f"Voting Pattern UMAP Projection {start_year}–{end_year}"
                note = "UMAP neighbourhood projection (exploratory, not causal)"
            except Exception:
                # Python 3.13 ARM64 environments frequently cannot install/run umap-learn yet.
                projection = "pca"
                fallback = "UMAP unavailable in this runtime; returned PCA projection."

        if projection == "pca":
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            components = pca.fit_transform(vote_matrix.values)
            ev = pca.explained_variance_ratio_
            x_label = f"PC1 ({ev[0]:.1%} var)"
            y_label = f"PC2 ({ev[1]:.1%} var)"
            title = f"Voting Pattern PCA Projection {start_year}–{end_year}"
            note = "PCA variance projection (exploratory, not causal ideology axes)"

        pca_df = pd.DataFrame(
            {"x": components[:, 0], "y": components[:, 1], "Country": countries}
        )
        fig = px.scatter(
            pca_df,
            x="x",
            y="y",
            hover_name="Country",
            labels={"x": x_label, "y": y_label},
            title=title,
            template="plotly_white",
        )
        payload = json.loads(fig.to_json())
        payload["meta"] = get_method_metadata(
            start_year,
            end_year,
            context={
                "analysis": "projection",
                "method": projection,
                "note": note,
                "fallback": fallback,
            },
        )
        return jsonify(payload)

    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Projection error", exc)


@bp.route("/issue-timeline", methods=["POST"])
@cached_api
@with_analysis_slot
def get_issue_timeline():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)

        start_year, end_year = validate_year_range(data)
        try:
            top_n = int(data.get("top_n", 10))
            if not (1 <= top_n <= 50):
                raise ValueError("top_n must be between 1 and 50")
        except (ValueError, TypeError):
            return make_error("Invalid top_n parameter", 400)

        import plotly.express as px

        mask = (get_df()["year"] >= start_year) & (get_df()["year"] <= end_year)
        df_period = get_df()[mask]
        if df_period.empty:
            return make_error("No data for specified period", 400)

        # Pick the best column for "topic" — primary_topic is the UNBISnet
        # subject; fall back to issue/agenda if the dataset doesn't carry it.
        topic_col = None
        for candidate in ("primary_topic", "agenda", "issue"):
            if candidate in df_period.columns:
                topic_col = candidate
                break
        if topic_col is None:
            return make_error("No topic column available in dataset", 500)

        # Count distinct resolutions per (year, topic) — voting on a resolution
        # is N≈193 rows, so we must dedupe by rcid to get real agenda counts.
        resolutions = df_period[["rcid", "year", topic_col]].dropna()
        resolutions = resolutions.drop_duplicates(subset=["rcid"])
        resolutions[topic_col] = (
            resolutions[topic_col]
            .astype(str)
            .str.strip()
            .replace({"": "Unknown", "nan": "Unknown"})
        )

        per_year_topic = (
            resolutions.groupby(["year", topic_col]).size().reset_index(name="resolutions")
        )
        top_topics = (
            resolutions[topic_col].value_counts().head(top_n).index.tolist()
        )
        filtered = per_year_topic[per_year_topic[topic_col].isin(top_topics)].copy()

        # Humanize all-caps UNBISnet subject strings.
        filtered["topic"] = filtered[topic_col].apply(
            lambda t: t.capitalize() if isinstance(t, str) and t.isupper() else t
        )
        topic_order = [
            t.capitalize() if isinstance(t, str) and t.isupper() else t
            for t in top_topics
        ]

        fig = px.area(
            filtered,
            x="year",
            y="resolutions",
            color="topic",
            category_orders={"topic": topic_order},
            title=f"Agenda composition — top {top_n} topics ({start_year}–{end_year})",
            template="plotly_white",
            labels={"resolutions": "Resolutions voted", "year": "Year"},
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{fullData.name}</b><br>Year: %{x}<br>Resolutions: %{y}<extra></extra>"
            ),
            mode="lines",
            line=dict(width=0.8),
        )
        fig.update_layout(
            legend_title_text="Topic",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=11),
            ),
            hovermode="x unified",
            margin=dict(l=60, r=240, t=50, b=50),
        )

        payload = json.loads(fig.to_json())
        payload["meta"] = get_method_metadata(
            start_year, end_year, context={"analysis": "issue_timeline", "top_n": top_n}
        )
        return jsonify(payload)

    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Timeline error", exc)


@bp.route("/soft-power-trends", methods=["POST"])
@cached_api
@with_analysis_slot
def get_soft_power_trends():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)
        start_year, end_year = validate_year_range(data)
        return jsonify(compute_soft_power_trends_payload(start_year, end_year))
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Soft power trends error", exc)
