"""Analysis routes: clustering, soft-power, comparison, divergence-report."""

from __future__ import annotations

import logging
import json
from collections import defaultdict

from flask import Blueprint, jsonify, request

from app.middleware import with_analysis_slot
from app.routes.core import make_error, make_server_error
from app.services import (
    country_names,
    get_df,
    validate_year_range,
    normalize_country_code,
    get_method_metadata,
    compute_cluster_stability,
    DEFAULT_STABILITY_BOOTSTRAPS,
)
from src.cluster_naming import label_clusters
from src.cache_utils import cached_api
from src.main import preprocess_for_similarity, calculate_similarity, perform_clustering
from src.network_analysis import VotingNetwork
from src.soft_power import SoftPowerCalculator
from src.abstention_analysis import calculate_abstention_rates
from src.pivotality_analysis import calculate_pivotality_index
from src.sankey_analysis import build_sankey_timeline
from src.network_viz import plot_bloc_sankey
import json

logger = logging.getLogger(__name__)
bp = Blueprint("analysis", __name__)


@bp.route("/clustering", methods=["POST"])
@cached_api
@with_analysis_slot
def run_clustering():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)

        start_year, end_year = validate_year_range(data)
        try:
            num_clusters = int(data.get("num_clusters", 10))
            if not (2 <= num_clusters <= 50):
                raise ValueError("Number of clusters must be between 2 and 50")
        except (ValueError, TypeError):
            return make_error("Invalid number of clusters", 400)

        include_stability = bool(data.get("include_stability", True))
        stability_bootstraps = max(
            4,
            min(
                50, int(data.get("stability_bootstraps", DEFAULT_STABILITY_BOOTSTRAPS))
            ),
        )

        vote_matrix, _, df_filtered = preprocess_for_similarity(
            get_df(), start_year, end_year
        )
        if vote_matrix is None or vote_matrix.empty:
            return make_error("No data for specified period", 400)

        similarity_matrix = calculate_similarity(vote_matrix)
        if similarity_matrix is None or similarity_matrix.empty:
            return make_error("Insufficient voting data for similarity", 400)

        sim_countries = similarity_matrix.index.tolist()
        if len(sim_countries) < 2:
            return make_error("Need at least two countries to cluster", 400)

        effective_clusters = min(num_clusters, len(sim_countries))
        clusters, final_n, cluster_labels = perform_clustering(
            similarity_matrix, effective_clusters, sim_countries
        )
        if clusters is None:
            return make_error("Clustering failed for selected parameters", 400)

        cluster_label_map = label_clusters(
            clusters,
            similarity_matrix,
            df_filtered if df_filtered is not None else get_df(),
            name_lookup=country_names(),
        )

        result = {
            "num_clusters": final_n,
            "clusters": {str(k): sorted(list(v)) for k, v in clusters.items()},
            "cluster_labels": {str(k): v for k, v in cluster_label_map.items()},
            "meta": get_method_metadata(
                start_year, end_year, context={"analysis": "clustering"}
            ),
        }
        if include_stability:
            result["stability"] = compute_cluster_stability(
                vote_matrix, effective_clusters, stability_bootstraps
            )

        return jsonify(result)
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Clustering error", exc)


@bp.route("/soft-power", methods=["POST"])
@cached_api
@with_analysis_slot
def calculate_soft_power():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)

        start_year, end_year = validate_year_range(data)
        vote_matrix, _, df_filtered = preprocess_for_similarity(
            get_df(), start_year, end_year
        )
        if vote_matrix is None or vote_matrix.empty:
            return make_error("No data for specified period", 400)

        network = VotingNetwork(vote_matrix)
        network.build_graph()
        if network.graph.number_of_nodes() == 0:
            return make_error("Unable to build network for selected period", 400)

        centrality_metrics = network.calculate_centrality_metrics()
        calculator = SoftPowerCalculator(network, df_filtered, centrality_metrics)
        soft_power_scores = calculator.aggregate_soft_power_score()

        return jsonify(
            {
                "scores": soft_power_scores.head(20).to_dict(),
                "centrality": {
                    "pagerank": centrality_metrics["pagerank"].head(10).to_dict(),
                    "betweenness": centrality_metrics["betweenness"].head(10).to_dict(),
                },
                "meta": get_method_metadata(
                    start_year, end_year, context={"analysis": "soft_power"}
                ),
            }
        )
    except Exception as exc:
        return make_server_error("Soft power error", exc)


@bp.route("/abstention", methods=["POST"])
@cached_api
@with_analysis_slot
def abstention_analysis():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)

        start_year, end_year = validate_year_range(data)

        # We need the full DataFrame, not just the vote matrix for abstention logic
        # since we want to count raw votes. The service handles its own filtering
        # but let's pass a safe minimum vote threshold
        min_votes = int(data.get("min_votes", 10))

        result = calculate_abstention_rates(get_df(), start_year, end_year, min_votes)
        if "error" in result:
            return make_error(result["error"], 400)

        result["meta"] = get_method_metadata(
            start_year, end_year, context={"analysis": "abstention"}
        )
        return jsonify(result)

    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Abstention analysis error", exc)


@bp.route("/pivotality", methods=["POST"])
@cached_api
@with_analysis_slot
def pivotality_analysis():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)

        start_year, end_year = validate_year_range(data)
        margin_threshold = float(data.get("margin_threshold", 0.15))
        min_votes = int(data.get("min_votes", 10))

        result = calculate_pivotality_index(
            get_df(), start_year, end_year, margin_threshold, min_votes
        )
        if "error" in result:
            return make_error(result["error"], 400)

        result["meta"] = get_method_metadata(
            start_year,
            end_year,
            context={"analysis": "pivotality", "margin_threshold": margin_threshold},
        )
        return jsonify(result)

    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Pivotality analysis error", exc)


@bp.route("/bloc-timeline", methods=["POST"])
@cached_api
@with_analysis_slot
def bloc_timeline_analysis():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)

        start_year, end_year = validate_year_range(data)
        window = int(data.get("window", 5))
        num_clusters = int(data.get("num_clusters", 4))

        sankey_data = build_sankey_timeline(
            get_df(), start_year, end_year, window, num_clusters
        )
        if "error" in sankey_data:
            return make_error(sankey_data["error"], 400)

        fig = plot_bloc_sankey(sankey_data)
        payload = json.loads(fig.to_json())

        payload["meta"] = get_method_metadata(
            start_year,
            end_year,
            context={
                "analysis": "bloc_timeline",
                "window": window,
                "num_clusters": num_clusters,
            },
        )
        return jsonify(payload)

    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Bloc timeline analysis error", exc)


@bp.route("/compare", methods=["POST"])
@with_analysis_slot
def compare_countries():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)

        country_a = normalize_country_code(data.get("country_a"))
        country_b = normalize_country_code(data.get("country_b"))
        start_year, end_year = validate_year_range(data)

        from src.divergence_analysis import DivergenceDetector

        df_period = get_df()[
            (get_df()["year"] >= start_year) & (get_df()["year"] <= end_year)
        ].copy()
        if df_period.empty:
            return make_error("No data for specified period", 400)

        vote_matrix, _, _ = preprocess_for_similarity(df_period, start_year, end_year)
        if vote_matrix is None:
            return make_error("Failed to process data for similarity", 400)

        sim_matrix = calculate_similarity(vote_matrix)
        if sim_matrix is None:
            return make_error("Failed to calculate similarity", 400)

        if country_a not in sim_matrix.index or country_b not in sim_matrix.columns:
            return make_error(
                f"Country codes not in selected window: {country_a}, {country_b}", 400
            )

        detector = DivergenceDetector(df_period, sim_matrix)
        anomalies = detector.detect_vote_anomalies(country_a, country_b)
        similarity = float(sim_matrix.loc[country_a, country_b])

        return jsonify(
            {
                "similarity": similarity,
                "anomalies": anomalies[:50],
                "meta": get_method_metadata(
                    start_year,
                    end_year,
                    context={
                        "analysis": "country_compare",
                        "country_a": country_a,
                        "country_b": country_b,
                    },
                ),
            }
        )
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Comparison error", exc)


@bp.route("/divergence-report", methods=["POST"])
@cached_api
@with_analysis_slot
def divergence_report():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)

        countries = data.get("countries", [])
        if not countries or not isinstance(countries, list):
            return make_error("List of countries is required", 400)
        countries = [normalize_country_code(c) for c in countries]

        window = int(data.get("window", 2))
        if not (1 <= window <= 10):
            return make_error("window must be between 1 and 10", 400)

        topic = data.get("topic")

        from src.divergence_analysis import DivergenceDetector
        from app.services import get_year_bounds

        min_year, max_year = get_year_bounds()
        # Default the analysis year to the most recent year present in the
        # dataset — never a hardcoded literal.
        year = int(data.get("year") or max_year)
        if not (min_year <= year <= max_year):
            return make_error(f"year must be between {min_year} and {max_year}", 400)

        period_start = max(min_year, year - window)
        period_end = min(max_year, year + window)
        df_window = get_df()[
            (get_df()["year"] >= period_start) & (get_df()["year"] <= period_end)
        ].copy()
        if df_window.empty:
            return make_error("No data for selected divergence window", 400)

        vote_matrix, _, _ = preprocess_for_similarity(df_window, year, year)
        if vote_matrix is None:
            return make_error("Failed to process data", 400)

        sim_matrix = calculate_similarity(vote_matrix)
        if sim_matrix is None or sim_matrix.empty:
            return make_error("Failed to calculate similarity", 400)

        detector = DivergenceDetector(df_window, sim_matrix)
        report = detector.generate_divergence_report(
            countries, year, window, topic=topic
        )
        report["meta"] = get_method_metadata(
            period_start,
            period_end,
            context={
                "analysis": "divergence_report",
                "year": year,
                "window": window,
                "topic": topic,
            },
        )
        return jsonify(report)
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Divergence report error", exc)
