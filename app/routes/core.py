"""Core routes: index, health, data summary, insights, methods."""

from __future__ import annotations

import logging
import re
from io import BytesIO
from collections import defaultdict

from flask import Blueprint, jsonify, render_template, request, Response, send_file

from app.services import (
    country_names,
    data_freshness,
    get_df,
    get_method_metadata,
    get_year_bounds,
    normalize_country_code,
    validate_year_range,
)
from src.cache_utils import cached_api
from src.coalition import build_coalition
from src.country_profile import build_country_profile
from src.drift_analysis import compose_drift_digest, find_alignment_drifts
from src.newsletter import build_newsletter_edition, edition_to_dict
from src.newsletter_archive import (
    list_editions as archive_list_editions,
    read_edition as archive_read_edition,
    retrospective_drift_count,
    save_edition as archive_save_edition,
)
from src.newsletter_render import render_html, render_markdown, render_text
from src.main import preprocess_for_similarity, calculate_similarity, perform_clustering
from src.network_analysis import VotingNetwork
from src.soft_power import SoftPowerCalculator

logger = logging.getLogger(__name__)
bp = Blueprint("core", __name__)


def make_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def make_server_error(label: str, err: Exception):
    logger.error("%s: %s", label, err, exc_info=True)
    return make_error("Internal server error", 500)


def _build_markdown_report(start_year: int, end_year: int) -> dict:
    df = get_df()
    if df is None or df.empty:
        raise ValueError("Data not loaded")

    df_scope = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    if df_scope.empty:
        raise ValueError("No data for selected period")

    total_votes = int(len(df_scope))
    total_countries = int(df_scope["country_identifier"].nunique())
    total_resolutions = int(df_scope["rcid"].nunique())
    top_issues = df_scope["issue"].value_counts().head(10)

    vote_matrix, _, df_filtered = preprocess_for_similarity(df, start_year, end_year)
    cluster_summary: list[tuple[str, int]] = []
    soft_power_summary: list[tuple[str, float]] = []

    if vote_matrix is not None and not vote_matrix.empty:
        similarity = calculate_similarity(vote_matrix)
        if similarity is not None and not similarity.empty and len(similarity) >= 2:
            countries = similarity.index.tolist()
            clusters, _, _ = perform_clustering(
                similarity,
                min(8, len(countries)),
                countries,
            )
            if clusters:
                cluster_summary = sorted(
                    [
                        (f"Cluster {int(cid) + 1}", len(members))
                        for cid, members in clusters.items()
                    ],
                    key=lambda item: item[1],
                    reverse=True,
                )[:8]

        network = VotingNetwork(vote_matrix)
        network.build_graph()
        if network.graph.number_of_nodes() > 0:
            centrality = network.calculate_centrality_metrics()
            calculator = SoftPowerCalculator(network, df_filtered, centrality)
            scores = calculator.aggregate_soft_power_score().head(10)
            soft_power_summary = [
                (country, float(score)) for country, score in scores.items()
            ]

    meta = get_method_metadata(
        start_year,
        end_year,
        context={"analysis": "report_export"},
    )

    lines: list[str] = []
    lines.append("# UN Voting Intelligence Report")
    lines.append("")
    lines.append(f"## Coverage")
    lines.append(f"- Window: {start_year} to {end_year}")
    lines.append(f"- Total votes: {total_votes}")
    lines.append(f"- Countries represented: {total_countries}")
    lines.append(f"- Resolutions represented: {total_resolutions}")
    lines.append("")
    lines.append("## Top Issues")
    lines.append("| Issue | Vote Count |")
    lines.append("|---|---:|")
    for issue, count in top_issues.items():
        clean_issue = str(issue).replace("|", " ")
        lines.append(f"| {clean_issue} | {int(count)} |")
    lines.append("")

    lines.append("## Bloc Structure Snapshot")
    if cluster_summary:
        lines.append("| Bloc | Members |")
        lines.append("|---|---:|")
        for bloc, members in cluster_summary:
            lines.append(f"| {bloc} | {members} |")
    else:
        lines.append("No stable clustering snapshot available for this window.")
    lines.append("")

    lines.append("## Soft Power Snapshot")
    if soft_power_summary:
        lines.append("| Country | Score |")
        lines.append("|---|---:|")
        for country, score in soft_power_summary:
            lines.append(f"| {country} | {score:.4f} |")
    else:
        lines.append("Soft power scores unavailable for this window.")
    lines.append("")

    lines.append("## Methods and Caveats")
    methods = meta.get("methods", {})
    lines.append(f"- Similarity: {methods.get('similarity', '-')}")
    lines.append(f"- Clustering: {methods.get('clustering', '-')}")
    lines.append(f"- Projection: {methods.get('projection', '-')}")
    lines.append(f"- Soft power label: {methods.get('soft_power_label', '-')}")
    for caveat in meta.get("caveats", []):
        lines.append(f"- Caveat: {caveat}")

    markdown = "\n".join(lines)
    return {
        "markdown": markdown,
        "summary": {
            "start_year": start_year,
            "end_year": end_year,
            "total_votes": total_votes,
            "countries": total_countries,
            "resolutions": total_resolutions,
        },
        "top_issues": [
            {"issue": str(issue), "count": int(count)}
            for issue, count in top_issues.items()
        ],
        "clusters": [
            {"bloc": bloc, "members": members} for bloc, members in cluster_summary
        ],
        "soft_power": [{"country": c, "score": s} for c, s in soft_power_summary],
        "meta": meta,
    }


def _markdown_to_pdf(markdown: str) -> BytesIO:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap

    stream = BytesIO()
    # Conservative line wrap to fit A4 portrait.
    wrapped_lines: list[str] = []
    for line in markdown.splitlines():
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=98) or [""])

    lines_per_page = 52
    with PdfPages(stream) as pdf:
        for i in range(0, len(wrapped_lines), lines_per_page):
            page_lines = wrapped_lines[i : i + lines_per_page]
            fig = plt.figure(figsize=(8.27, 11.69), dpi=120)
            fig.patch.set_facecolor("white")
            fig.text(
                0.05,
                0.98,
                "\n".join(page_lines),
                va="top",
                ha="left",
                fontsize=8.6,
                family="monospace",
                color="#111111",
            )
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    stream.seek(0)
    return stream


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/health")
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "data_loaded": get_df() is not None,
            "record_count": len(get_df()) if get_df() is not None else 0,
        }
    )


@bp.route("/api/data/freshness")
def get_data_freshness():
    """Report data coverage + days since the most recent vote in the dataset."""
    return jsonify(data_freshness())


@bp.route("/api/data/summary")
@cached_api
def get_data_summary():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        min_year, max_year = get_year_bounds()
        summary = {
            "total_votes": len(get_df()),
            "countries": int(get_df()["country_identifier"].nunique()),
            "resolutions": int(get_df()["rcid"].nunique()),
            "year_range": {"min": min_year, "max": max_year},
            "issues": get_df()["issue"].value_counts().head(10).to_dict(),
            "meta": get_method_metadata(
                min_year, max_year, context={"analysis": "data_summary"}
            ),
        }
        return jsonify(summary)
    except Exception as exc:
        return make_server_error("Error getting summary", exc)


@bp.route("/api/insights", methods=["GET", "POST"])
def get_insights():
    import pandas as pd

    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            start_year, end_year = validate_year_range(payload)
            df_scope = get_df()[
                (get_df()["year"] >= start_year) & (get_df()["year"] <= end_year)
            ].copy()
        else:
            min_year, max_year = get_year_bounds()
            start_year, end_year = min_year, max_year
            df_scope = get_df()

        if df_scope.empty:
            return make_error("No data for selected period", 400)

        insights = []

        votes_per_year = df_scope.groupby("year")["rcid"].nunique()
        most_active_year = int(votes_per_year.idxmax())
        max_votes = int(votes_per_year.max())
        insights.append(
            {
                "type": "info",
                "title": "Peak Activity",
                "text": f"Year {most_active_year} saw the highest activity with {max_votes} resolutions.",
            }
        )

        issue_no_votes = df_scope[df_scope["vote"] == -1].groupby("issue").size()
        issue_total = df_scope.groupby("issue").size()
        controversy = (issue_no_votes / issue_total).sort_values(ascending=False)
        if not controversy.empty:
            top_issue = controversy.index[0]
            ratio = controversy.iloc[0]
            insights.append(
                {
                    "type": "warning",
                    "title": "Controversial Topic",
                    "text": f"'{top_issue}' is the most contentious, with {ratio:.1%} 'No' votes.",
                }
            )

        recent_years = sorted(df_scope["year"].unique())[-5:]
        recent_part = (
            df_scope[df_scope["year"].isin(recent_years)]
            .groupby("year")["country_identifier"]
            .nunique()
        )
        if len(recent_part) >= 2:
            trend = (
                "increasing"
                if recent_part.iloc[-1] > recent_part.iloc[0]
                else "decreasing"
            )
            insights.append(
                {
                    "type": "success" if trend == "increasing" else "warning",
                    "title": "Participation Trend",
                    "text": f"Member-state participation has been {trend} over the last 5 years.",
                }
            )

        return jsonify(
            {
                "insights": insights,
                "meta": get_method_metadata(
                    start_year, end_year, context={"analysis": "insights"}
                ),
            }
        )
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Insights error", exc)


@bp.route("/api/country/<code>/profile", methods=["GET"])
@cached_api
def country_profile(code: str):
    """Country-first narrative payload — top allies, P5 alignment, biggest splits."""
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        country = normalize_country_code(code)
        start_year, end_year = validate_year_range(request.args.to_dict())
        profile = build_country_profile(
            get_df(),
            country,
            start_year,
            end_year,
            name_lookup=country_names(),
        )
        profile["meta"] = get_method_metadata(
            start_year,
            end_year,
            context={"analysis": "country_profile", "country": country},
        )
        return jsonify(profile)
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Country profile error", exc)


@bp.route("/api/drift", methods=["GET"])
@cached_api
def alignment_drift():
    """Top alignment shifts — newsroom-ready drift cards."""
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        min_year, max_year = get_year_bounds()
        args = request.args
        recent_year = int(args.get("recent_year", max_year))
        baseline_window = int(args.get("baseline_window", 5))
        top_n = max(1, min(50, int(args.get("top", 10))))
        direction = str(args.get("direction", "all")).lower()
        if direction not in {"all", "up", "down"}:
            return make_error("direction must be one of: all, up, down", 400)
        country_filter = args.get("country")
        if country_filter:
            from app.services import normalize_country_code as _norm

            country_filter = _norm(country_filter)
        if not (min_year < recent_year <= max_year):
            return make_error(
                f"recent_year must be between {min_year + 1} and {max_year}", 400
            )
        if baseline_window < 1:
            return make_error("baseline_window must be >= 1", 400)

        drifts = find_alignment_drifts(
            get_df(),
            recent_year=recent_year,
            baseline_window=baseline_window,
            top_n=top_n,
            direction=direction,
            country_filter=country_filter,
        )
        return jsonify(
            {
                "recent_year": recent_year,
                "baseline_window": {
                    "start": recent_year - baseline_window,
                    "end": recent_year - 1,
                },
                "country_filter": country_filter,
                "direction": direction,
                "drifts": drifts,
                "meta": get_method_metadata(
                    recent_year - baseline_window,
                    recent_year,
                    context={
                        "analysis": "alignment_drift",
                        "direction": direction,
                        "country_filter": country_filter,
                    },
                ),
            }
        )
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Drift endpoint error", exc)


@bp.route("/api/newsletter/weekly", methods=["GET"])
@cached_api
def newsletter_weekly():
    """Reproducible weekly UN-voting newsletter.

    Query params:
      - recent_year (int, default: max year in data)
      - baseline_window (int, default: 3) — years to compare against
      - topics (csv str)              — override the default watched topics
      - format (json | markdown | html, default: json)
    """
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        min_year, max_year = get_year_bounds()
        args = request.args
        recent_year = int(args.get("recent_year", max_year))
        baseline_window = int(args.get("baseline_window", 3))
        fmt = str(args.get("format", "json")).lower()
        if fmt not in {"json", "markdown", "md", "html", "text", "txt"}:
            return make_error(
                "format must be one of: json, markdown, html, text", 400
            )
        if not (min_year < recent_year <= max_year):
            return make_error(
                f"recent_year must be between {min_year + 1} and {max_year}", 400
            )
        if baseline_window < 1 or baseline_window > 20:
            return make_error("baseline_window must be between 1 and 20", 400)

        topics_arg = args.get("topics")
        if topics_arg:
            watched = tuple(
                t.strip() for t in topics_arg.split(",") if t.strip()
            )[:10]
        else:
            from src.newsletter import DEFAULT_WATCHED_TOPICS

            watched = DEFAULT_WATCHED_TOPICS

        country_focus = args.get("country")
        if country_focus:
            country_focus = normalize_country_code(country_focus)

        edition = build_newsletter_edition(
            get_df(),
            recent_year=recent_year,
            baseline_window_years=baseline_window,
            watched_topics=watched,
            name_lookup=country_names(),
            country_focus=country_focus,
        )

        if fmt in ("markdown", "md"):
            body = render_markdown(edition)
            return Response(
                body,
                mimetype="text/markdown; charset=utf-8",
                headers={
                    "Content-Disposition": (
                        f"inline; filename=weekly-atlas-{recent_year}.md"
                    )
                },
            )
        if fmt == "html":
            return Response(render_html(edition), mimetype="text/html; charset=utf-8")
        if fmt in ("text", "txt"):
            return Response(
                render_text(edition),
                mimetype="text/plain; charset=utf-8",
                headers={
                    "Content-Disposition": (
                        f"inline; filename=weekly-atlas-{recent_year}.txt"
                    )
                },
            )

        payload = edition_to_dict(edition)
        payload["markdown"] = render_markdown(edition)
        payload["meta"] = get_method_metadata(
            edition.baseline_window["start"],
            edition.recent_year,
            context={"analysis": "newsletter_weekly", "topics": list(watched)},
        )
        return jsonify(payload)
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Newsletter endpoint error", exc)


@bp.route("/api/newsletter/archive", methods=["GET"])
def newsletter_archive_list():
    """List archived editions, newest first."""
    try:
        return jsonify({"editions": archive_list_editions()})
    except Exception as exc:
        return make_server_error("Archive list error", exc)


@bp.route("/api/newsletter/archive/<int:year>/<slug>.<fmt>", methods=["GET"])
def newsletter_archive_get(year: int, slug: str, fmt: str):
    """Retrieve an archived edition. Slug is filesystem-safe by construction."""
    result = archive_read_edition(year, slug, fmt)
    if result is None:
        return make_error("Archived edition not found", 404)
    mimetype, body = result
    return Response(body, mimetype=mimetype)


@bp.route("/api/newsletter/archive", methods=["POST"])
def newsletter_archive_save():
    """Compose the current edition and persist all formats to the archive.

    Same query params as ``/api/newsletter/weekly`` (recent_year,
    baseline_window, topics, country). Idempotent: re-archiving an edition
    with the same slug overwrites the files.
    """
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        min_year, max_year = get_year_bounds()
        args = request.args
        recent_year = int(args.get("recent_year", max_year))
        baseline_window = int(args.get("baseline_window", 3))
        if not (min_year < recent_year <= max_year):
            return make_error(
                f"recent_year must be between {min_year + 1} and {max_year}", 400
            )
        if baseline_window < 1 or baseline_window > 20:
            return make_error("baseline_window must be between 1 and 20", 400)
        topics_arg = args.get("topics")
        if topics_arg:
            watched = tuple(t.strip() for t in topics_arg.split(",") if t.strip())[:10]
        else:
            from src.newsletter import DEFAULT_WATCHED_TOPICS
            watched = DEFAULT_WATCHED_TOPICS
        country_focus = args.get("country")
        if country_focus:
            country_focus = normalize_country_code(country_focus)

        edition = build_newsletter_edition(
            get_df(),
            recent_year=recent_year,
            baseline_window_years=baseline_window,
            watched_topics=watched,
            name_lookup=country_names(),
            country_focus=country_focus,
        )
        written = archive_save_edition(edition)
        return jsonify({
            "slug": edition.edition_slug,
            "edition_date": edition.edition_date,
            "edition_number": edition.edition_number,
            "country_focus": edition.country_focus,
            "headline": edition.headline,
            "email_subject": edition.email_subject,
            "written": written,
        })
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Archive save error", exc)


@bp.route("/api/newsletter/retrospective", methods=["GET"])
def newsletter_retrospective():
    """Aggregate top-drift counts across the archive — drift of the year."""
    year = request.args.get("year")
    try:
        year_int = int(year) if year else None
        report = retrospective_drift_count(year=year_int)
        # Bounds for the methodology metadata — when no year filter is set,
        # use the full year range of the actual dataset rather than a literal.
        bound_lo, bound_hi = get_year_bounds()
        report["meta"] = get_method_metadata(
            year_int or bound_lo, year_int or bound_hi,
            context={"analysis": "newsletter_retrospective", "year": year_int},
        )
        return jsonify(report)
    except ValueError:
        return make_error("year must be an integer", 400)
    except Exception as exc:
        return make_server_error("Retrospective error", exc)


@bp.route("/api/events", methods=["GET"])
@cached_api
def known_events():
    """Hand-curated event annotations to overlay on time-series charts."""
    import json
    from pathlib import Path

    events_path = Path(__file__).resolve().parents[2] / "data" / "known_events.json"
    try:
        with events_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        events = payload.get("events", [])
        # Defensive: validate minimum schema before returning.
        clean: list[dict] = []
        for e in events:
            try:
                clean.append(
                    {
                        "year": int(e["year"]),
                        "label": str(e["label"]),
                        "category": str(e.get("category", "")),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        clean.sort(key=lambda x: x["year"])
        return jsonify({"events": clean})
    except FileNotFoundError:
        return jsonify({"events": []})
    except Exception as exc:
        return make_server_error("Events endpoint error", exc)


@bp.route("/api/drift/digest", methods=["GET"])
@cached_api
def alignment_drift_digest():
    """3-paragraph narrative summary of the top alignment drifts."""
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        min_year, max_year = get_year_bounds()
        args = request.args
        recent_year = int(args.get("recent_year", max_year))
        baseline_window = int(args.get("baseline_window", 5))
        top_n = max(3, min(20, int(args.get("top", 6))))
        if not (min_year < recent_year <= max_year):
            return make_error(
                f"recent_year must be between {min_year + 1} and {max_year}", 400
            )
        if baseline_window < 1:
            return make_error("baseline_window must be >= 1", 400)

        drifts = find_alignment_drifts(
            get_df(),
            recent_year=recent_year,
            baseline_window=baseline_window,
            top_n=top_n,
        )
        window = {"start": recent_year - baseline_window, "end": recent_year - 1}
        digest = compose_drift_digest(
            drifts, recent_year, window, name_lookup=country_names()
        )
        return jsonify(
            {
                "recent_year": recent_year,
                "baseline_window": window,
                "digest": digest,
                "drifts": drifts,
                "meta": get_method_metadata(
                    window["start"],
                    recent_year,
                    context={"analysis": "drift_digest"},
                ),
            }
        )
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Drift digest error", exc)


@bp.route("/api/coalition", methods=["GET"])
@cached_api
def coalition():
    """Topic-driven Coalition Builder — predicted Yes/No/Abstain tally + tiers."""
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        topic = request.args.get("topic", "").strip()
        if not topic:
            return make_error("topic query parameter is required", 400)
        if len(topic) > 80:
            return make_error("topic query too long (max 80 chars)", 400)
        start_year, end_year = validate_year_range(request.args.to_dict())
        report = build_coalition(
            get_df(),
            topic,
            start_year,
            end_year,
            name_lookup=country_names(),
        )
        report["meta"] = get_method_metadata(
            start_year, end_year, context={"analysis": "coalition", "topic": topic}
        )
        return jsonify(report)
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Coalition endpoint error", exc)


@bp.route("/api/methods")
def get_methods():
    min_year, max_year = get_year_bounds()
    return jsonify(
        get_method_metadata(min_year, max_year, context={"analysis": "methods"})
    )


@bp.route("/api/report", methods=["POST"])
def export_report():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        payload = request.get_json(silent=True) or {}
        if "start_year" in payload or "end_year" in payload:
            start_year, end_year = validate_year_range(payload)
        else:
            start_year, end_year = get_year_bounds()

        fmt = str(payload.get("format", "markdown")).strip().lower()
        if fmt not in {"markdown", "pdf", "json"}:
            return make_error("format must be one of: markdown, pdf, json", 400)

        report = _build_markdown_report(start_year, end_year)
        filename_base = f"un-voting-report-{start_year}-{end_year}"

        if fmt == "json":
            return jsonify(report)
        if fmt == "pdf":
            pdf_stream = _markdown_to_pdf(report["markdown"])
            return send_file(
                pdf_stream,
                mimetype="application/pdf",
                as_attachment=True,
                download_name=f"{filename_base}.pdf",
            )

        return Response(
            report["markdown"],
            mimetype="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename_base}.md"},
        )
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Report export error", exc)
