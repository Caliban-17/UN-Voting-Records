"""Big-picture routes: the agenda, division, alignment and landmark votes
across the whole record. Everything is whole-history, so nothing here takes
the global year window except the Washington–Beijing scatter."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from app.routes.core import make_error, make_server_error
from app.services import country_names, get_df, get_year_bounds, normalize_country_code
from src.cache_utils import cached_api
from src.story_analysis import (
    RECURRING_VOTES,
    agenda_by_year,
    alignment_scatter,
    country_story,
    division_by_year,
    landmark_votes,
    last_full_year,
    recent_votes,
    recurring_vote_series,
    resolution_vote_map,
    world_alignment_with,
)

logger = logging.getLogger(__name__)
bp = Blueprint("story", __name__)

P5 = ("USA", "RUS", "CHN", "GBR", "FRA")


@bp.route("/agenda", methods=["GET"])
@cached_api
def story_agenda():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        return jsonify(agenda_by_year(get_df()))
    except Exception as exc:
        return make_server_error("Story agenda error", exc)


@bp.route("/division", methods=["GET"])
@cached_api
def story_division():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        return jsonify(division_by_year(get_df()))
    except Exception as exc:
        return make_server_error("Story division error", exc)


@bp.route("/alignment", methods=["GET"])
@cached_api
def story_alignment():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        anchor = normalize_country_code(request.args.get("anchor", "USA")) or "USA"
        payload = world_alignment_with(get_df(), anchor, name_lookup=country_names())
        payload["anchors_available"] = list(P5)
        return jsonify(payload)
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Story alignment error", exc)


@bp.route("/scatter", methods=["GET"])
@cached_api
def story_scatter():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    min_year, max_year = get_year_bounds()
    args = request.args
    try:
        # Defaults: the last three complete years against the same width a
        # decade earlier. A single year is noisy; a wide window buries change.
        latest = last_full_year(get_df())
        end = int(args.get("end", latest))
        start = int(args.get("start", end - 2))
        base_end = args.get("base_end")
        base_start = args.get("base_start")
        if base_end is None and base_start is None and "end" not in args and "start" not in args:
            base_end, base_start = end - 10, start - 10
        base_end = int(base_end) if base_end else None
        base_start = int(base_start) if base_start else None
        if not (min_year <= start <= end <= max_year):
            return make_error(f"start/end must satisfy {min_year} <= start <= end <= {max_year}", 400)
        if (base_start is None) != (base_end is None):
            return make_error("base_start and base_end must be given together", 400)
        if base_start is not None and not (min_year <= base_start <= base_end <= max_year):
            return make_error("baseline window out of range", 400)
        return jsonify(
            alignment_scatter(
                get_df(), start, end, base_start, base_end, name_lookup=country_names()
            )
        )
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Story scatter error", exc)


@bp.route("/landmarks", methods=["GET"])
@cached_api
def story_landmarks():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        return jsonify({"landmarks": landmark_votes(get_df())})
    except Exception as exc:
        return make_server_error("Story landmarks error", exc)


@bp.route("/resolution/<int:rcid>/map", methods=["GET"])
@cached_api
def story_resolution_map(rcid: int):
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        return jsonify(resolution_vote_map(get_df(), rcid, name_lookup=country_names()))
    except ValueError as exc:
        return make_error(str(exc), 404)
    except Exception as exc:
        return make_server_error("Story resolution map error", exc)


@bp.route("/recurring/<key>", methods=["GET"])
@cached_api
def story_recurring(key: str):
    if get_df() is None:
        return make_error("Data not loaded", 500)
    if key not in RECURRING_VOTES:
        return make_error(f"Unknown recurring vote; choose from {sorted(RECURRING_VOTES)}", 404)
    try:
        return jsonify(recurring_vote_series(get_df(), key))
    except Exception as exc:
        return make_server_error("Story recurring error", exc)


@bp.route("/country/<code>", methods=["GET"])
@cached_api
def story_country(code: str):
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        return jsonify(country_story(get_df(), normalize_country_code(code), name_lookup=country_names()))
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Story country error", exc)


@bp.route("/this-week", methods=["GET"])
@cached_api
def story_this_week():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        days = max(1, min(60, int(request.args.get("days", 14))))
        as_of = request.args.get("as_of") or None
        return jsonify(recent_votes(get_df(), days=days, name_lookup=country_names(), as_of=as_of))
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Story this-week error", exc)


@bp.route("/calendar", methods=["GET"])
@cached_api
def story_calendar():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        from src.session_calendar import calendar_for

        as_of = request.args.get("as_of") or None
        return jsonify(calendar_for(get_df(), as_of=as_of))
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Story calendar error", exc)


@bp.route("/emergency", methods=["GET"])
@cached_api
def story_emergency():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        from src.story_analysis import emergency_sessions

        return jsonify({"sessions": emergency_sessions(get_df())})
    except Exception as exc:
        return make_server_error("Story emergency error", exc)
