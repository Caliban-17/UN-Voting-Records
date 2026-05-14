"""Prediction routes: train, predict, issues list."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from flask import Blueprint, jsonify, request

from app.middleware import with_analysis_slot
from app.routes.core import make_error, make_server_error
from app.services import (
    get_df,
    validate_train_test_years,
    get_method_metadata,
    compute_training_payload,
)
from src.cache_utils import model_registry

logger = logging.getLogger(__name__)
bp = Blueprint("prediction", __name__)


@bp.route("/train", methods=["POST"])
@with_analysis_slot
def train_model():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True) or {}
        train_end, test_start = validate_train_test_years(data)
        payload = compute_training_payload(train_end, test_start)
        return jsonify(payload)
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Training error", exc)


@bp.route("/predict", methods=["POST"])
@with_analysis_slot
def predict_vote():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True)
        if not data:
            return make_error("No JSON data provided", 400)

        issue = data.get("issue")
        countries = data.get("countries", [])
        train_end, test_start = validate_train_test_years(data)
        prediction_year = data.get("prediction_year", test_start)

        if not issue:
            return make_error("issue is required", 400)
        try:
            prediction_year = int(prediction_year)
        except (TypeError, ValueError):
            return make_error("prediction_year must be an integer", 400)

        cache_key = f"{train_end}_{test_start}"
        model_entry = model_registry.get(cache_key)
        vote_model = (
            model_entry.get("model") if isinstance(model_entry, dict) else model_entry
        )

        if vote_model is None:
            return make_error("Model not trained for this period. Train first.", 400)

        if not countries:
            countries = get_df()["country_identifier"].unique().tolist()
        elif not isinstance(countries, list):
            return make_error("countries must be a list", 400)

        countries = [str(c).strip() for c in countries if str(c).strip()]
        if len(countries) > 500:
            return make_error("countries list exceeds maximum size (500)", 400)

        from src.model import predict_votes

        vote_counts, detailed_preds = predict_votes(
            vote_model,
            countries,
            issue,
            prediction_year=prediction_year,
        )
        if vote_counts is None:
            return make_error("Prediction failed", 500)

        model_meta = (
            model_entry.get("metrics", {}) if isinstance(model_entry, dict) else {}
        )
        return jsonify(
            {
                "summary": vote_counts.to_dict("records"),
                "details": detailed_preds.to_dict("records"),
                "meta": get_method_metadata(
                    train_end,
                    test_start,
                    context={
                        "analysis": "prediction_inference",
                        "issue": issue,
                        "prediction_year": prediction_year,
                        "trained_at": model_meta.get("trained_at"),
                    },
                ),
            }
        )
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Prediction error", exc)


@bp.route("/issues")
def get_issues():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        raw_issues = [str(i).strip() for i in get_df()["issue"].dropna().tolist()]
        dedup: dict[str, str] = {}
        for issue in raw_issues:
            key = re.sub(r"\s+", " ", issue).strip().lower()
            if key and key not in dedup:
                dedup[key] = issue
        issues = sorted(dedup.values())

        buckets: dict[str, list] = defaultdict(list)
        for issue in issues:
            topic = issue.split(":")[0][:60].strip() or "Other"
            buckets[topic].append(issue)

        ranked = sorted(buckets.items(), key=lambda x: len(x[1]), reverse=True)[:15]
        issue_index = [
            {"topic": topic, "count": len(entries), "sample": entries[:3]}
            for topic, entries in ranked
        ]

        return jsonify(
            {
                "issues": issues,
                "issue_index": issue_index,
                "meta": get_method_metadata(context={"analysis": "prediction_issues"}),
            }
        )
    except Exception as exc:
        return make_server_error("Error getting issues", exc)
