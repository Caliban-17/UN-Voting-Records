"""Job queue routes: status polling, train-model job, soft-power-trends job."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from app.routes.core import make_error, make_server_error
from app.services import (
    get_df,
    validate_year_range,
    validate_train_test_years,
    start_background_job,
    job_store,
    job_lock,
    compute_training_payload,
    compute_soft_power_trends_payload,
    compute_network_animation_payload,
)

logger = logging.getLogger(__name__)
bp = Blueprint("jobs", __name__)


@bp.route("/<job_id>")
def get_job_status(job_id: str):
    from app.services import get_df, _cleanup_jobs

    _cleanup_jobs()
    with job_lock:
        payload = job_store.get(job_id)
        if not payload:
            return make_error("Job not found", 404)
        response = {
            "id": payload["id"],
            "type": payload["type"],
            "status": payload["status"],
            "progress": payload["progress"],
            "message": payload["message"],
            "error": payload["error"],
            "created_at": payload["created_at"],
            "updated_at": payload["updated_at"],
        }
        if payload["status"] == "completed":
            response["result"] = payload["result"]
        return jsonify(response)


@bp.route("/train-model", methods=["POST"])
def start_train_model_job():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True) or {}
        train_end, test_start = validate_train_test_years(data)
        job_id = start_background_job(
            "prediction_train",
            {"train_end": train_end, "test_start": test_start},
            lambda progress: compute_training_payload(train_end, test_start, progress),
        )
        return jsonify({"job_id": job_id, "status": "queued"}), 202
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Failed to queue training job", exc)


@bp.route("/soft-power-trends", methods=["POST"])
def start_soft_power_trends_job():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True) or {}
        start_year, end_year = validate_year_range(data)
        job_id = start_background_job(
            "soft_power_trends",
            {"start_year": start_year, "end_year": end_year},
            lambda progress: compute_soft_power_trends_payload(
                start_year, end_year, progress
            ),
        )
        return jsonify({"job_id": job_id, "status": "queued"}), 202
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Failed to queue soft-power trends job", exc)


@bp.route("/network-animation", methods=["POST"])
def start_network_animation_job():
    if get_df() is None:
        return make_error("Data not loaded", 500)
    try:
        data = request.get_json(silent=True) or {}
        start_year, end_year = validate_year_range(data)
        window = str(data.get("window", "5Y"))

        if window not in ["1Y", "2Y", "5Y", "10Y"]:
            return make_error("Window must be one of '1Y', '2Y', '5Y', '10Y'", 400)

        job_id = start_background_job(
            "network_animation",
            {"start_year": start_year, "end_year": end_year, "window": window},
            lambda progress: compute_network_animation_payload(
                start_year, end_year, window, progress
            ),
        )
        return jsonify({"job_id": job_id, "status": "queued"}), 202
    except ValueError as exc:
        return make_error(str(exc), 400)
    except Exception as exc:
        return make_server_error("Failed to queue network animation job", exc)
