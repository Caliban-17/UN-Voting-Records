"""
Shared application services: data store, job queue, helpers.

Everything that multiple route modules need lives here so there is a
single import point and no circular dependencies.
"""

from __future__ import annotations

import logging
import os
import re
import difflib
import threading
import time
import uuid
import json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.cache_utils import LRUCache, model_registry  # noqa: F401 – re-exported
from src.config import UN_VOTES_CSV_PATH

logger = logging.getLogger(__name__)

# ── Global data ──────────────────────────────────────────────────────────────

df_global: Optional[pd.DataFrame] = None


def get_df() -> Optional[pd.DataFrame]:
    """Retrieve the global DataFrame to avoid stale imports."""
    global df_global
    return df_global


def load_data(use_cache=True) -> bool:
    """Load UN voting data at startup.  Returns True on success."""
    global df_global
    from src.data_processing import load_and_preprocess_data

    try:
        logger.info("Loading data from %s", UN_VOTES_CSV_PATH)
        df_result, _ = load_and_preprocess_data(str(UN_VOTES_CSV_PATH))
        df_global = df_result
        reset_country_name_cache()
        logger.info("Loaded %d voting records", len(df_global))
        info = data_freshness()
        latest = info.get("latest_vote_date") or "?"
        days = info.get("days_since_latest_vote")
        if info.get("is_stale"):
            logger.warning(
                "Data appears stale: latest vote %s (%s days ago, > %s). "
                "Refresh with `python scripts/refresh_data.py`.",
                latest, days, info.get("stale_threshold_days"),
            )
        else:
            logger.info(
                "Data covers %s–%s; latest vote %s (%s days ago).",
                info.get("min_year"), info.get("max_year"), latest, days,
            )
        return True
    except Exception:
        logger.exception("Failed to load data")
        return False


def get_year_bounds() -> tuple[int, int]:
    if df_global is None or df_global.empty:
        # UN GA started voting in 1946; the upper bound is whatever year
        # we're in right now. Never hardcode a year.
        return 1946, datetime.now(timezone.utc).year
    return int(df_global["year"].min()), int(df_global["year"].max())


def data_freshness() -> dict:
    """How fresh is the underlying voting data? Used by /api/data/freshness."""
    if df_global is None or df_global.empty:
        return {
            "loaded": False,
            "records": 0,
            "min_year": None,
            "max_year": None,
            "latest_vote_date": None,
            "days_since_latest_vote": None,
            "is_stale": True,
            "stale_threshold_days": 60,
        }
    dates = pd.to_datetime(df_global["date"], errors="coerce").dropna()
    latest_dt = dates.max() if not dates.empty else None
    min_year, max_year = get_year_bounds()
    days_since = None
    if latest_dt is not None:
        days_since = int((datetime.now(timezone.utc) - latest_dt.tz_localize("UTC")).days)
    threshold = 60
    return {
        "loaded": True,
        "records": int(len(df_global)),
        "min_year": min_year,
        "max_year": max_year,
        "latest_vote_date": latest_dt.strftime("%Y-%m-%d") if latest_dt is not None else None,
        "days_since_latest_vote": days_since,
        "is_stale": days_since is not None and days_since > threshold,
        "stale_threshold_days": threshold,
        "csv_path": str(UN_VOTES_CSV_PATH),
    }


def country_codes() -> set[str]:
    if df_global is None or df_global.empty:
        return set()
    return {str(c).strip().upper() for c in df_global["country_identifier"].dropna()}


_country_name_cache: dict[str, str] | None = None


def country_names() -> dict[str, str]:
    """Map ISO-3 code -> editorial-quality short display name.

    Built once per process. Applies the hand-curated override list from
    :mod:`src.country_display` so headlines say "Iran" not
    "Iran (Islamic Republic Of)".
    """
    global _country_name_cache
    if _country_name_cache is not None:
        return _country_name_cache
    if df_global is None or df_global.empty or "country_name" not in df_global.columns:
        _country_name_cache = {}
        return _country_name_cache
    from src.country_display import display_lookup

    names = (
        df_global[["country_identifier", "country_name"]]
        .dropna()
        .drop_duplicates(subset=["country_identifier"])
    )
    raw = {
        str(row.country_identifier).strip().upper(): str(row.country_name)
        for row in names.itertuples(index=False)
    }
    _country_name_cache = display_lookup(raw)
    return _country_name_cache


def reset_country_name_cache() -> None:
    """Test helper — clear the cached lookup so the next call rebuilds it."""
    global _country_name_cache
    _country_name_cache = None


def normalize_country_code(raw: Any) -> str:
    code = str(raw or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise ValueError("Country codes must be ISO alpha-3 codes (e.g., USA, RUS).")
    available = country_codes()
    if code not in available:
        suggestions = difflib.get_close_matches(
            code, sorted(available), n=3, cutoff=0.5
        )
        hint = f"  Try: {', '.join(suggestions)}" if suggestions else ""
        raise ValueError(f"Unknown country code '{code}'.{hint}")
    return code


# ── Year / train-test validation ─────────────────────────────────────────────


def validate_year_range(data: dict) -> tuple[int, int]:
    try:
        min_year, max_year = get_year_bounds()
        start_default = max(min_year, max_year - 10)
        start_year = int(data.get("start_year", start_default))
        end_year = int(data.get("end_year", max_year))
        if start_year < min_year or end_year > max_year:
            raise ValueError(f"Year must be between {min_year} and {max_year}")
        if start_year > end_year:
            raise ValueError("start_year cannot be greater than end_year")
        return start_year, end_year
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid year range: {exc}") from exc


def validate_train_test_years(data: dict) -> tuple[int, int]:
    try:
        min_year, max_year = get_year_bounds()
        train_end = int(data.get("train_end", max(min_year, max_year - 5)))
        test_start = int(data.get("test_start", min(max_year, train_end + 1)))
        if not (min_year <= train_end <= max_year):
            raise ValueError(f"train_end must be between {min_year} and {max_year}")
        if not (min_year <= test_start <= max_year):
            raise ValueError(f"test_start must be between {min_year} and {max_year}")
        if test_start <= train_end:
            raise ValueError("test_start must be greater than train_end")
        return train_end, test_start
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid train/test year range: {exc}") from exc


# ── Job queue ────────────────────────────────────────────────────────────────

MAX_JOB_WORKERS = max(1, int(os.getenv("MAX_JOB_WORKERS", "2")))
JOB_TTL_SEC = max(60, int(os.getenv("JOB_TTL_SEC", "7200")))
job_executor = ThreadPoolExecutor(
    max_workers=MAX_JOB_WORKERS, thread_name_prefix="unv-job"
)
job_lock = threading.RLock()
job_store: dict[str, dict] = {}


def _job_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cleanup_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SEC
    with job_lock:
        stale = [
            jid for jid, p in job_store.items() if p.get("updated_epoch", 0) < cutoff
        ]
        for jid in stale:
            job_store.pop(jid, None)


def _set_job_state(job_id: str, **updates: Any) -> None:
    with job_lock:
        payload = job_store.get(job_id)
        if not payload:
            return
        payload.update(updates)
        payload["updated_at"] = _job_now_iso()
        payload["updated_epoch"] = time.time()


def start_background_job(
    job_type: str,
    params: dict,
    worker: Callable[[Callable], Any],
) -> str:
    _cleanup_jobs()
    job_id = uuid.uuid4().hex
    with job_lock:
        job_store[job_id] = {
            "id": job_id,
            "type": job_type,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "params": params,
            "result": None,
            "error": None,
            "created_at": _job_now_iso(),
            "updated_at": _job_now_iso(),
            "updated_epoch": time.time(),
        }

    def run() -> None:
        _set_job_state(job_id, status="running", progress=0.05, message="Running")
        try:
            result = worker(
                lambda progress, message: _set_job_state(
                    job_id,
                    progress=float(max(0.0, min(1.0, progress))),
                    message=str(message),
                )
            )
            _set_job_state(
                job_id,
                status="completed",
                progress=1.0,
                message="Completed",
                result=result,
                error=None,
            )
        except ValueError as exc:
            _set_job_state(
                job_id, status="failed", progress=1.0, message="Failed", error=str(exc)
            )
        except Exception as exc:
            logger.error("Background job failed (%s): %s", job_type, exc, exc_info=True)
            _set_job_state(
                job_id,
                status="failed",
                progress=1.0,
                message="Failed",
                error="Internal server error",
            )

    job_executor.submit(run)
    return job_id


# ── Method metadata ──────────────────────────────────────────────────────────


def get_method_metadata(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    context: Optional[dict] = None,
) -> dict:
    min_year, max_year = get_year_bounds()
    payload: dict = {
        "generated_at": _job_now_iso(),
        "data_window": {"min_year": min_year, "max_year": max_year},
        "selected_window": {"start_year": start_year, "end_year": end_year},
        "methods": {
            "vote_encoding": {
                "yes": 1,
                "abstain": 0,
                "no": -1,
                "missing": "excluded from pairwise comparisons",
                "abstention_note": (
                    "Abstentions are encoded as 0 (neutral) but may encode "
                    "strategic non-commitment — see abstention_rate metrics."
                ),
            },
            "similarity": "cosine similarity on country-by-resolution vote matrix",
            "clustering": "agglomerative clustering (average linkage, precomputed distance)",
            "projection": "PCA 2D variance projection (exploratory, not causal ideology axes)",
            "network": "countries as nodes, similarity thresholded edges",
            "soft_power_label": (
                "structural alignment centrality "
                "(not causal influence without additional tests)"
            ),
            "pivotality": (
                "fraction of resolutions where the country was a swing vote "
                "(resolution outcome reverses without this vote)"
            ),
            "topic_divergence": (
                "cosine similarity computed within UNBISnet topic buckets, "
                "removing agenda-composition confounders"
            ),
        },
        "caveats": [
            "Agenda composition shifts year-to-year and can affect apparent divergence.",
            "Abstentions reflect strategy, not indifference — use abstention_rate metrics.",
            "Network centrality captures structural position, not direct causal influence.",
            "PCA and UMAP projections are exploratory only; axes have no fixed meaning.",
            "Topic-adjusted similarity removes topic-composition bias but requires subject tags.",
        ],
    }
    if context:
        payload["context"] = context
    return payload


# ── Caches ───────────────────────────────────────────────────────────────────

soft_power_trends_registry = LRUCache(capacity=20, ttl_seconds=1800)
network_animation_registry = LRUCache(capacity=10, ttl_seconds=1800)
DEFAULT_STABILITY_BOOTSTRAPS = max(
    4, int(os.getenv("CLUSTER_STABILITY_BOOTSTRAPS", "12"))
)


# ── Cluster stability ─────────────────────────────────────────────────────────


def compute_cluster_stability(
    vote_matrix: pd.DataFrame,
    num_clusters: int,
    n_bootstraps: int,
) -> dict:
    from src.main import calculate_similarity, perform_clustering

    if vote_matrix is None or vote_matrix.empty:
        return {"available": False, "reason": "no data"}
    if vote_matrix.shape[0] < 2 or vote_matrix.shape[1] < 2:
        return {"available": False, "reason": "insufficient matrix size"}

    similarity = calculate_similarity(vote_matrix)
    if similarity is None or similarity.empty or similarity.shape[0] < 2:
        return {"available": False, "reason": "insufficient similarity data"}

    countries = similarity.index.tolist()
    effective_clusters = min(num_clusters, len(countries))
    if effective_clusters < 2:
        return {"available": False, "reason": "insufficient countries"}

    baseline_clusters, _, baseline_labels = perform_clustering(
        similarity, effective_clusters, countries
    )
    if baseline_clusters is None or baseline_labels is None:
        return {"available": False, "reason": "baseline clustering failed"}

    baseline_map = {
        country: int(label)
        for country, label in zip(countries, baseline_labels, strict=False)
    }
    rng = np.random.default_rng(42)
    cols = vote_matrix.columns.to_numpy()
    ari_scores: list[float] = []
    nmi_scores: list[float] = []

    for _ in range(max(1, n_bootstraps)):
        sample_cols = rng.choice(cols, size=len(cols), replace=True)
        sampled = vote_matrix.loc[:, sample_cols]
        sampled_sim = calculate_similarity(sampled)
        if sampled_sim is None or sampled_sim.empty:
            continue
        common = [c for c in countries if c in sampled_sim.index]
        if len(common) < 2:
            continue
        sampled_sim = sampled_sim.loc[common, common]
        sc, _, sl = perform_clustering(
            sampled_sim, min(effective_clusters, len(common)), common
        )
        if sc is None or sl is None:
            continue
        bl = np.array([baseline_map[c] for c in common])
        ari_scores.append(float(adjusted_rand_score(bl, np.array(sl))))
        nmi_scores.append(float(normalized_mutual_info_score(bl, np.array(sl))))

    if not ari_scores:
        return {"available": False, "reason": "bootstrap comparisons unavailable"}

    return {
        "available": True,
        "n_bootstraps": int(n_bootstraps),
        "n_effective": int(len(ari_scores)),
        "ari_mean": float(np.mean(ari_scores)),
        "ari_std": float(np.std(ari_scores)),
        "nmi_mean": float(np.mean(nmi_scores)),
        "nmi_std": float(np.std(nmi_scores)),
    }


# ── Prediction helpers ────────────────────────────────────────────────────────


def prepare_ml_split(
    train_end: int, test_start: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_global is None or df_global.empty:
        raise ValueError("Data not loaded")
    required_cols = ["country_identifier", "issue", "vote", "year"]
    missing = [c for c in required_cols if c not in df_global.columns]
    if missing:
        raise ValueError(f"Missing columns required for diagnostics: {missing}")
    df_ml = df_global[required_cols].dropna(
        subset=["country_identifier", "issue", "vote"]
    )
    train_df = df_ml[df_ml["year"] <= train_end].copy()
    test_df = df_ml[df_ml["year"] >= test_start].copy()
    return train_df, test_df


def compute_prediction_diagnostics(model: Any, train_end: int, test_start: int) -> dict:
    from src.model import prepare_features_for_model

    train_df, test_df = prepare_ml_split(train_end, test_start)
    if test_df.empty:
        return {
            "brier_score": None,
            "reliability_bins": [],
            "baselines": {},
            "coverage": {"train_samples": int(len(train_df)), "test_samples": 0},
        }

    X_test = prepare_features_for_model(
        model,
        test_df[["country_identifier", "issue", "year"]].copy(),
        prediction_year=test_start,
    )
    y_test = test_df["vote"].to_numpy()

    if not hasattr(model, "predict_proba"):
        return {
            "brier_score": None,
            "reliability_bins": [],
            "baselines": {},
            "coverage": {
                "train_samples": int(len(train_df)),
                "test_samples": int(len(test_df)),
            },
        }

    probabilities = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)
    pred_idx = np.argmax(probabilities, axis=1)
    pred_labels = classes[pred_idx]
    confidence = probabilities[np.arange(len(pred_idx)), pred_idx]
    correctness = (pred_labels == y_test).astype(float)

    encoded_truth = np.zeros_like(probabilities)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    for row_idx, label in enumerate(y_test):
        col_idx = class_to_idx.get(label)
        if col_idx is not None:
            encoded_truth[row_idx, col_idx] = 1.0

    brier_score = float(np.mean((probabilities - encoded_truth) ** 2))

    bins = np.linspace(0.0, 1.0, 11)
    reliability: list[dict] = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (
            (confidence >= lo) & (confidence <= hi)
            if i == len(bins) - 2
            else (confidence >= lo) & (confidence < hi)
        )
        if not np.any(mask):
            continue
        reliability.append(
            {
                "bin_start": float(lo),
                "bin_end": float(hi),
                "mean_confidence": float(np.mean(confidence[mask])),
                "empirical_accuracy": float(np.mean(correctness[mask])),
                "count": int(np.sum(mask)),
            }
        )

    majority_vote = train_df["vote"].mode().iloc[0] if not train_df.empty else None
    majority_acc = (
        float(np.mean(y_test == majority_vote)) if majority_vote is not None else None
    )
    country_majority = (
        train_df.groupby("country_identifier")["vote"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .to_dict()
    )
    fallback = majority_vote if majority_vote is not None else 0
    prior_preds = np.array(
        [
            country_majority.get(c, fallback)
            for c in test_df["country_identifier"].tolist()
        ]
    )
    prior_acc = float(np.mean(y_test == prior_preds)) if len(prior_preds) else None

    return {
        "brier_score": brier_score,
        "reliability_bins": reliability,
        "baselines": {
            "majority_vote_accuracy": majority_acc,
            "country_prior_accuracy": prior_acc,
        },
        "coverage": {
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
        },
    }


def compute_training_payload(
    train_end: int,
    test_start: int,
    progress: Optional[Callable] = None,
) -> dict:
    from src.model import train_vote_predictor

    cache_key = f"{train_end}_{test_start}"
    cached_bundle = model_registry.get(cache_key)
    if isinstance(cached_bundle, dict) and "model" in cached_bundle:
        metrics = cached_bundle.get("metrics", {})
        return {
            "status": "success",
            "accuracy": metrics.get("accuracy", 0.0),
            "train_samples": metrics.get("train_samples", 0),
            "test_samples": metrics.get("test_samples", 0),
            "report": metrics.get("report", {}),
            "cached": True,
            "trained_at": metrics.get("trained_at"),
            "diagnostics": metrics.get("diagnostics", {}),
            "meta": get_method_metadata(
                train_end,
                test_start,
                context={"analysis": "prediction_train", "cached": True},
            ),
        }

    if progress:
        progress(0.15, "Training vote predictor")

    model, accuracy, report, _, n_train, n_test = train_vote_predictor(
        df_global, train_end, test_start
    )
    if model is None:
        raise ValueError("Model training failed")

    if progress:
        progress(0.7, "Computing calibration diagnostics")
    diagnostics = compute_prediction_diagnostics(model, train_end, test_start)

    metrics = {
        "accuracy": float(accuracy),
        "train_samples": int(n_train),
        "test_samples": int(n_test),
        "report": report,
        "trained_at": _job_now_iso(),
        "diagnostics": diagnostics,
    }
    model_registry.put(cache_key, {"model": model, "metrics": metrics})
    return {
        "status": "success",
        "accuracy": metrics["accuracy"],
        "train_samples": metrics["train_samples"],
        "test_samples": metrics["test_samples"],
        "report": metrics["report"],
        "cached": False,
        "trained_at": metrics["trained_at"],
        "diagnostics": diagnostics,
        "meta": get_method_metadata(
            train_end,
            test_start,
            context={"analysis": "prediction_train", "cached": False},
        ),
    }


def compute_soft_power_trends_payload(
    start_year: int,
    end_year: int,
    progress: Optional[Callable] = None,
) -> dict:
    import json
    from src.soft_power import track_soft_power_over_time
    from src.network_viz import plot_soft_power_trends

    cache_key = f"{start_year}_{end_year}"
    cached = soft_power_trends_registry.get(cache_key)
    if cached:
        if progress:
            progress(1.0, "Loaded cached trends")
        return cached

    if progress:
        progress(0.1, "Computing yearly soft power traces")
    trends_df = track_soft_power_over_time(df_global, start_year, end_year)
    if trends_df.empty:
        raise ValueError("No trend data available")
    if progress:
        progress(0.85, "Rendering trend plot")
    fig = plot_soft_power_trends(trends_df)
    payload = json.loads(fig.to_json())
    payload["meta"] = get_method_metadata(
        start_year,
        end_year,
        context={
            "analysis": "soft_power_trends",
            "years_computed": int(end_year - start_year + 1),
        },
    )
    soft_power_trends_registry.put(cache_key, payload)
    return payload


def compute_network_animation_payload(
    start_year: int,
    end_year: int,
    window: str,
    progress: Optional[Callable] = None,
) -> dict:
    import json
    from src.network_analysis import build_network_over_time
    from src.network_viz import plot_network_animation

    cache_key = f"{start_year}_{end_year}_{window}"
    cached = network_animation_registry.get(cache_key)
    if cached:
        if progress:
            progress(1.0, "Loaded cached animation")
        return cached

    if progress:
        progress(0.1, f"Building networks over time ({window} windows)")

    networks = build_network_over_time(df_global, start_year, end_year, window=window)
    if not networks:
        raise ValueError("No networks could be built for the specified period.")

    # Extract raw nx.Graph for visualization
    graphs_by_time = [(label, net.graph) for label, net in networks]

    if progress:
        progress(0.7, "Rendering network animation")

    fig = plot_network_animation(graphs_by_time)
    payload = json.loads(fig.to_json())
    payload["meta"] = get_method_metadata(
        start_year,
        end_year,
        context={
            "analysis": "network_animation",
            "window": window,
            "frames": len(networks),
        },
    )

    network_animation_registry.put(cache_key, payload)
    return payload
