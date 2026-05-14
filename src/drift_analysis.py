"""
Drift analysis — find country pairs whose UN-voting alignment shifted most
between a baseline period and a recent year, and attribute the shift to
specific topics.

This is the core "interpretability + practical application" feature: each
result is a self-contained narrative card — "USA ↔ RUS alignment fell 31
points; driven by Ukraine resolutions" — usable verbatim in journalism,
diplomatic briefings, or research notes.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Per-vote agreement score used everywhere in the platform: 1.0 same, 0.0 opposite,
# 0.5 one-abstention. Kept consistent with src.country_profile.
_VOTE_VALUES: tuple[int, ...] = (-1, 0, 1)
_AGREEMENT_SCORE: dict[tuple[int, int], float] = {
    (a, b): (1.0 if a == b else 0.0 if a * b == -1 else 0.5)
    for a in _VOTE_VALUES
    for b in _VOTE_VALUES
}


def _vote_matrix(df_period: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Build a (country x rcid) matrix with NaN for absent votes."""
    if df_period.empty:
        return [], np.empty((0, 0))
    pivot = df_period.pivot_table(
        index="country_identifier",
        columns="rcid",
        values="vote",
        aggfunc="first",
    )
    return pivot.index.astype(str).tolist(), pivot.to_numpy(dtype=float)


def _humanize_topic(raw: str) -> str:
    """Turn 'MIDDLE EAST SITUATION' into 'Middle East situation' — UNBISnet
    subject strings are all-caps and look shouty in UI."""
    if not raw:
        return raw
    cleaned = str(raw).strip()
    if cleaned.isupper():
        return cleaned.capitalize()
    return cleaned


def pairwise_agreement_matrix(
    df_period: pd.DataFrame, min_overlap: int = 1
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Compute the full N×N mean-agreement matrix for the period in one pass.

    Returns
    -------
    countries : list of ISO-3 codes (row/column ordering)
    agreement : (n, n) float matrix; cell [i,j] is mean agreement; NaN when
                fewer than ``min_overlap`` shared rcids.
    n_overlap : (n, n) int matrix of shared-rcid counts.
    """
    countries, values = _vote_matrix(df_period)
    if not countries:
        return [], np.empty((0, 0)), np.empty((0, 0), dtype=int)

    # Sanitize: replace NaN with a sentinel value (-9) before masking. We
    # never compare to -9 again, so it acts as "not voted" everywhere.
    sanitized = np.where(np.isnan(values), -9, values)
    present_mask = (sanitized != -9)
    present_f = present_mask.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        n_overlap = (present_f @ present_f.T).astype(int)

        # Encode each vote value as a 0/1 mask, then express the agreement
        # matrix as a weighted sum of mask products — O(N^2 * R) with 3
        # cheap matmuls, far faster than enumerating ~18k pairs.
        masks: dict[int, np.ndarray] = {
            v: ((sanitized == v) & present_mask).astype(np.float64)
            for v in _VOTE_VALUES
        }
        sum_agreement = np.zeros((len(countries), len(countries)), dtype=np.float64)
        for (va, vb), score in _AGREEMENT_SCORE.items():
            if score == 0.0:
                continue
            sum_agreement += score * (masks[va] @ masks[vb].T)

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_agreement = sum_agreement / np.maximum(n_overlap, 1)
    mean_agreement[n_overlap < max(1, min_overlap)] = np.nan
    np.fill_diagonal(mean_agreement, 1.0)
    return countries, mean_agreement, n_overlap


def _disagreement_rate_by_topic(
    df_period: pd.DataFrame,
    country_a: str,
    country_b: str,
    topic_col: str,
) -> dict[str, tuple[int, int]]:
    """For (a, b), return {topic: (n_disagree, n_total_pair_votes)} in df_period."""
    pair = df_period[df_period["country_identifier"].isin([country_a, country_b])]
    if pair.empty:
        return {}
    pivot = pair.pivot_table(
        index="rcid",
        columns="country_identifier",
        values="vote",
        aggfunc="first",
    )
    if country_a not in pivot.columns or country_b not in pivot.columns:
        return {}
    pivot = pivot[[country_a, country_b]].dropna()
    if pivot.empty:
        return {}
    disagreement_mask = pivot[country_a] != pivot[country_b]
    disagree_rcids = set(pivot.index[disagreement_mask].tolist())
    all_rcids = set(pivot.index.tolist())

    topics = (
        df_period[df_period["rcid"].isin(all_rcids)]
        .drop_duplicates(subset=["rcid"])[["rcid", topic_col]]
        .dropna(subset=[topic_col])
    )
    out: dict[str, tuple[int, int]] = {}
    for topic, group in topics.groupby(topic_col):
        rcids_in_topic = set(group["rcid"].tolist())
        total = len(rcids_in_topic)
        if total == 0:
            continue
        disagree = len(rcids_in_topic & disagree_rcids)
        out[str(topic)] = (disagree, total)
    return out


def _topic_breakdown(
    df_baseline: pd.DataFrame,
    df_recent: pd.DataFrame,
    country_a: str,
    country_b: str,
    max_topics: int = 3,
    min_recent_disagreements: int = 1,
) -> list[dict]:
    """
    Return topics where (a, b) *newly* disagree most — i.e. their per-topic
    disagreement rate rose the most between baseline and recent. This is
    much more interpretable than "topics they disagree on now," because
    chronic-disagreement topics are filtered out automatically.
    """
    topic_col = "primary_topic"
    if topic_col not in df_recent.columns:
        topic_col = "issue" if "issue" in df_recent.columns else None
    if topic_col is None:
        return []

    baseline = _disagreement_rate_by_topic(df_baseline, country_a, country_b, topic_col)
    recent = _disagreement_rate_by_topic(df_recent, country_a, country_b, topic_col)
    if not recent:
        return []

    results: list[dict] = []
    for topic, (r_disagree, r_total) in recent.items():
        if r_disagree < min_recent_disagreements:
            continue
        r_rate = r_disagree / max(1, r_total)
        b_disagree, b_total = baseline.get(topic, (0, 0))
        b_rate = b_disagree / b_total if b_total else 0.0
        delta = r_rate - b_rate
        if delta <= 0:
            continue
        results.append(
            {
                "topic": _humanize_topic(topic),
                "count": r_disagree,
                "recent_rate": float(r_rate),
                "baseline_rate": float(b_rate),
                "delta_rate": float(delta),
            }
        )
    # Bigger jumps first; tie-break on absolute count so we prefer common
    # topics over rare topics with high noise.
    results.sort(key=lambda d: (d["delta_rate"], d["count"]), reverse=True)
    return results[:max_topics]


def find_alignment_drifts(
    df: pd.DataFrame,
    recent_year: Optional[int] = None,
    baseline_window: int = 5,
    top_n: int = 10,
    min_baseline_overlap: int = 15,
    min_recent_overlap: int = 5,
    direction: str = "all",
    country_filter: Optional[str] = None,
) -> list[dict]:
    """
    Surface the country pairs whose alignment shifted most between a baseline
    period and a recent year.

    Parameters
    ----------
    direction : "all", "up", or "down" — restrict to convergences / divergences.
    country_filter : if set, only return drifts involving this ISO-3 code.
    """
    if df is None or df.empty:
        return []
    if recent_year is None:
        recent_year = int(df["year"].max())
    baseline_start = recent_year - baseline_window
    baseline_end = recent_year - 1
    if baseline_start >= recent_year:
        raise ValueError("baseline_window must be at least 1")

    df_base = df[(df["year"] >= baseline_start) & (df["year"] <= baseline_end)]
    df_recent = df[df["year"] == recent_year]
    if df_base.empty or df_recent.empty:
        return []

    countries_b, agreement_b, overlap_b = pairwise_agreement_matrix(
        df_base, min_baseline_overlap
    )
    countries_r, agreement_r, overlap_r = pairwise_agreement_matrix(
        df_recent, min_recent_overlap
    )

    idx_b = {c: i for i, c in enumerate(countries_b)}
    idx_r = {c: i for i, c in enumerate(countries_r)}
    common = sorted(set(countries_b) & set(countries_r))
    if country_filter and country_filter not in common:
        return []
    iter_anchor = [country_filter] if country_filter else common

    drifts: list[dict] = []
    for ci in iter_anchor:
        for cj in common:
            if ci == cj:
                continue
            # Symmetric pair — emit only once unless filtering on a country.
            if not country_filter and cj <= ci:
                continue
            bi, bj = idx_b[ci], idx_b[cj]
            ri, rj = idx_r[ci], idx_r[cj]
            base_val = agreement_b[bi, bj]
            recent_val = agreement_r[ri, rj]
            if np.isnan(base_val) or np.isnan(recent_val):
                continue
            delta = float(recent_val - base_val)
            if direction == "up" and delta <= 0:
                continue
            if direction == "down" and delta >= 0:
                continue
            drifts.append(
                {
                    "country_a": ci,
                    "country_b": cj,
                    "baseline_agreement": float(base_val),
                    "recent_agreement": float(recent_val),
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "n_baseline_votes": int(overlap_b[bi, bj]),
                    "n_recent_votes": int(overlap_r[ri, rj]),
                }
            )

    drifts.sort(key=lambda d: d["abs_delta"], reverse=True)
    drifts = drifts[:top_n]

    # Attach topic-level interpretation for each drift (the "on what?" half).
    for drift in drifts:
        drift["driving_topics"] = _topic_breakdown(
            df_base, df_recent, drift["country_a"], drift["country_b"]
        )

    return drifts


def _format_pair(drift: dict, name_lookup: Optional[dict[str, str]]) -> str:
    a = drift["country_a"]
    b = drift["country_b"]
    if name_lookup:
        a = name_lookup.get(a, a)
        b = name_lookup.get(b, b)
    return f"{a} ↔ {b}"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def compose_drift_digest(
    drifts: list[dict],
    recent_year: int,
    baseline_window: dict,
    name_lookup: Optional[dict[str, str]] = None,
) -> str:
    """
    Compose a 3-paragraph press-release-style summary of the top drifts.

    No LLM call — pure templating. The result is suitable for daily-email
    or Slack-post automation: short, factual, with named topics and named
    countries throughout. Designed to read well copy-pasted with no edits.
    """
    if not drifts:
        return (
            f"No alignment drifts above threshold were detected for {recent_year} "
            f"(baseline {baseline_window['start']}–{baseline_window['end']}). "
            "Voting patterns held steady across the dataset."
        )

    divergences = [d for d in drifts if d["delta"] < 0]
    convergences = [d for d in drifts if d["delta"] > 0]

    # Paragraph 1 — the headline drift.
    lead = drifts[0]
    direction_word = "fell" if lead["delta"] < 0 else "rose"
    delta_pts = abs(lead["delta"]) * 100
    topics = [t["topic"] for t in lead.get("driving_topics", [])][:2]
    topic_line = ""
    if topics:
        if len(topics) == 1:
            topic_line = f", driven by {topics[0]}"
        else:
            topic_line = f", driven by {topics[0]} and {topics[1]}"
    para1 = (
        f"The biggest UN voting realignment in {recent_year} was between "
        f"{_format_pair(lead, name_lookup)}, whose alignment {direction_word} "
        f"{delta_pts:.0f} points from {_pct(lead['baseline_agreement'])} in "
        f"{baseline_window['start']}–{baseline_window['end']} to "
        f"{_pct(lead['recent_agreement'])} this year{topic_line}."
    )

    # Paragraph 2 — the broader pattern. Surface up to 4 more notable drifts.
    para2_bullets: list[str] = []
    for d in drifts[1:5]:
        arrow = "↓" if d["delta"] < 0 else "↑"
        topic_str = ""
        ts = [t["topic"] for t in d.get("driving_topics", [])][:1]
        if ts:
            topic_str = f" on {ts[0]}"
        para2_bullets.append(
            f"{_format_pair(d, name_lookup)} {arrow} {abs(d['delta']) * 100:.0f} pts{topic_str}"
        )
    if para2_bullets:
        para2 = (
            f"Other notable shifts: " + "; ".join(para2_bullets) + "."
        )
    else:
        para2 = "No other significant shifts beyond the lead pair."

    # Paragraph 3 — the texture of the year: divergences vs convergences.
    n_div = len(divergences)
    n_conv = len(convergences)
    if n_div and not n_conv:
        para3 = (
            f"Across the top {len(drifts)} moves, every pair drifted apart — a "
            f"year of broad-based divergence with no offsetting convergences."
        )
    elif n_conv and not n_div:
        para3 = (
            f"All {n_conv} top moves were convergences — alliances strengthened "
            f"across the board with no notable break-ups."
        )
    else:
        para3 = (
            f"The year split between divergences ({n_div}) and convergences "
            f"({n_conv}) among the top {len(drifts)} moves, suggesting realignment "
            f"rather than uniform polarization."
        )

    return "\n\n".join([para1, para2, para3])


def pair_percentile(
    df_period: pd.DataFrame,
    country_a: str,
    country_b: str,
    min_overlap: int = 15,
) -> Optional[dict]:
    """
    Where does the (country_a, country_b) alignment sit relative to *all*
    pairs in the same period? Returns percentile + sample-size context.
    """
    countries, agreement, overlap = pairwise_agreement_matrix(df_period, min_overlap)
    if country_a not in countries or country_b not in countries:
        return None
    idx = {c: i for i, c in enumerate(countries)}
    a, b = idx[country_a], idx[country_b]
    pair_value = agreement[a, b]
    if np.isnan(pair_value):
        return None

    # Use the upper-triangle of valid entries — exclude self-pairs and NaNs.
    iu = np.triu_indices(len(countries), k=1)
    valid = agreement[iu]
    valid = valid[~np.isnan(valid)]
    if valid.size == 0:
        return None

    percentile = float((valid <= pair_value).mean() * 100.0)
    return {
        "agreement": float(pair_value),
        "percentile": percentile,
        "n_pairs_compared": int(valid.size),
        "n_overlap": int(overlap[a, b]),
    }
