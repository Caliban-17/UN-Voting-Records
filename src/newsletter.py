"""
Weekly Atlas — the newsletter composer.

Produces a structured ``NewsletterEdition`` from the underlying UN-voting data.
Pure templating, deterministic, no LLM. Designed to read like an editorial
brief, not a data dump — every number is anchored to a comparison point and
woven into a sentence with named countries and named topics.

The edition is rendered by :mod:`src.newsletter_render` into Markdown or HTML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.coalition import TIER_THRESHOLDS, build_coalition
from src.country_profile import P5_REFERENCE
from src.drift_analysis import (
    _humanize_topic,
    find_alignment_drifts,
    pairwise_agreement_matrix,
)
from src.newsletter_voice import (
    SECTION_TITLES,
    humanize_topic_full,
    mover_detail_for,
    mover_headline_for,
    nut_graf as compose_nut_graf,
    pick_headline,
    pick_subhead,
    strange_bedfellows_intro,
    strange_bedfellows_one_liner,
)

logger = logging.getLogger(__name__)

# Canonical "watched topics" — narrow enough that a coalition snapshot on each
# is meaningful, broad enough to cover the perennial UN agenda. The editor can
# override via the API ``topics`` parameter.
DEFAULT_WATCHED_TOPICS: tuple[str, ...] = (
    "nuclear",
    "palestine",
    "climate",
    "human rights",
    "sanctions",
)


# ── dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class StatHighlight:
    """A single "by the numbers" callout: value, label, comparison context."""
    value: str
    label: str
    context: str


@dataclass
class WatchItem:
    """A country that moved — with a one-sentence narrative and per-P5 deltas
    so the renderer can draw a small 'fingerprint' chart."""
    country: str
    name: str
    shift_score: float
    headline: str
    detail: str
    p5_deltas: dict[str, float] = field(default_factory=dict)


@dataclass
class CoalitionMove:
    """A country whose Coalition tier changed between baseline and recent."""
    country: str
    name: str
    from_tier: str
    to_tier: str
    mean_baseline: float
    mean_recent: float


@dataclass
class CoalitionSnapshot:
    """Coalition Builder result on one watched topic, with movers vs baseline."""
    topic: str
    matched_resolutions_recent: int
    matched_resolutions_baseline: int
    predicted_recent: dict
    predicted_baseline: dict
    headline_stat: str
    movers: list[CoalitionMove]
    sample_titles: list[str]


@dataclass
class ResolutionSpotlight:
    """Most contested recent vote with vote-breakaway analysis."""
    rcid: int
    title: str
    date: str
    year: int
    yes: int
    no: int
    abstain: int
    margin_summary: str
    breakaways: list[dict]  # countries voting against their bloc's mean lean
    topic: Optional[str]


@dataclass
class QuietConvergence:
    """A pair drifting *together* in a year mostly defined by divergence."""
    country_a: str
    country_b: str
    name_a: str
    name_b: str
    baseline: float
    recent: float
    delta: float
    top_topics: list[str]


@dataclass
class BlocState:
    """Aggregate snapshot per bloc — predicted Yes/No on watched topics."""
    bloc_name: str
    member_count: int
    average_alignment_with_lead: float
    description: str


@dataclass
class LeadStory:
    """Expanded narrative around the headline drift."""
    headline: str
    body: str
    supporting_drifts: list[dict]


@dataclass
class TOCItem:
    number: int
    anchor: str
    title: str


@dataclass
class NextToWatch:
    """Forward-looking section — topics or pairs to track in the next session."""
    topic: str
    rationale: str


@dataclass
class NewsletterEdition:
    """Top-level edition object — what the API and renderers consume."""
    # Masthead
    publication: str
    edition_number: int
    edition_date: str
    edition_slug: str           # filesystem-safe id e.g. "2024-w20-argentina-turkmenistan"
    email_subject: str          # stable email subject line, ≤78 chars
    content_hash: str           # sha256(editorial content) — same data → same hash
    country_focus: Optional[str]  # ISO-3 if this is a per-country edition
    dateline: str
    byline: str

    # Period coverage
    period_label: str
    recent_year: int
    baseline_window: dict

    # Editorial
    headline: str
    subhead: str
    lede: str
    nut_graf: str
    in_this_issue: list[TOCItem]
    by_the_numbers: list[StatHighlight]
    lead_story: LeadStory
    lead_story_why_it_matters: str
    top_movers: list[WatchItem]
    top_movers_why_it_matters: str
    coalition_watch: list[CoalitionSnapshot]
    coalition_why_it_matters: str
    quiet_convergences: list[QuietConvergence]
    resolution_spotlight: Optional[ResolutionSpotlight]
    next_to_watch: list[NextToWatch]
    bloc_state: list[BlocState]

    # Apparatus
    freshness: dict
    methodology: list[str]
    sources: list[str]

    # Reusable raw artifacts so renderers can embed charts without recomputing.
    chart_payloads: dict = field(default_factory=dict)


# ── helpers ─────────────────────────────────────────────────────────────────


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _name(code: str, lookup: Optional[dict[str, str]]) -> str:
    if lookup and code in lookup:
        return lookup[code]
    return code


def _resolve_lookup(lookup: Optional[dict[str, str]]) -> dict[str, str]:
    return lookup or {}


def _topic_str(topics: list[dict]) -> str:
    """Render a list of driving-topic dicts as a comma-separated phrase."""
    names = [t["topic"] for t in (topics or [])[:2]]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return f"{names[0]} and {names[1]}"


# ── composers ───────────────────────────────────────────────────────────────


def _compute_top_movers(
    df: pd.DataFrame,
    baseline_window: dict,
    recent_year: int,
    name_lookup: dict[str, str],
    drifts: list[dict],
    top_n: int = 5,
    min_overlap: int = 10,
) -> list[WatchItem]:
    """Rank countries by total absolute alignment shift vs the P5.

    Each country's "shift score" is the sum of ``|recent − baseline|``
    over its agreement with each P5 member. We surface the named driver
    most associated with that country (top-1 driving topic from the
    biggest drift involving them).
    """
    df_base = df[
        (df["year"] >= baseline_window["start"])
        & (df["year"] <= baseline_window["end"])
    ]
    df_recent = df[df["year"] == recent_year]
    if df_base.empty or df_recent.empty:
        return []

    base_countries, base_mat, base_overlap = pairwise_agreement_matrix(
        df_base, min_overlap=min_overlap
    )
    recent_countries, recent_mat, recent_overlap = pairwise_agreement_matrix(
        df_recent, min_overlap=max(3, min_overlap // 3)
    )
    base_idx = {c: i for i, c in enumerate(base_countries)}
    recent_idx = {c: i for i, c in enumerate(recent_countries)}
    common = sorted(set(base_countries) & set(recent_countries))
    if not common:
        return []

    # scores carries the per-P5 signed delta so the renderer can draw a
    # 'fingerprint' chart (5 horizontal bars, one per Security Council power).
    scores: list[tuple[str, float, dict[str, float]]] = []
    for country in common:
        if country not in base_idx or country not in recent_idx:
            continue
        bi = base_idx[country]
        ri = recent_idx[country]
        p5_deltas: dict[str, float] = {}
        for ref in P5_REFERENCE:
            if ref == country or ref not in base_idx or ref not in recent_idx:
                continue
            base_val = base_mat[bi, base_idx[ref]]
            recent_val = recent_mat[ri, recent_idx[ref]]
            if np.isnan(base_val) or np.isnan(recent_val):
                continue
            # SIGNED delta — preserves direction for the fingerprint.
            p5_deltas[ref] = float(recent_val - base_val)
        if not p5_deltas:
            continue
        agg_score = float(sum(abs(d) for d in p5_deltas.values()))
        scores.append((country, agg_score, p5_deltas))

    scores.sort(key=lambda r: r[1], reverse=True)

    # Map country → "biggest drift this year" so we have a named topic to cite.
    biggest_drift_by_country: dict[str, dict] = {}
    for drift in drifts:
        for code in (drift["country_a"], drift["country_b"]):
            existing = biggest_drift_by_country.get(code)
            if existing is None or drift["abs_delta"] > existing["abs_delta"]:
                biggest_drift_by_country[code] = drift

    items: list[WatchItem] = []
    for country, score, p5_deltas in scores[:top_n]:
        name = _name(country, name_lookup)
        anchor = biggest_drift_by_country.get(country)
        # Use the voice module for plain-English copy.
        items.append(
            WatchItem(
                country=country, name=name, shift_score=score,
                headline=mover_headline_for(name, score * 100),
                detail=mover_detail_for(name, score * 100, anchor, name_lookup),
                p5_deltas=p5_deltas,
            )
        )
    return items


def _tier_for(mean_vote: float) -> str:
    for label, threshold in TIER_THRESHOLDS:
        if mean_vote > threshold:
            return label
    return "Champion opposed"


def _coalition_snapshot(
    df: pd.DataFrame,
    topic: str,
    recent_year: int,
    baseline_window: dict,
    name_lookup: dict[str, str],
    max_movers: int = 8,
) -> Optional[CoalitionSnapshot]:
    """Run Coalition Builder for recent year + baseline, return diffs."""
    try:
        recent = build_coalition(
            df, topic, recent_year, recent_year, name_lookup=name_lookup
        )
        baseline = build_coalition(
            df, topic, baseline_window["start"], baseline_window["end"],
            name_lookup=name_lookup,
        )
    except ValueError:
        return None
    if recent["matched_resolutions"] == 0:
        return None

    # Build country → tier maps for both periods.
    def _tier_map(report: dict) -> dict[str, tuple[str, float]]:
        out: dict[str, tuple[str, float]] = {}
        for tier, entries in report.get("tiers", {}).items():
            for entry in entries:
                out[entry["country"]] = (tier, float(entry["mean_vote"]))
        return out

    recent_tiers = _tier_map(recent)
    baseline_tiers = _tier_map(baseline)

    movers: list[CoalitionMove] = []
    for country, (recent_tier, recent_mean) in recent_tiers.items():
        baseline_entry = baseline_tiers.get(country)
        if not baseline_entry:
            continue
        baseline_tier, baseline_mean = baseline_entry
        if recent_tier == baseline_tier:
            continue
        movers.append(
            CoalitionMove(
                country=country,
                name=_name(country, name_lookup),
                from_tier=baseline_tier,
                to_tier=recent_tier,
                mean_baseline=baseline_mean,
                mean_recent=recent_mean,
            )
        )
    movers.sort(
        key=lambda m: abs(m.mean_recent - m.mean_baseline), reverse=True
    )
    movers = movers[:max_movers]

    tally_recent = recent["predicted_tally"]
    headline = (
        f"On a fresh resolution mentioning \"{topic}\", {tally_recent['yes']} "
        f"states would back Yes, {tally_recent['no']} would oppose, "
        f"{tally_recent['abstain']} would abstain "
        f"(based on {recent['matched_resolutions']} matched resolutions in "
        f"{recent_year})."
    )

    sample_titles = [
        s["title"] for s in recent.get("sample_resolutions", [])[:3] if s.get("title")
    ]
    return CoalitionSnapshot(
        topic=topic,
        matched_resolutions_recent=recent["matched_resolutions"],
        matched_resolutions_baseline=baseline["matched_resolutions"],
        predicted_recent=tally_recent,
        predicted_baseline=baseline["predicted_tally"],
        headline_stat=headline,
        movers=movers,
        sample_titles=sample_titles,
    )


def _resolution_spotlight(
    df: pd.DataFrame,
    recent_year: int,
    name_lookup: dict[str, str],
    drifts: list[dict],
) -> Optional[ResolutionSpotlight]:
    """Surface the most contested (smallest-margin) vote in the recent year."""
    df_recent = df[df["year"] == recent_year]
    if df_recent.empty:
        return None

    # Pick rcids with a real spread of votes — drop unanimous ones.
    by_rcid = df_recent.groupby("rcid")["vote"].agg(
        n_yes=lambda v: int((v == 1).sum()),
        n_no=lambda v: int((v == -1).sum()),
        n_abstain=lambda v: int((v == 0).sum()),
    )
    by_rcid = by_rcid[(by_rcid["n_yes"] > 0) & (by_rcid["n_no"] > 0)]
    if by_rcid.empty:
        return None
    by_rcid["margin"] = (by_rcid["n_yes"] - by_rcid["n_no"]).abs()
    by_rcid["spread"] = by_rcid["margin"] / (by_rcid["n_yes"] + by_rcid["n_no"])
    by_rcid = by_rcid.sort_values(["spread", "margin"])
    candidate_rcid = int(by_rcid.index[0])
    row = by_rcid.loc[candidate_rcid]

    meta_row = df_recent[df_recent["rcid"] == candidate_rcid].iloc[0]
    title_field = "descr" if "descr" in meta_row.index else "issue"
    title = str(meta_row.get(title_field, "(untitled resolution)")).strip()
    topic = (
        _humanize_topic(meta_row.get("primary_topic"))
        if "primary_topic" in meta_row.index
        else None
    )

    # Breakaways = countries whose vote on this rcid disagrees with the
    # majority of countries that share their "preferred ally" (top-aligned).
    # Cheap proxy: countries voting against the prevailing direction. For each
    # such country, attach their P5 affiliations so the reader knows the bloc.
    votes_on = df_recent[df_recent["rcid"] == candidate_rcid][
        ["country_identifier", "vote"]
    ].dropna()
    majority_vote = 1 if row["n_yes"] >= row["n_no"] else -1
    breakaway_codes = votes_on[votes_on["vote"] == -majority_vote][
        "country_identifier"
    ].astype(str).tolist()
    breakaways = [
        {"country": c, "name": _name(c, name_lookup), "vote": int(-majority_vote)}
        for c in breakaway_codes[:20]
    ]

    # Cast to int — by_rcid is a mixed-type frame so .loc returns floats.
    n_yes_i, n_no_i, n_abstain_i = (
        int(row["n_yes"]), int(row["n_no"]), int(row["n_abstain"])
    )
    abstain_word = "abstention" if n_abstain_i == 1 else "abstentions"
    margin_summary = (
        f"passed {n_yes_i}-{n_no_i} with {n_abstain_i} {abstain_word}"
        if majority_vote == 1
        else f"defeated {n_no_i}-{n_yes_i} with {n_abstain_i} {abstain_word}"
    )

    return ResolutionSpotlight(
        rcid=candidate_rcid,
        title=title[:200],
        date=str(meta_row.get("date", ""))[:10],
        year=int(meta_row.get("year", recent_year)),
        yes=int(row["n_yes"]),
        no=int(row["n_no"]),
        abstain=int(row["n_abstain"]),
        margin_summary=margin_summary,
        breakaways=breakaways,
        topic=topic,
    )


def _quiet_convergences(
    df: pd.DataFrame,
    recent_year: int,
    baseline_window: dict,
    name_lookup: dict[str, str],
    n: int = 2,
) -> list[QuietConvergence]:
    """Find the largest UP drifts when the broader year is mostly DOWN."""
    convergences = find_alignment_drifts(
        df,
        recent_year=recent_year,
        baseline_window=recent_year - baseline_window["start"],
        top_n=8,
        direction="up",
        min_baseline_overlap=15,
        min_recent_overlap=5,
    )
    out: list[QuietConvergence] = []
    for c in convergences[:n]:
        out.append(
            QuietConvergence(
                country_a=c["country_a"],
                country_b=c["country_b"],
                name_a=_name(c["country_a"], name_lookup),
                name_b=_name(c["country_b"], name_lookup),
                baseline=float(c["baseline_agreement"]),
                recent=float(c["recent_agreement"]),
                delta=float(c["delta"]),
                top_topics=[
                    humanize_topic_full(t["topic"])
                    for t in c.get("driving_topics", [])[:2]
                    if t.get("topic")
                ],
            )
        )
    return out


def _bloc_state(
    coalition_snapshots: list[CoalitionSnapshot],
) -> list[BlocState]:
    """Roll the coalition tallies up into a per-bloc summary across topics."""
    if not coalition_snapshots:
        return []
    aggregate = {"Yes side": 0, "No side": 0, "Abstain side": 0}
    for snap in coalition_snapshots:
        aggregate["Yes side"] += int(snap.predicted_recent.get("yes", 0))
        aggregate["No side"] += int(snap.predicted_recent.get("no", 0))
        aggregate["Abstain side"] += int(snap.predicted_recent.get("abstain", 0))
    total_topics = max(1, len(coalition_snapshots))
    return [
        BlocState(
            bloc_name=label,
            member_count=count // total_topics,
            average_alignment_with_lead=count / max(1, sum(aggregate.values())),
            description=(
                f"{count // total_topics} states on average back the "
                f"{label.split()[0].lower()} on the watched topics this period."
            ),
        )
        for label, count in aggregate.items()
    ]


def _chart_top_drifts(drifts: list[dict], name_lookup: dict[str, str]) -> dict:
    """Plotly JSON for a horizontal bar chart of top drifts — embedded inline."""
    if not drifts:
        return {}
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for d in drifts[:10][::-1]:  # reverse so biggest is on top
        labels.append(
            f"{_name(d['country_a'], name_lookup)} ↔ {_name(d['country_b'], name_lookup)}"
        )
        values.append(d["delta"] * 100)
        colors.append("#2a9d57" if d["delta"] > 0 else "#c64141")
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": values,
                "y": labels,
                "marker": {"color": colors},
                "hovertemplate": "%{y}: %{x:.0f} pts<extra></extra>",
            }
        ],
        "layout": {
            "title": {"text": "Top alignment shifts (percentage points)"},
            "xaxis": {"title": "Δ percentage points", "zeroline": True},
            "yaxis": {"automargin": True},
            "template": "plotly_white",
            "margin": {"l": 220, "r": 30, "t": 50, "b": 40},
        },
    }


def _slugify(text: str, max_len: int = 60) -> str:
    """Lowercase ascii-friendly slug. Used for archive filenames."""
    import re as _re
    s = _re.sub(r"[^a-zA-Z0-9]+", "-", str(text or "")).strip("-").lower()
    return s[:max_len].rstrip("-") or "edition"


def _truncate_subject(s: str, limit: int = 78) -> str:
    """Email subject ≤78 chars by RFC 2822 convention. Truncate at word."""
    s = str(s or "").strip()
    if len(s) <= limit:
        return s
    cut = s[: limit - 1].rsplit(" ", 1)[0]
    return (cut + "…") if cut else s[: limit - 1] + "…"


MIN_RECENT_RESOLUTIONS = 20
"""Minimum unique resolutions required for a year to be eligible as the
'recent_year' anchor. Stops the pipeline from picking a barely-started
in-progress year (e.g. May 2026 having only 1 resolution) and producing
statistical nonsense."""


def pick_recent_year(df: pd.DataFrame, min_resolutions: int = MIN_RECENT_RESOLUTIONS) -> int:
    """Pick the most recent calendar year with enough voting data to analyse.

    Returns the latest year in the dataset that has at least ``min_resolutions``
    unique recorded resolutions. Falls back to the last year ever if no year
    qualifies (e.g. extremely sparse synthetic test data).
    """
    if df is None or df.empty:
        raise ValueError("Data not loaded")
    counts = df.groupby("year")["rcid"].nunique().sort_index()
    eligible = counts[counts >= min_resolutions]
    if not eligible.empty:
        return int(eligible.index.max())
    return int(df["year"].max())


def build_newsletter_edition(
    df: pd.DataFrame,
    recent_year: Optional[int] = None,
    baseline_window_years: int = 3,
    watched_topics: tuple[str, ...] = DEFAULT_WATCHED_TOPICS,
    name_lookup: Optional[dict[str, str]] = None,
    edition_date: Optional[str] = None,
    country_focus: Optional[str] = None,
) -> NewsletterEdition:
    """Compose a full edition object. Caller renders with newsletter_render.

    If ``recent_year`` is None, the latest calendar year with at least
    ``MIN_RECENT_RESOLUTIONS`` recorded resolutions is auto-picked. This
    is the right default when the workflow fires mid-cycle: it skips a
    sparse in-progress year (e.g. May 2026 having 1 resolution so far)
    and anchors the analysis on the most recent complete-enough year.
    """
    if df is None or df.empty:
        raise ValueError("Data not loaded")
    if recent_year is None:
        recent_year = pick_recent_year(df)
        logger.info("Auto-picked recent_year=%s for analysis.", recent_year)
    name_lookup = _resolve_lookup(name_lookup)
    baseline_window = {
        "start": recent_year - baseline_window_years,
        "end": recent_year - 1,
    }

    drifts = find_alignment_drifts(
        df,
        recent_year=recent_year,
        baseline_window=baseline_window_years,
        top_n=10,
        country_filter=country_focus,
    )

    # ── Lead story ─────────────────────────────────────────────────────────
    # Seed for deterministic template selection — same slug always picks
    # the same headline/subhead, so an archived edition is reproducible.
    seed_for_voice = f"{recent_year}-w{baseline_window_years}-{country_focus or 'global'}"

    if drifts:
        lead = drifts[0]
        a_name = _name(lead["country_a"], name_lookup)
        b_name = _name(lead["country_b"], name_lookup)
        topics_phrase = _topic_str(lead.get("driving_topics", []))
        # Headline + subhead now come from the voice module — gossipy, varied
        # across editions, deterministic within one.
        headline = pick_headline(lead, name_lookup, seed=seed_for_voice)
        lede_subhead = pick_subhead(lead, seed=seed_for_voice)
        direction_word = "fell" if lead["delta"] < 0 else "rose"
        on_topics = (
            f" The split tracked votes on {topics_phrase}."
            if topics_phrase else ""
        )
        # Body keeps the editorial detail — it's still a serious newsletter.
        lead_story = LeadStory(
            headline=headline,
            body=(
                f"Between the {baseline_window['start']}–{baseline_window['end']} "
                f"baseline and the {recent_year} session, "
                f"{a_name} and {b_name} moved from {_pct(lead['baseline_agreement'])} "
                f"agreement to {_pct(lead['recent_agreement'])}.{on_topics} "
                f"The pair shared {lead['n_baseline_votes']} votes in the baseline "
                f"window and {lead['n_recent_votes']} in {recent_year}, so the "
                f"sample is large enough for the shift to be meaningful rather "
                f"than statistical noise."
            ),
            supporting_drifts=drifts[1:5],
        )
        lede = lede_subhead
    else:
        headline = f"A rare quiet session at the UN"
        lead_story = LeadStory(
            headline=headline,
            body="Voting patterns held steady across the dataset this period — itself a story.",
            supporting_drifts=[],
        )
        lede = "Voting patterns at the UN General Assembly were unusually stable this period."

    # ── Top movers ─────────────────────────────────────────────────────────
    top_movers = _compute_top_movers(
        df, baseline_window, recent_year, name_lookup, drifts, top_n=5
    )

    # ── Coalition watch ────────────────────────────────────────────────────
    # Suppress snapshots that have too thin a signal to be useful — under 2
    # matched resolutions in the recent year, or a totally empty predicted
    # tally. Saves the reader from the "Sanctions: 0 yes, 0 no, 0 abstain"
    # noise that was in earlier drafts.
    coalition_watch: list[CoalitionSnapshot] = []
    for topic in watched_topics:
        snap = _coalition_snapshot(
            df, topic, recent_year, baseline_window, name_lookup
        )
        if snap is None:
            continue
        if snap.matched_resolutions_recent < 2:
            continue
        tally = snap.predicted_recent or {}
        if (tally.get("yes", 0) + tally.get("no", 0) + tally.get("abstain", 0)) == 0:
            continue
        coalition_watch.append(snap)

    # ── Resolution spotlight + quiet convergences ──────────────────────────
    spotlight = _resolution_spotlight(df, recent_year, name_lookup, drifts)
    quiet = _quiet_convergences(df, recent_year, baseline_window, name_lookup)

    # ── By the numbers ─────────────────────────────────────────────────────
    by_numbers: list[StatHighlight] = []
    if drifts:
        by_numbers.append(StatHighlight(
            value=f"{abs(drifts[0]['delta']) * 100:.0f} pts",
            label="Biggest pair shift this period",
            context=(
                f"{_name(drifts[0]['country_a'], name_lookup)} ↔ "
                f"{_name(drifts[0]['country_b'], name_lookup)}"
            ),
        ))
        n_div = sum(1 for d in drifts if d["delta"] < 0)
        n_conv = sum(1 for d in drifts if d["delta"] > 0)
        by_numbers.append(StatHighlight(
            value=f"{n_div}-{n_conv}",
            label="Divergences vs convergences (top 10)",
            context=(
                "broad-based fragmentation" if n_div >= 7
                else "tilted toward convergence" if n_conv >= 7
                else "mixed signal"
            ),
        ))
    if top_movers:
        by_numbers.append(StatHighlight(
            value=f"{top_movers[0].shift_score * 100:.0f} pts",
            label=f"Aggregate P5-alignment shift — {top_movers[0].name}",
            context="largest single-country move of the period",
        ))
    if coalition_watch:
        first_topic = coalition_watch[0]
        n_movers = len(first_topic.movers)
        by_numbers.append(StatHighlight(
            value=str(n_movers),
            label=f"Coalition tier-jumpers on \"{first_topic.topic}\"",
            context=(
                f"out of {first_topic.matched_resolutions_recent} "
                f"matched {'resolution' if first_topic.matched_resolutions_recent == 1 else 'resolutions'} "
                f"in {recent_year}"
            ),
        ))
    if spotlight:
        by_numbers.append(StatHighlight(
            value=f"{spotlight.yes}-{spotlight.no}",
            label="Most contested vote of the period",
            context=spotlight.title[:80] + ("…" if len(spotlight.title) > 80 else ""),
        ))

    # ── Freshness + methodology ────────────────────────────────────────────
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    freshness = {
        "records": int(len(df)),
        "min_year": int(df["year"].min()),
        "max_year": int(df["year"].max()),
        "latest_vote_date": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
        "days_since_latest_vote": (
            int((datetime.now(timezone.utc) - dates.max().tz_localize("UTC")).days)
            if not dates.empty else None
        ),
    }
    methodology = [
        "Per-vote agreement scored 1.0 (same vote), 0.5 (one abstention), 0.0 (opposite vote).",
        f"Baseline window {baseline_window['start']}–{baseline_window['end']} compared to {recent_year}.",
        "Topic attribution uses the change in disagreement rate per topic between baseline and recent — chronic disagreements drop out.",
        "Coalition tier shifts are computed on a per-country basis from mean vote on topic-matched resolutions; tiers run from Champion supporter (>+0.85) to Champion opposed (<−0.85).",
        "Resolution spotlight selects the smallest-margin contested vote of the recent year.",
        "Aggregate P5-alignment shift sums |Δ| of each country's agreement with each of the five permanent Security Council members.",
    ]
    sources = [
        "UN Digital Library — General Assembly voting records.",
        "Method: cosine similarity on (country × resolution) vote matrices; agglomerative clustering; topic strings from UNBISnet subject headings.",
        "Code & methodology: github repo for UN-Scrupulous.",
    ]

    # ── Editorial connective tissue ────────────────────────────────────────
    edition_dt = (
        datetime.strptime(edition_date, "%Y-%m-%d")
        if edition_date else datetime.now(timezone.utc)
    )
    # ISO week-numbering gives stable edition numbers within a year.
    iso_year, iso_week, _ = edition_dt.isocalendar()
    edition_number = int(iso_week)
    dateline = edition_dt.strftime("%B %-d, %Y").upper()
    if country_focus:
        focus_name = _name(country_focus, name_lookup)
        byline = f"Country edition — {focus_name} ({country_focus})"
        publication_label = f"UN-Scrupulous — {focus_name}"
    else:
        byline = "By Dominic Garvey"
        publication_label = "UN-Scrupulous"

    # Subhead summarises the lede in one terse line.
    if drifts:
        lead = drifts[0]
        a_name = _name(lead["country_a"], name_lookup)
        b_name = _name(lead["country_b"], name_lookup)
        subhead = (
            f"{a_name}–{b_name} alignment moved "
            f"{abs(lead['delta']) * 100:.0f} points; "
            f"{'divergence' if lead['delta'] < 0 else 'convergence'} dominates the period."
        )
    else:
        subhead = "Voting patterns held steady — a rare period of UNGA stability."

    # Nut graf — why this matters, written in plain English by mood.
    n_div = sum(1 for d in drifts if d["delta"] < 0)
    n_conv = sum(1 for d in drifts if d["delta"] > 0)
    nut_graf = compose_nut_graf(n_div, n_conv, drifts[0] if drifts else None)

    # In-this-issue TOC (numbered).
    toc_titles: list[tuple[str, str]] = [
        ("by-the-numbers", "By the Numbers"),
        ("the-shift", f"The Shift — {lead_story.headline}"),
    ]
    if top_movers:
        toc_titles.append(("top-movers", "Top Movers"))
    if coalition_watch:
        toc_titles.append(("coalition-watch", "Coalition Watch"))
    if spotlight:
        toc_titles.append(("resolution-spotlight", "Resolution Spotlight"))
    if quiet:
        toc_titles.append(("quiet-convergences", "Quiet Convergences"))
    toc_titles.append(("next-to-watch", "Next to Watch"))
    toc_titles.append(("methodology", "Methodology & Sources"))
    in_this_issue = [
        TOCItem(number=i + 1, anchor=anchor, title=title)
        for i, (anchor, title) in enumerate(toc_titles)
    ]

    # Per-section "Why it matters" cues — short, deliberately direct.
    lead_why = (
        f"A {abs(drifts[0]['delta']) * 100:.0f}-point swing between two "
        f"countries with {drifts[0]['n_baseline_votes']}+ shared votes "
        f"is not random noise; it is foreign policy moving in real time."
    ) if drifts else "No significant moves — but absence of motion can itself be news."
    movers_why = (
        "The Top Movers list is the diplomat's shortlist: countries whose "
        "aggregate alignment with the P5 has shifted the most. Their next "
        "vote is the one that changes coalitions."
    )
    coalition_why = (
        "Coalition Watch is the lobby-target table. A country that "
        "jumped from supporter to fence-sitter is the country worth a phone call."
    )

    # Forward-looking "Next to Watch" — derive from topics with the largest
    # tier-jumper count, plus the spotlight's topic if any. We rotate
    # through three rationale templates (seeded by topic) so the section
    # doesn't read as five copies of the same sentence.
    next_to_watch: list[NextToWatch] = []
    coalition_sorted = sorted(
        coalition_watch, key=lambda s: len(s.movers), reverse=True
    )

    _NEXT_TEMPLATES = (
        # "Fresh resolution would test…" — the testable hypothesis frame.
        "{n} states changed tier on \"{topic}\" this period — notably "
        "{example} ({from_tier} → {to_tier}). A fresh resolution would "
        "test whether the shift is structural.",
        # "Coalition math…" — the diplomacy frame.
        "{n} states shifted their stance on \"{topic}\" — including "
        "{example} ({from_tier} → {to_tier}). The next vote on this topic "
        "is now a coalition-math problem, not a foregone conclusion.",
        # "Watch for…" — the editor's frame.
        "Watch {topic}: {n} states moved tier this period, "
        "{example} most dramatically ({from_tier} → {to_tier}). "
        "The next vote will tell us whether this is drift or doctrine.",
    )

    import hashlib as _hashlib
    for snap in coalition_sorted[:3]:
        if not snap.movers:
            continue
        example = snap.movers[0]
        # Deterministic template choice keyed on topic so each topic always
        # gets the same framing across editions.
        tmpl = _NEXT_TEMPLATES[
            int(_hashlib.sha256(snap.topic.encode()).hexdigest()[:8], 16)
            % len(_NEXT_TEMPLATES)
        ]
        next_to_watch.append(
            NextToWatch(
                topic=snap.topic.title(),
                rationale=tmpl.format(
                    n=len(snap.movers),
                    topic=snap.topic,
                    example=example.name,
                    from_tier=example.from_tier,
                    to_tier=example.to_tier,
                ),
            )
        )
    if spotlight and not any(
        s.topic.lower() in (spotlight.topic or "").lower() for s in coalition_watch
    ):
        if spotlight.topic:
            next_to_watch.append(NextToWatch(
                topic=spotlight.topic,
                rationale=(
                    f"This period's most contested vote — "
                    f"{spotlight.margin_summary} — surfaces a fault line "
                    f"worth a dedicated drill-down."
                ),
            ))

    chart_payloads = {
        "top_drifts_plotly": _chart_top_drifts(drifts, name_lookup),
        # Raw drift list — used by the SVG renderer for inline email charts.
        "top_drifts_raw": drifts,
        "name_lookup": name_lookup,
    }

    # Stable email subject — same headline becomes the same subject across
    # composer reruns, which is what makes idempotent auto-publish safe.
    if country_focus:
        subject_prefix = f"UN-Scrupulous №{edition_number} · {country_focus}"
    else:
        subject_prefix = f"UN-Scrupulous №{edition_number}"
    email_subject = _truncate_subject(f"{subject_prefix}: {headline}")

    # Filesystem slug for the archive — week + (optional country) + lead pair.
    slug_parts = [
        f"{edition_dt.year:04d}-w{edition_number:02d}",
    ]
    if country_focus:
        slug_parts.append(country_focus.lower())
    if drifts:
        slug_parts.append(_slugify(
            f"{drifts[0]['country_a']}-{drifts[0]['country_b']}", max_len=40
        ))
    edition_slug = "-".join(slug_parts)

    # Deterministic content hash — depends ONLY on editorial output, not on
    # time-varying things like edition_date. Two runs of the same data + the
    # same recent_year + the same window produce the same hash, so the
    # publish workflow can refuse to ship a duplicate edition.
    import hashlib as _hashlib
    import json as _json

    content_payload = {
        "recent_year": int(recent_year),
        "baseline_window": baseline_window,
        "country_focus": country_focus,
        "watched_topics": sorted(watched_topics),
        # The data signature — pair, magnitudes, and topic drivers.
        "drifts": [
            {
                "a": d["country_a"], "b": d["country_b"],
                "delta": round(float(d["delta"]), 4),
                "topics": [t["topic"] for t in (d.get("driving_topics") or [])[:3]],
            }
            for d in drifts
        ],
        # Coalition movers per topic — the other major editorial input.
        "coalition_movers": [
            {
                "topic": snap.topic,
                "movers": [
                    {"c": m.country, "from": m.from_tier, "to": m.to_tier}
                    for m in snap.movers
                ],
            }
            for snap in coalition_watch
        ],
        # Spotlight rcid alone — same rcid = same spotlight, same hash.
        "spotlight_rcid": spotlight.rcid if spotlight else None,
    }
    content_hash = _hashlib.sha256(
        _json.dumps(content_payload, sort_keys=True, default=str).encode()
    ).hexdigest()

    return NewsletterEdition(
        publication=publication_label,
        edition_number=edition_number,
        edition_date=edition_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        edition_slug=edition_slug,
        email_subject=email_subject,
        content_hash=content_hash,
        country_focus=country_focus,
        dateline=dateline,
        byline=byline,
        # Self-explanatory phrasing. We're saying "the latest full session
        # we have data for is YEAR, and we're comparing it against the N
        # years before". That framing works whether the newsletter ships
        # mid-cycle or right after the session ends.
        period_label=(
            f"The {recent_year} UN session vs the "
            f"{baseline_window['start']}–{baseline_window['end']} baseline"
        ),
        recent_year=int(recent_year),
        baseline_window=baseline_window,
        headline=headline,
        subhead=subhead,
        lede=lede,
        nut_graf=nut_graf,
        in_this_issue=in_this_issue,
        by_the_numbers=by_numbers,
        lead_story=lead_story,
        lead_story_why_it_matters=lead_why,
        top_movers=top_movers,
        top_movers_why_it_matters=movers_why,
        coalition_watch=coalition_watch,
        coalition_why_it_matters=coalition_why,
        quiet_convergences=quiet,
        resolution_spotlight=spotlight,
        next_to_watch=next_to_watch,
        bloc_state=_bloc_state(coalition_watch),
        freshness=freshness,
        methodology=methodology,
        sources=sources,
        chart_payloads=chart_payloads,
    )


def edition_to_dict(edition: NewsletterEdition) -> dict:
    """Convert dataclass tree to plain dict for JSON serialization."""
    from dataclasses import asdict

    return asdict(edition)
