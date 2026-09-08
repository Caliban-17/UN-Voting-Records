"""
The General Assembly's annual rhythm — for saying what is coming, not only
what happened.

Two sources:

* A hand-written calendar of the session's phases: opening in mid-September,
  the general debate, the six Main Committees through October–November,
  plenary adoptions in December, then resumed sessions and the emergency
  special sessions that can be called at any time.
* The record itself: each recurring resolution's typical date is the median
  calendar day of its last five votes, so "the Cuba embargo vote usually
  comes on 31 October" is measured, not remembered.

One fact shapes every note: this dataset records **plenary** roll-calls. The
Main Committees vote first (mostly in November), but those votes are not in
the record; the plenary re-votes on the committees' reports in December,
which is why six in seven recorded votes since 2015 fall in that one month.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd

from src.story_analysis import RECURRING_VOTES, recurring_vote_series, resolutions_table

# (start month-day, end month-day, key, label, note, what to expect)
PHASES: list[tuple[tuple[int, int], tuple[int, int], str, str, str, list[str]]] = [
    (
        (1, 1), (2, 28), "recess", "Recess",
        "The main part of the session closed in December. Plenary votes are rare "
        "until spring, though an emergency special session can be convened at any time.",
        ["An emergency special session on Gaza or Ukraine, if one is called"],
    ),
    (
        (3, 1), (5, 31), "resumed", "Resumed session",
        "The Fifth Committee (budget) meets in March and May; the plenary votes only "
        "occasionally, usually on a budget text or in an emergency special session.",
        ["Budget and financing texts", "Emergency special sessions, if called"],
    ),
    (
        (6, 1), (8, 31), "summer", "Between sessions",
        "Recorded votes are scarce over the summer and usually come from emergency "
        "special sessions or one-off plenary meetings.",
        ["Emergency special sessions, if called"],
    ),
    (
        (9, 1), (9, 14), "opening", "Session opening",
        "The new session opens on the third Tuesday of September and the general "
        "debate follows. Recorded votes before October are unusual.",
        ["The opening plenary and any high-level meetings"],
    ),
    (
        (9, 15), (10, 15), "debate", "General debate and committee start",
        "Heads of state and government speak, and the six Main Committees begin work. "
        "Recorded plenary votes stay rare until the committees report.",
        ["Occasional plenary votes on the outcomes of high-level meetings"],
    ),
    (
        (10, 16), (11, 15), "committees", "Committee voting season",
        "The First Committee (disarmament) votes on its drafts from late October, with "
        "the Fourth (decolonization and the Middle East) and Third (human rights) to "
        "follow. Committee votes are not in this record; the plenary re-votes on their "
        "reports in December.",
        [
            "The annual Cuba embargo vote in the plenary around the end of October",
            "First Committee disarmament drafts, recorded when the plenary adopts them in December",
        ],
    ),
    (
        (11, 16), (11, 30), "late_committees", "Late committee season",
        "The Third Committee's country resolutions (Iran, North Korea, Myanmar, Syria, "
        "Ukraine) and the Fourth Committee's Israel–Palestine texts go to committee "
        "votes; the plenary follows in the first half of December.",
        ["Human-rights country resolutions in committee", "Israel–Palestine and UNRWA texts in committee"],
    ),
    (
        (12, 1), (12, 24), "plenary", "Plenary adoptions",
        "The busiest weeks of the year: the plenary votes on every committee report. "
        "Six in seven of the year's recorded votes fall in December.",
        [
            "UNRWA, the Golan and the nuclear ban treaty in the first week",
            "Palestinian self-determination and the Nazism resolution around 16 December",
            "Human rights in Iran and unilateral sanctions around 18 December",
        ],
    ),
    (
        (12, 25), (12, 31), "recess", "Recess",
        "The main part of the session has closed; nothing is scheduled until the "
        "resumed session in March.",
        [],
    ),
]


def _md(d: date) -> tuple[int, int]:
    return (d.month, d.day)


def session_phase(day: date) -> dict:
    """The phase of the Assembly's year that ``day`` falls in."""
    md = _md(day)
    for start, end, key, label, note, expect in PHASES:
        if start <= md <= end:
            return {"key": key, "label": label, "note": note, "expect": list(expect)}
    raise ValueError(f"No phase covers {day}")  # PHASES cover the whole year


def _as_timestamp(as_of: Optional[str | date | datetime]) -> pd.Timestamp:
    if as_of is None:
        return pd.Timestamp.utcnow().tz_localize(None).normalize()
    return pd.Timestamp(as_of).normalize()


def upcoming_recurring_votes(
    df: pd.DataFrame,
    as_of: Optional[str | date | datetime] = None,
    horizon_days: int = 35,
    lookback: int = 5,
) -> list[dict]:
    """Recurring resolutions whose typical date falls within ``horizon_days``
    of ``as_of`` and which have not yet been voted on this year, soonest
    first. The typical date is the median calendar day of the last
    ``lookback`` votes."""
    today = _as_timestamp(as_of)
    res = resolutions_table(df).set_index("rcid")
    dates_by_rcid = pd.to_datetime(res["date"], errors="coerce")
    out = []
    for key in RECURRING_VOTES:
        series = recurring_vote_series(df, key)["series"]
        if not series:
            continue
        recent = series[-lookback:]
        days = sorted(
            int(dates_by_rcid.get(s["rcid"]).dayofyear)
            for s in recent
            if s["rcid"] in dates_by_rcid.index and pd.notna(dates_by_rcid.get(s["rcid"]))
        )
        if not days:
            continue
        median_doy = days[len(days) // 2]
        already_this_year = any(
            s["year"] == today.year and dates_by_rcid.get(s["rcid"]) is not None
            and dates_by_rcid.get(s["rcid"]) <= today
            for s in series
        )
        year = today.year + (1 if already_this_year else 0)
        expected = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=median_doy - 1)
        if expected < today:
            expected = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=median_doy - 1)
        days_until = int((expected - today).days)
        if days_until > horizon_days:
            continue
        last = series[-1]
        out.append({
            "key": key,
            "label": RECURRING_VOTES[key]["label"],
            "typical_date": expected.strftime("%-d %B"),
            "days_until": days_until,
            "last": {"year": last["year"], "yes": last["yes"], "no": last["no"], "abstain": last["abstain"]},
        })
    out.sort(key=lambda u: u["days_until"])
    return out


def calendar_for(
    df: pd.DataFrame,
    as_of: Optional[str | date | datetime] = None,
    horizon_days: int = 35,
) -> dict:
    """Where the session stands on ``as_of`` and what is due next."""
    today = _as_timestamp(as_of)
    phase = session_phase(today.date())
    upcoming = upcoming_recurring_votes(df, today, horizon_days=horizon_days)
    if upcoming:
        first = upcoming[0]
        last = first["last"]
        note = (
            f"{phase['label']}. Coming up: {first['label']}, usually around "
            f"{first['typical_date']}; last time it passed {last['yes']} to {last['no']} "
            f"with {last['abstain']} abstaining."
        )
        if len(upcoming) > 1:
            note += f" Then: {upcoming[1]['label']}, around {upcoming[1]['typical_date']}."
    else:
        note = f"{phase['label']}. {phase['note']}"
    return {
        "as_of": today.strftime("%Y-%m-%d"),
        "phase": phase,
        "upcoming": upcoming,
        "note": note,
        "caveat": (
            "This record holds plenary roll-calls only; committee votes are not in it. "
            "Typical dates are the median of the last five years."
        ),
    }
