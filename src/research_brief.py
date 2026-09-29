"""Reproducible political findings, with denominators and explicit scope."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json

import pandas as pd

from src.pulse import UTC, stamp
from src.story_analysis import resolutions_table


def _percentage(value):
    return f"{value * 100:.1f}%"


def build_research_brief(df: pd.DataFrame, now: datetime | None = None) -> dict:
    """Compare eligible current YTD with the same dates in three prior years.

    A past year is not asserted to be complete: data coverage is always shown.
    Topic standardisation holds the reference-period agenda weights constant.
    It cannot hold resolution wording or preferences constant.
    """
    now = now or datetime.now(UTC)
    base = {"status": "insufficient", "findings": [], "generated_at": stamp(now)}
    if df is None or df.empty:
        return {**base, "message": "Voting data is not available; no research claims have been generated."}
    df = df.drop_duplicates(["rcid", "country_identifier"]).copy()
    res = resolutions_table(df)
    counts = res.groupby("year")["rcid"].nunique()
    latest_date = pd.to_datetime(df["date"], errors="coerce").max()
    base["data_through"] = latest_date.date().isoformat() if pd.notna(latest_date) else None
    partial = False
    cutoff_label = ""
    # Current-year work is seasonally compared, never against full prior years.
    # Require 20 recent and 20 pooled baseline resolutions before using YTD.
    if counts.get(now.year, 0) >= 20 and pd.notna(latest_date) and latest_date.year == now.year:
        cutoff = latest_date.strftime("%m-%d")
        seasonal = res[pd.to_datetime(res.date).dt.strftime("%m-%d") <= cutoff]
        prior_count = seasonal[seasonal.year.between(now.year - 3, now.year - 1)].rcid.nunique()
        if prior_count >= 20:
            partial = True
            cutoff_label = latest_date.strftime("%d %B").lstrip("0")
            res = seasonal
            df = df[df.rcid.isin(res.rcid)]
            counts = res.groupby("year")["rcid"].nunique()
    eligible = counts[(counts.index <= now.year) & (counts > 0)] if partial else counts[(counts.index < now.year) & (counts >= 20)]
    if len(eligible) < 2:
        return {**base, "message": "At least two past calendar years with 20 recorded votes each are required."}
    recent_year = int(eligible.index.max())
    baseline_years = [int(y) for y in eligible.index if recent_year - 3 <= y < recent_year]
    if not baseline_years:
        return {**base, "message": "No eligible comparison years in the preceding three years."}
    recent = res[res.year == recent_year]
    previous = res[res.year.isin(baseline_years)]
    period = f"{recent_year} versus " + (str(baseline_years[0]) if len(baseline_years) == 1 else f"{baseline_years[0]}–{baseline_years[-1]}")
    if partial:
        period = f"January–{cutoff_label} {recent_year} versus the same dates in {baseline_years[0]}–{baseline_years[-1]}"
    findings = []

    def add(key, section, title, finding, interpretation, caveat, evidence, method):
        findings.append({"id": key, "section": section, "title": title, "finding": finding,
                         "interpretation": interpretation, "caveat": caveat,
                         "evidence": evidence, "method": method, "period": period})

    # Identical country panel removes accession/composition changes from the
    # comparison. Missing votes still affect the denominator and are disclosed.
    year_sets = [set(df.loc[df.year == y, "country_identifier"]) for y in baseline_years + [recent_year]]
    common = set.intersection(*year_sets)
    panel = df[df.country_identifier.isin(common) & df.year.isin(baseline_years + [recent_year])]
    meta = res.set_index("rcid")["theme"]
    for code, name in (("USA", "the United States"), ("CHN", "China"), ("RUS", "Russia")):
        anchors = panel[(panel.country_identifier == code) & panel.vote.isin([-1, 1])].set_index("rcid").vote
        others = panel[(panel.country_identifier != code) & panel.vote.isin([-1, 1]) & panel.rcid.isin(anchors.index)].copy()
        if others.empty:
            continue
        others["same"] = others.vote.to_numpy() == anchors.reindex(others.rcid).to_numpy()
        others["theme"] = others.rcid.map(meta)
        r, b = others[others.year == recent_year], others[others.year.isin(baseline_years)]
        if r.rcid.nunique() < 20 or b.rcid.nunique() < 20:
            continue
        ra, ba = float(r.same.mean()), float(b.same.mean())
        rt, bt = r.groupby("theme").same.agg(["mean", "size"]), b.groupby("theme").same.agg(["mean", "size"])
        shared = rt.index.intersection(bt.index)
        weights = bt.loc[shared, "size"] / bt.loc[shared, "size"].sum()
        adjusted = float(((rt.loc[shared, "mean"] - bt.loc[shared, "mean"]) * weights).sum())
        delta = (ra - ba) * 100
        add(f"alignment-{code}", "The world through the voting record",
            f"Agreement with {name}: {delta:+.1f} percentage points",
            f"Other members sided with {name} in {_percentage(ra)} of shared Yes/No comparisons, against {_percentage(ba)} in the baseline.",
            f"Holding the baseline's broad topic weights constant, the change is {adjusted * 100:+.1f} points. This helps distinguish changing agenda composition from changes within topics.",
            "Voting agreement measures expressed positions, not influence, alliances or causes. Abstentions are excluded. Wording, attendance and the mix of resolutions within topics can still change.",
            {"recent_pct": round(ra * 100, 2), "baseline_pct": round(ba * 100, 2),
             "topic_adjusted_change_pp": round(adjusted * 100, 2), "recent_comparisons": len(r), "baseline_comparisons": len(b),
             "recent_resolutions": int(r.rcid.nunique()), "baseline_resolutions": int(b.rcid.nunique()),
             "common_countries": len(common), "shared_topics": len(shared),
             "recent_topic_coverage_pct": round(float(r.theme.isin(shared).mean()) * 100, 2),
             "baseline_topic_coverage_pct": round(float(b.theme.isin(shared).mean()) * 100, 2)},
            "Same-side country–resolution comparisons / all shared Yes–No comparisons; pooled baseline; fixed country panel; topic adjustment restricted to shared themes.")

    rdiv, bdiv = float(recent.divided.mean()), float(previous.divided.mean())
    add("division", "The world through the voting record",
        f"{_percentage(rdiv)} of recorded votes were divided",
        f"The share with more than 10% of Yes/No voters on the losing side changed by {(rdiv - bdiv) * 100:+.1f} points from {_percentage(bdiv)}.",
        "This measures the frequency of substantive minority opposition on the recorded agenda.",
        "This is not a measure of all global conflict or all UN disagreement. Consensus adoptions are absent, and changes in which questions reach a roll call affect the result.",
        {"recent_divided": int(recent.divided.sum()), "recent_resolutions": len(recent),
         "baseline_divided": int(previous.divided.sum()), "baseline_resolutions": len(previous)},
        "A recorded vote is divided when the larger of Yes/No is below 90% of Yes+No. Each resolution has equal weight.")

    rs, bs = recent.theme.value_counts(normalize=True), previous.theme.value_counts(normalize=True)
    themes = rs.index.union(bs.index)
    changes = rs.reindex(themes, fill_value=0) - bs.reindex(themes, fill_value=0)
    theme = changes.abs().idxmax()
    add("agenda", "The UN as an institution", f"The largest agenda shift: {theme}",
        f"This theme accounts for {_percentage(rs.get(theme, 0))} of recorded votes, against {_percentage(bs.get(theme, 0))}: {changes[theme] * 100:+.1f} percentage points.",
        "The subjects placed before members help determine the alignment patterns we observe. An apparent geopolitical shift can partly be a change in the questions being asked.",
        "Topic labels use documented keyword rules with one primary theme per resolution. Recorded-vote shares do not measure spending, operational activity or the UN's full agenda.",
        {"theme": theme, "recent_count": int((recent.theme == theme).sum()), "recent_total": len(recent),
         "baseline_count": int((previous.theme == theme).sum()), "baseline_total": len(previous)},
        "Resolution counts by primary theme / all recorded resolutions; largest absolute percentage-point change.")

    cast = panel[panel.vote.isin([-1, 0, 1])]
    r, b = cast[cast.year == recent_year], cast[cast.year.isin(baseline_years)]
    if len(r) and len(b):
        ra, ba = float((r.vote == 0).mean()), float((b.vote == 0).mean())
        add("abstention", "The UN as an institution", f"Abstentions account for {_percentage(ra)} of votes cast",
            f"The abstention share moved {(ra - ba) * 100:+.1f} points from {_percentage(ba)} across the same {len(common)} members.",
            "Abstention is a distinct way of registering a position. It should remain visible alongside Yes/No alignment.",
            "A higher abstention rate does not establish hedging or neutrality. Absences and non-voting records are excluded; explanations of vote are needed to establish motives.",
            {"recent_abstentions": int((r.vote == 0).sum()), "recent_votes_cast": len(r),
             "baseline_abstentions": int((b.vote == 0).sum()), "baseline_votes_cast": len(b)},
            "Abstentions / Yes+No+Abstain, with a country panel present in every comparison year.")

    result = {**base, "status": "ready", "title": "The world and the institution",
              "period": period, "recent_year": recent_year, "baseline_years": baseline_years,
              "findings": findings, "recent_resolutions": len(recent), "baseline_resolutions": len(previous),
              "scope": "UN General Assembly recorded votes. These are descriptive comparisons, not causal tests or a representative sample of all UN activity. " + ("Current year-to-date is compared with the same calendar dates in prior years. " if partial else "Past calendar years are used. ") + "Source completeness is not assumed.",
              "source_url": "https://github.com/Caliban-17/UN-Voting-Records",
              "coverage": [{"year": int(y), "resolutions": int(counts[y]),
                            "first_date": str(res.loc[res.year == y, "date"].min())[:10],
                            "last_date": str(res.loc[res.year == y, "date"].max())[:10]} for y in baseline_years + [recent_year]]}
    baseline_average = len(previous) / len(baseline_years)
    ratio = len(recent) / baseline_average
    result["coverage_note"] = (
        f"Recorded-vote volume differs substantially: {len(recent)} in {recent_year}, versus an average of {baseline_average:.0f} per baseline year. "
        "This may reflect changes in the agenda, recording practice or source coverage. Treat comparisons cautiously; broad topic adjustment cannot resolve all three."
        if ratio > 1.5 or ratio < (2 / 3) else None
    )
    result["content_hash"] = hashlib.sha256(json.dumps({k: v for k, v in result.items() if k != "generated_at"}, sort_keys=True).encode()).hexdigest()
    return result
