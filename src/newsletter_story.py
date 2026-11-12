"""Finding the story in an edition's numbers, and drawing it.

An edition leads with its most notable vote, not with a count of votes. A vote
is notable for how far it moved from the last time the same resolution was put
(the record holds every earlier vote), for falling short of half the
membership, or for being close. Each chart makes one point and carries that
point as its title; its form follows the story (who is missing from a vote, a
long decline, a rare event, what is due next).
"""

from __future__ import annotations

from datetime import date, datetime
from html import escape
import re

import pandas as pd

MEMBERS = 193
HALF = (MEMBERS + 1) // 2
INK, MUTED, RULE, PAPER = "#102c35", "#5d6f73", "#cbd5d3", "#fffdf8"
YES, ABSTAIN, NO, ACCENT = "#126271", "#b9b4a6", "#c8553d", "#b07d12"
MEMBER_COLOURS = {"Russia": "#c8553d", "USSR": "#c8553d", "China": "#b07d12",
                  "United States": "#126271", "United Kingdom": "#6b7f99", "France": "#6b7f99"}
# How to refer to a recurring resolution in a headline: (pattern, short name).
SHORT_NAMES = (
    (r"embargo imposed by the United States of America against Cuba", "the Cuba embargo"),
    (r"Report of the International Criminal Court", "the International Criminal Court"),
    (r"Report of the International Atomic Energy Agency", "the nuclear watchdog's report"),
    (r"glorification of Nazism", "the anti-Nazism resolution"),
    (r"Assistance to Palestine refugees", "UNRWA"),
    (r"Treaty on the Prohibition of Nuclear Weapons", "the nuclear ban treaty"),
)


def day(value: str, year: bool = False) -> str:
    """'2026-11-02' -> '2 November' (or '2 November 2026')."""
    when = datetime.strptime(str(value)[:10], "%Y-%m-%d")
    return f"{when.day} {when:%B}" + (f" {when.year}" if year else "")


def short_name(title: str) -> str:
    for pattern, name in SHORT_NAMES:
        if re.search(pattern, title, re.I):
            return name
    text = re.sub(r"^Report of the ", "the ", title.split(" : ")[0].strip())
    if len(text) > 70:
        text = text[:70].rsplit(" ", 1)[0] + "…"
    return text[:1].lower() + text[1:] if not text[:2].isupper() else text


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


# ── history from the record ─────────────────────────────────────────────────


def vote_history(df: pd.DataFrame, title: str, before: str, names: dict | None = None) -> list[dict]:
    """Earlier votes on the same resolution, oldest first, one per year, with
    the countries that voted against the latest one."""
    from src.story_analysis import _title_key, resolutions_table

    res = resolutions_table(df)
    hits = res[(res["title"].map(_title_key) == _title_key(title))
               & (pd.to_datetime(res["date"], errors="coerce") < pd.Timestamp(before))]
    hits = hits.dropna(subset=["total_yes", "total_no"]).sort_values("date").drop_duplicates("year", keep="last")
    out = [{"year": int(r.year), "yes": int(r.total_yes), "no": int(r.total_no),
            "abstain": int(r.total_abstentions) if pd.notna(r.total_abstentions) else 0,
            "rcid": int(r.rcid)} for r in hits.itertuples(index=False)]
    if out:
        rows = df[(df["rcid"] == out[-1]["rcid"]) & (df["vote"] == -1)]
        out[-1]["against"] = sorted((names or {}).get(c, c) for c in rows["country_identifier"].unique())
    return out


def enrich(decisions: list[dict], df: pd.DataFrame | None, names: dict | None = None) -> list[dict]:
    """Attach each recorded decision's earlier votes and its change since the last."""
    out = []
    for item in decisions:
        item = dict(item)
        if df is not None and item.get("tally"):
            history = vote_history(df, item["title"], item["date"], names)[-8:]
            if history:
                last = history[-1]
                item["history"] = [{k: v for k, v in h.items() if k != "rcid"} for h in history]
                item["change"] = {k: item["tally"][k] - last[k] for k in ("yes", "no", "abstain")}
        out.append(item)
    return out


def notability(item: dict) -> float:
    tally = item.get("tally")
    if not tally:
        return 0.0
    score = 0.0
    change = item.get("change")
    if change:
        score += 3 * abs(change["no"]) + abs(change["yes"]) + abs(change["abstain"])
    if tally["yes"] < HALF:
        score += 25
    sided = tally["yes"] + tally["no"]
    if sided:
        score += 40 * (1 - max(tally["yes"], tally["no"]) / sided)
    return score


def headline_for(item: dict) -> str:
    tally, change, name = item["tally"], item.get("change"), short_name(item["title"])
    last = (item.get("history") or [{}])[-1]
    if change and change["no"] >= 5 and tally["no"] >= 2 * max(1, last.get("no", 0)):
        return f"Opposition on {name} more than doubles: {tally['no']} vote against, up from {last['no']}"
    if tally["yes"] < HALF:
        return f"{_cap(name)} passes with {tally['yes']} votes, short of half the membership"
    if change and change["yes"] <= -10:
        return f"Support for {name} falls to {tally['yes']}, from {last['yes']}"
    if change and change["yes"] >= 10:
        return f"Support for {name} rises to {tally['yes']}, from {last['yes']}"
    if change and change["no"] >= 3:
        return f"{_cap(name)}: {tally['no']} now vote against, up from {last['no']}"
    return f"{_cap(name)} passes {tally['yes']} to {tally['no']}"


def comparison(item: dict) -> str:
    """'Last year: 165 to 7, with 12 abstaining.' or '' when there is no earlier vote."""
    history = item.get("history") or []
    if not history:
        return ""
    last = history[-1]
    label = "Last year" if int(item["date"][:4]) - last["year"] == 1 else f"In {last['year']}"
    return f"{label}: {last['yes']} to {last['no']}, with {last['abstain']} abstaining."


def _list(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def standfirst(lead: dict, second: dict | None) -> str:
    tally = lead["tally"]
    text = (f"On {day(lead['date'])} the General Assembly voted {tally['yes']} to {tally['no']}, "
            f"with {tally['abstain']} abstaining, on {short_name(lead['title'])}.")
    history = lead.get("history") or []
    if history:
        text += " " + comparison(lead)
        earlier = [h for h in history[:-1] if h["year"] >= history[-1]["year"] - 2]
        if earlier and (earlier[-1]["yes"], earlier[-1]["no"]) != (history[-1]["yes"], history[-1]["no"]):
            text += f" In {earlier[-1]['year']} it was {earlier[-1]['yes']} to {earlier[-1]['no']}."
        against = history[-1].get("against") or []
        if 0 < len(against) <= 8:
            text += (f" The {len(against)} against in {history[-1]['year']} were {_list(against)}; "
                     "this year's country-by-country record is not yet published.")
    if second:
        t2 = second["tally"]
        text += (f" Separately, {short_name(second['title'])} passed {t2['yes']} to {t2['no']} "
                 f"on {day(second['date'])}")
        text += f" ({comparison(second).rstrip('.').lower()})." if second.get("history") else "."
    return text


def choose(decisions: list[dict]) -> tuple[dict | None, dict | None]:
    ranked = sorted((d for d in decisions if d.get("tally")), key=notability, reverse=True)
    return (ranked[0] if ranked else None), (ranked[1] if len(ranked) > 1 and notability(ranked[1]) >= 15 else None)


# ── charts ──────────────────────────────────────────────────────────────────


def _svg(width: int, height: int, title: str, subtitle: str, body: list[str], source: str) -> str:
    return "".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="{escape((title + ". " + subtitle).strip(), quote=True)}">',
        f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
        f'<rect x="0" y="0" width="44" height="4" fill="{INK}"/>',
        f'<text x="0" y="28" font-family="Georgia,serif" font-size="19" font-weight="700" fill="{INK}">{escape(title)}</text>',
        f'<text x="0" y="48" font-family="Arial,sans-serif" font-size="12.5" fill="{MUTED}">{escape(subtitle)}</text>',
        *body,
        f'<text x="0" y="{height - 6}" font-family="Arial,sans-serif" font-size="10.5" fill="{MUTED}">{escape(source)}</text>',
        "</svg>"])


def _text(x, y, value, size=12, colour=MUTED, anchor="start", weight="400") -> str:
    return (f'<text x="{x:.0f}" y="{y:.0f}" font-family="Arial,sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{colour}" text-anchor="{anchor}">{escape(str(value))}</text>')


def membership_chart(item: dict, title: str) -> str:
    """Every member accounted for, year by year: for, abstaining, against,
    and the gap of those who did not vote. Shows who peeled away."""
    rows = [*(item.get("history") or [])[-4:], {"year": int(item["date"][:4]), **item["tally"]}]
    left, span, top, step = 52, 470, 72, 30
    scale = span / MEMBERS
    body = []
    for i, row in enumerate(rows):
        current = i == len(rows) - 1
        y, h, opacity = top + i * step, (20 if current else 15), (1 if current else 0.5)
        x = left
        body.append(_text(0, y + h - 4, row["year"], 13, INK if current else MUTED, weight="700" if current else "400"))
        for key, colour in (("yes", YES), ("abstain", ABSTAIN), ("no", NO)):
            w = row[key] * scale
            body.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 1.5, 0):.1f}" height="{h}" fill="{colour}" opacity="{opacity}"/>')
            x += w
        body.append(f'<rect x="{x:.1f}" y="{y}" width="{max(left + span - x, 0):.1f}" height="{h}" fill="none" stroke="{RULE}" stroke-width="1"/>')
        body.append(_text(left + span + 10, y + h - 4, f"{row['no']} against", 12.5,
                          NO if current else MUTED, weight="700" if current else "400"))
    y = top + len(rows) * step + 8
    last = rows[-1]
    body.append(_text(left, y + 6, f"{last['yes']} in favour", 12, YES, weight="700"))
    body.append(_text(left + last["yes"] * scale, y + 6, f"{last['abstain']} abstained", 12, MUTED))
    body.append(_text(left + span + 10, y + 6, "outline: did not vote", 11, MUTED))
    return _svg(640, y + 34, title, f"{_cap(short_name(item['title']))}: how all {MEMBERS} members voted, by year", body,
                "Sources: UN General Assembly resolutions register; UN Digital Library")


def support_chart(item: dict, title: str) -> str:
    """Votes in favour over the years against the half-membership line."""
    rows = [*(item.get("history") or [])[-7:], {"year": int(item["date"][:4]), **item["tally"]}]
    left, right, top, bottom = 46, 560, 70, 210
    low = min(min(r["yes"] for r in rows), HALF) - 12
    high = max(r["yes"] for r in rows) + 10

    def px(i): return left + (right - left) * (i / max(1, len(rows) - 1))
    def py(v): return bottom - (bottom - top) * ((v - low) / (high - low))
    body = [f'<line x1="{left}" y1="{py(HALF):.0f}" x2="{right}" y2="{py(HALF):.0f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 3"/>',
            _text(left, py(HALF) - 6, f"Half the membership ({HALF})", 11)]
    points = " ".join(f"{px(i):.0f},{py(r['yes']):.0f}" for i, r in enumerate(rows))
    body.append(f'<polyline points="{points}" fill="none" stroke="{YES}" stroke-width="2.5" stroke-linejoin="round"/>')
    for i, row in enumerate(rows):
        end = i in (0, len(rows) - 1)
        body.append(f'<circle cx="{px(i):.0f}" cy="{py(row["yes"]):.0f}" r="{5 if end else 3}" fill="{YES}" stroke="{PAPER}" stroke-width="2"/>')
        body.append(_text(px(i), bottom + 22, row["year"], 11.5, anchor="middle"))
        if end:
            body.append(_text(px(i), py(row["yes"]) - 12, row["yes"], 14, INK, anchor="middle", weight="700"))
    return _svg(640, 262, title, f"Votes in favour of {short_name(item['title'])}", body,
                "Sources: UN General Assembly resolutions register; UN Digital Library")


def veto_chart(by_member: dict, drafts: dict, title: str) -> str:
    """Vetoes cast per year, stacked by the member casting them."""
    years = sorted(int(y) for y in drafts)
    left, bottom, top, step = 30, 200, 70, 40
    peak = max([sum(by_member.get(str(y), by_member.get(y, {})).values()) for y in years] + [1])
    unit = (bottom - top) / peak
    body = [f'<line x1="{left}" y1="{bottom}" x2="{left + step * len(years)}" y2="{bottom}" stroke="{INK}" stroke-width="1"/>']
    order = ("Russia", "China", "United States", "United Kingdom", "France")
    for i, year in enumerate(years):
        cast = by_member.get(str(year), by_member.get(year, {}))
        x, y = left + i * step + 8, bottom
        for member in order:
            n = cast.get(member, 0)
            if n:
                y -= n * unit
                body.append(f'<rect x="{x}" y="{y:.1f}" width="24" height="{n * unit - 1.5:.1f}" fill="{MEMBER_COLOURS[member]}"/>')
        total = sum(cast.values())
        if total:
            body.append(_text(x + 12, y - 5, total, 11.5, INK, anchor="middle", weight="700"))
        body.append(_text(x + 12, bottom + 16, f"’{str(year)[2:]}", 11, anchor="middle"))
    lx = left
    for member in ("Russia", "China", "United States"):
        body.append(f'<rect x="{lx}" y="56" width="10" height="10" fill="{MEMBER_COLOURS[member]}"/>')
        body.append(_text(lx + 14, 65, member, 11.5))
        lx += 30 + 7 * len(member)
    return _svg(640, 244, title, "", body,
                "Source: Dag Hammarskjöld Library veto list. A draft vetoed by two members counts twice.")


def timeline_chart(today: date, events: list[dict], committee_until: date | None, title: str) -> str:
    """What is due, one row per vote against a shared line of time.
    ``events``: [{label, date}]."""
    events = sorted(events, key=lambda e: e["date"])[:7]
    start = today.toordinal()
    end = max([e["date"].toordinal() for e in events] + [start + 14]) + 3
    left, right, top, step = 250, 620, 84, 24

    def px(d): return left + (right - left) * ((d.toordinal() - start) / (end - start))
    bottom = top + step * len(events)
    body = []
    tick = today
    while tick.toordinal() <= end:
        x = px(tick)
        body.append(f'<line x1="{x:.0f}" y1="{top - 10}" x2="{x:.0f}" y2="{bottom}" stroke="{RULE}" stroke-width="1"/>')
        body.append(_text(x, top - 16, "Today" if tick == today else f"{tick.day} {tick:%b}", 11,
                          INK if tick == today else MUTED, anchor="middle", weight="700" if tick == today else "400"))
        tick = date.fromordinal(tick.toordinal() + 7)
    if committee_until and committee_until > today:
        body.append(f'<rect x="{left}" y="{bottom + 6}" width="{px(committee_until) - left:.0f}" height="6" fill="{ABSTAIN}"/>')
        body.append(_text(left - 10, bottom + 13, "Committees still voting", 11.5, MUTED, anchor="end"))
        bottom += 16
    for i, event in enumerate(events):
        y = top + i * step + 8
        x = px(event["date"])
        body.append(_text(left - 10, y + 4, event["label"], 12.5, INK, anchor="end"))
        body.append(f'<line x1="{left}" y1="{y}" x2="{x:.0f}" y2="{y}" stroke="{YES}" stroke-width="1.5" opacity="0.35"/>')
        body.append(f'<circle cx="{x:.0f}" cy="{y}" r="5" fill="{YES}"/>')
    return _svg(640, bottom + 30, title, "Typical dates of recurring plenary votes, from the last five years", body,
                "Source: UN Digital Library voting record")
