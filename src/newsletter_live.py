"""A concise, email-safe newsletter shared by HTML, Markdown and text exports."""

from datetime import datetime, timedelta
from html import escape, unescape
import re

from src.newsletter_story import (
    choose, comparison, day, headline_for, membership_chart, standfirst, support_chart,
    timeline_chart, veto_chart,
)
from src.un_bodies import COMMITTEE_REMIT


# Keys of current_affairs that never change an edition's content hash.
EDITION_CONTEXT_KEYS = ("checked_at", "sources", "window_start", "window_end", "headlines",
                        "bodies", "calendar")


def _plural(count, noun):
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _outcome(item, compare=True):
    tally = item["tally"]
    if not tally:
        return f"{day(item['date'])} · adopted without a vote"
    text = f"{day(item['date'])} · {tally['yes']} for, {tally['no']} against, {tally['abstain']} abstaining."
    return text + (" " + comparison(item) if compare and item.get("history") else "")


def _lead_chart_title(item):
    tally, last = item["tally"], item["history"][-1]
    if tally["no"] != last["no"]:
        return f"{tally['no']} against, where there were {last['no']}"
    return f"{tally['yes']} in favour, where there were {last['yes']}"


def _support_title(item):
    first, yes = item["history"][0], item["tally"]["yes"]
    verb = "down from" if yes < first["yes"] else "up from"
    return f"{yes} in favour, {verb} {first['yes']} in {first['year']}"


def sections(edition):
    live = edition.current_affairs
    decisions = live.get("decisions", [])
    blocks = []

    def add(kind, text, url=None):
        blocks.append((kind, str(text), url))

    add("kicker", f"{edition.dateline} · UN-Scrupulous")
    add("headline", edition.headline)
    add("intro", edition.lede)
    lead, second = choose(decisions)
    if lead and lead.get("history"):
        add("chart", membership_chart(lead, _lead_chart_title(lead)))

    recorded = [d for d in decisions if d["tally"]]
    quiet = [d for d in decisions if not d["tally"]]
    late = live.get("late_decisions", [])
    add("h2", "The votes")
    for item in recorded:
        add("link", item["title"], item.get("url") or item["source_url"])
        add("small", _outcome(item))
    if quiet:
        add("p", "Adopted without a vote: " + "; ".join(d["title"] for d in quiet[:8])
            + (f"; and {len(quiet) - 8} more." if len(quiet) > 8 else "."))
    for item in late:
        add("link", item["title"], item.get("url") or item["source_url"])
        add("small", _outcome(item, compare=False) + f" Published in the register only on {day(item['registered'])}.")
    if not decisions and not late:
        add("p", "The Assembly's register shows no resolutions adopted in the past 30 days.")
    if second and len(second.get("history") or []) >= 2:
        add("chart", support_chart(second, _support_title(second)))

    bodies = live.get("bodies") or {}
    vetoes = live.get("vetoes_this_year") or []
    if bodies or vetoes:
        year = edition.edition_date[:4]
        add("h2", "The Security Council")
        if vetoes:
            latest = vetoes[0]
            add("p", f"{_plural(len(vetoes), 'draft resolution')} {'has' if len(vetoes) == 1 else 'have'} been vetoed in "
                f"{year}. The latest was on {day(latest['date'])}, on {latest['agenda'][:1].lower() + latest['agenda'][1:]}, "
                f"cast by {' and '.join(latest['members'])}.")
        else:
            add("p", f"No draft resolution has been vetoed in {year}.")
        drafts = bodies.get("vetoed_drafts_by_year") or {}
        cast = {int(y): sum(m.values()) for y, m in (bodies.get("vetoes_by_member") or {}).items()}
        if drafts and cast:
            peak_year = max(cast, key=lambda y: (cast[y], y))
            this_year = cast.get(int(year), 0)
            add("chart", veto_chart(bodies.get("vetoes_by_member") or {}, drafts,
                                    f"{_plural(this_year, 'veto')} cast on {_plural(len(vetoes), 'draft')} so far in "
                                    f"{year}; the recent peak was {cast[peak_year]} in {peak_year}"
                                    .replace("vetos", "vetoes")))
        for item in bodies.get("council", [])[:5]:
            add("link", item["title"], item["url"])
            add("small", f"{day(item['date'])} · {'not adopted' if item['action'] == 'defeated' else item['action']}")

    committee = bodies.get("committee") or []
    if committee:
        add("h2", "In committee")
        totals = bodies.get("committee_totals") or {}
        counted = [f"the {name} ({COMMITTEE_REMIT[name]}) {_plural(t['texts'], 'text')}"
                   for name, t in totals.items() if t.get("texts")]
        add("p", "Committees vote first; the plenary votes again on the same texts in December, so these are a "
            "preview of next month." + (" Approved in the past 30 days: " + "; ".join(counted) + "." if counted else ""))
        for item in committee[:5]:
            add("link", item["title"], item["url"])
            add("small", f"{day(item['date'])} · {item['body']}")

    calendar = live.get("calendar") or {}
    upcoming = calendar.get("upcoming") or []
    if upcoming:
        add("h2", "What's coming")
        today = datetime.strptime(edition.edition_date, "%Y-%m-%d").date()
        events = [{"label": u["label"], "date": today + timedelta(days=u["days_until"])} for u in upcoming[:7]]
        until = datetime.strptime(calendar["committees_until"], "%Y-%m-%d").date() if calendar.get("committees_until") else None
        add("chart", timeline_chart(today, events, until, f"{_plural(len(events), 'recurring vote')} due in the next five weeks"))
        for u in upcoming[:6]:
            last = u["last"]
            add("p", f"{u['label']}, usually around {u['typical_date']}. Last time: {last['yes']} to {last['no']}, "
                f"with {last['abstain']} abstaining.")

    if live.get("headlines"):
        add("h2", "Also at the UN")
        for item in live["headlines"]:
            add("link", item["title"], item["url"])
            add("small", f"{item['source']} · {day(item['published_at'][:10])}")

    through = live.get("rollcall_data_through")
    findings = live.get("findings", [])
    if findings and through:
        stale = (datetime.strptime(edition.edition_date, "%Y-%m-%d") - datetime.strptime(through, "%Y-%m-%d")).days > 45
        add("h2", "Who votes with whom")
        if stale:
            add("p", f"No country-by-country votes have been published since {day(through, year=True)}, so this "
                f"picture is unchanged. {findings[0]['finding']}")
        else:
            for finding in findings:
                add("p", f"{finding['title']}. {finding['finding']}")

    add("h2", "How to read this")
    pending = live.get("pending_rollcalls")
    add("small", "Vote totals come from the General Assembly's official register and are compared with the same "
        "resolution's earlier votes in the UN Digital Library record. "
        + (f"Country-by-country results for {_plural(pending, 'recent vote')} are not yet published, so no country's "
           "position is inferred from a total. " if pending else "")
        + "Security Council and committee items come from the Dag Hammarskjöld Library's veto list and UN meetings "
        "coverage, which report outcomes, not each member's vote. Agreement between countries counts Yes and No "
        "votes only. Typical dates are medians of the last five years. This edition is compiled automatically.")
    registers = list(dict.fromkeys(d["source_url"] for d in decisions + late))
    for url in registers[:2]:
        add("small", "The Assembly's official register of resolutions", url)
    add("small", f"Edition {edition.content_hash[:12]} · decisions to {day(live['decision_data_through']) if live.get('decision_data_through') else 'n/a'}"
        f" · country votes to {day(through, year=True) if through else 'n/a'}")
    return blocks


def render_live(edition, fmt, footer=None):
    """``footer``: optional (text, url) pairs appended to the end, e.g. an unsubscribe link."""
    blocks = sections(edition)
    for text, url in footer or []:
        blocks.append(("small", text, url))
    if fmt != "html":
        lines = []
        for kind, value, url in blocks:
            if kind == "chart":
                label = re.search(r'aria-label="([^"]*)"', value)
                lines.append(f"[Chart: {unescape(label[1]) if label else 'see the web edition'}]")
            elif fmt == "md":
                prefix = {"headline": "# ", "h2": "## ", "h3": "### "}.get(kind, "")
                # Source titles are data, not Markdown markup.
                value = value.replace("[", "\\[").replace("]", "\\]").replace("<", "&lt;")
                lines.append(f"[{value}]({url})" if url else prefix + value)
            else:
                lines.append(value.upper() if kind == "h2" else value + (f"\n{url}" if url else ""))
        return "\n\n".join(lines) + "\n"
    styles = {
        "kicker": "font:12px Arial,sans-serif;letter-spacing:1px;text-transform:uppercase;color:#5d6f73;border-bottom:3px solid #102c35;padding-bottom:14px",
        "headline": "font:700 34px/1.12 Georgia,serif;margin:24px 0 14px;color:#102c35",
        "intro": "font:19px/1.5 Georgia,serif;color:#243d44;margin:0 0 22px",
        "p": "font:16px/1.6 Georgia,serif;color:#243d44;margin:10px 0",
        "h2": "font:700 13px Arial,sans-serif;letter-spacing:1.2px;text-transform:uppercase;border-top:1px solid #102c35;padding-top:12px;margin:34px 0 8px;color:#102c35",
        "h3": "font:700 18px/1.4 Georgia,serif;margin:20px 0 6px;color:#102c35",
        "link": "font:700 17px/1.35 Georgia,serif;margin:16px 0 2px;color:#102c35",
        "small": "font:13px/1.55 Arial,sans-serif;color:#5d6f73;margin:2px 0 12px",
    }
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           '<meta name="robots" content="noindex,nofollow">',
           # Sizing lives here, not on the <svg>: the email build turns each chart
           # into a PNG, and the converter cannot read a styled height.
           '<style>svg{max-width:100%;height:auto}img{max-width:100%;height:auto}</style>',
           f'<title>{escape(edition.email_subject)}</title></head>',
           '<body style="margin:0;background:#f3f1eb;padding:20px 10px">',
           '<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr><td align="center">',
           '<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width:688px;background:#fffdf8">',
           '<tr><td style="padding:28px 24px">']
    for kind, value, url in blocks:
        if kind == "chart":
            out.append(f'<div style="margin:22px 0 26px">{value}</div>')
            continue
        tag = {"headline": "h1", "h2": "h2", "h3": "h3"}.get(kind, "p")
        content = escape(value)
        if url:
            content = f'<a href="{escape(url, quote=True)}" style="color:inherit;text-decoration:underline;text-decoration-color:#9fb3b0">{content}</a>'
        out.append(f'<{tag} style="{styles[kind]}">{content}</{tag}>')
    out.append('</td></tr></table></td></tr></table></body></html>')
    return "\n".join(out)


def build_current_edition(live, edition_date=None):
    """Compose once from validated evidence, without rerunning legacy analytics."""
    from datetime import timezone
    import hashlib
    import json
    from src.newsletter import NewsletterEdition, LeadStory, TOCItem

    date = datetime.strptime(edition_date, "%Y-%m-%d") if edition_date else datetime.now(timezone.utc)
    # A new edition means new decisions, tally corrections, a veto or changed
    # research. Headlines, committee coverage and the calendar are context,
    # like the legacy calendar and big-picture sections: none of them alone
    # may mint an edition, a release and a tag.
    editorial = {k: v for k, v in live.items() if k not in EDITION_CONTEXT_KEYS}
    digest = hashlib.sha256(json.dumps({"editorial_version": 5, **editorial}, sort_keys=True).encode()).hexdigest()
    lead, second = choose(live.get("decisions", []))
    vetoes = [v for v in live.get("vetoes_this_year") or [] if v["date"] >= live.get("window_start", "")]
    if lead:
        headline, lede = headline_for(lead), standfirst(lead, second)
    elif vetoes:
        headline = f"{' and '.join(vetoes[0]['members'])} veto{'es' if len(vetoes[0]['members']) == 1 else ''} a Security Council draft"
        lede = (f"On {day(vetoes[0]['date'])} a draft resolution on "
                f"{vetoes[0]['agenda'][:1].lower() + vetoes[0]['agenda'][1:]} was vetoed in the Security Council. "
                "The General Assembly recorded no votes in the past 30 days.")
    else:
        headline = "A quiet month in the Assembly"
        lede = "No recorded votes in the General Assembly in the past 30 days. What the committees and the Security Council did is below."
    period = live.get("research_period") or "Country-level research not available"
    return NewsletterEdition(
        publication="UN-Scrupulous", edition_number=date.isocalendar().week,
        edition_date=date.date().isoformat(), edition_slug=f"{date.date()}-{digest[:12]}",
        email_subject=headline[:110], content_hash=digest,
        country_focus=None, dateline=f"{date.day} {date:%B %Y}",
        byline="Automated political research", period_label=period,
        recent_year=int((live.get("rollcall_data_through") or str(date.year))[:4]), baseline_window={},
        headline=headline, subhead="", lede=lede, nut_graf="",
        in_this_issue=[TOCItem(number=i + 1, anchor=anchor, title=title) for i, (anchor, title) in enumerate([
            ("world", "The world at the UN"), ("decisions", "Decisions that matter"),
            ("institution", "The UN as an institution"), ("research", "What the voting data reveals")])],
        by_the_numbers=[], lead_story=LeadStory(headline=headline, body=lede, supporting_drifts=[]),
        lead_story_why_it_matters="", top_movers=[], top_movers_why_it_matters="",
        coalition_watch=[], coalition_why_it_matters="", quiet_convergences=[],
        resolution_spotlight=None, next_to_watch=[], bloc_state=[],
        freshness={"latest_vote_date": live.get("rollcall_data_through"), "latest_decision_date": live.get("decision_data_through")},
        methodology=[], sources=[], current_affairs=live,
    )
