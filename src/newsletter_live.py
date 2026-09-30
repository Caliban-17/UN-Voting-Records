"""A concise, email-safe newsletter shared by HTML, Markdown and text exports."""

from html import escape


# Keys of current_affairs that never change an edition's content hash.
EDITION_CONTEXT_KEYS = ("checked_at", "sources", "window_start", "window_end", "headlines")


def _plural(count, noun):
    return f"{count} {noun}" + ("" if count == 1 else "s")


def sections(edition):
    live = edition.current_affairs
    decisions = live.get("decisions", [])
    blocks = []

    def add(kind, text, url=None):
        blocks.append((kind, str(text), url))

    add("kicker", f"{edition.dateline} · Automated political research")
    add("title", edition.publication)
    add("headline", edition.headline)
    add("intro", edition.lede)
    add("p", edition.nut_graf)
    add("coverage", f"Official GA decisions through {live.get('decision_data_through') or 'unavailable'} · "
        f"Country votes through {live.get('rollcall_data_through') or 'unavailable'}.")

    add("h2", "The world at the UN")
    add("p", "Latest official reporting. These are attributed headlines, not independently verified research findings.")
    for item in live.get("headlines", []):
        add("link", item["title"], item["url"])
        add("small", f"{item['source']} · {item['published_at'][:10]}")
    if not live.get("headlines"):
        add("p", "No recent official headlines were available at composition time.")

    add("h2", "Decisions that matter")
    add("small", f"Published regular-session GA resolutions · {live['window_start']} to {live['window_end']}")
    for item in decisions:
        # Title opens the resolution itself; the metadata line cites the register.
        add("link", item["title"], item.get("url") or item["source_url"])
        tally = item["tally"]
        outcome = (f"{tally['yes']} for · {tally['no']} against · {tally['abstain']} abstaining"
                   if tally else "Adopted without a vote")
        add("small", f"{item['date']} · {item['symbol']} · {outcome}", item["source_url"])
    if not decisions:
        add("p", "No resolutions in this date window were available in the monitored registers. This is not evidence that the UN was inactive.")

    add("h2", "The UN as an institution")
    if decisions:
        total = len(decisions)
        without = live["without_vote_count"]
        add("h3", f"{without} of {total} resolutions adopted without a vote")
        add("p", f"In these registers, {without / total:.0%} of resolutions in the 30-day window were adopted without a recorded vote; "
            f"{live['recorded_count']} had recorded tallies. A roll-call-only dataset therefore omits part of the Assembly's adopted agenda.")
        add("p", "Adoption without a vote does not establish unanimous enthusiasm: members may still explain reservations. "
            "This count covers published regular-session GA resolutions, not failed drafts, emergency sessions or other UN bodies.")
    else:
        add("p", "The current window is too sparse to calculate an institutional decision-making indicator.")

    add("h2", "What the voting data reveals")
    add("small", live.get("research_period") or "Research period unavailable")
    for finding in live.get("findings", []):
        add("h3", finding["title"])
        add("p", finding["finding"])
        add("p", finding["interpretation"])
        evidence = finding.get("evidence", {})
        if "recent_resolutions" in evidence:
            add("small", f"Sample: {evidence['recent_resolutions']} recent resolutions; "
                f"{evidence.get('baseline_resolutions', 'unspecified')} baseline resolutions.")
        add("small", finding["caveat"])
    if not live.get("findings"):
        add("p", "No sufficiently supported research findings are available.")

    add("h2", "What to watch next")
    missing = [r for r in decisions if r["tally"] and r["date"] > (live.get("rollcall_data_through") or "")]
    if missing:
        add("p", f"Complete country roll calls for {len(missing)} newer recorded resolutions are not yet in the research dataset. "
            "Their official totals are reported above, but no country positions or alignment shifts are inferred from those totals. "
            "The automatic refresh will incorporate complete roll calls when the upstream dataset supplies them.")
    add("p", "Watch whether subsequent votes sustain the measured alignment and division patterns. A change in the agenda, attendance or resolution wording could alter them.")
    add("h2", "Sources and coverage")
    add("p", "Decisions: UN General Assembly's public e-deleGATE registers. Country votes: UN Digital Library and DGACM extracts. "
        "Research uses seasonally matched dates when analysing the current year and shows its own coverage period. "
        "The newsletter is composed and formatted automatically from published source records and reproducible calculations.")
    seen = set()
    for source in live.get("sources", []):
        url = source.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        name = source.get("name") or f"GA session {source.get('session')} register"
        status = {"ok": "available", "quiet": "no recent items", "error": "fetch failed; retained records may be older"}.get(source.get("status"), "unknown")
        add("link", f"{name} — {status}", url)
    add("small", f"Source checks: {live.get('checked_at') or 'not available'} · Edition {edition.content_hash[:12]}")
    return blocks


def render_live(edition, fmt):
    blocks = sections(edition)
    if fmt != "html":
        lines = []
        for kind, value, url in blocks:
            if fmt == "md":
                prefix = {"title": "# ", "headline": "## ", "h2": "## ", "h3": "### "}.get(kind, "")
                # Source titles are data, not Markdown markup.
                value = value.replace("[", "\\[").replace("]", "\\]").replace("<", "&lt;")
                lines.append(f"[{value}]({url})" if url else prefix + value)
            else:
                lines.append(value + (f"\n{url}" if url else ""))
        return "\n\n".join(lines) + "\n"
    styles = {
        "kicker": "font:12px Arial,sans-serif;letter-spacing:1px;text-transform:uppercase;color:#52696d",
        "title": "font:700 22px Georgia,serif;color:#126271;border-bottom:3px solid #126271;padding-bottom:18px",
        "headline": "font:700 32px/1.15 Georgia,serif;margin:26px 0 16px;color:#102c35",
        "intro": "font:20px/1.5 Georgia,serif;color:#102c35",
        "p": "font:16px/1.65 Georgia,serif;color:#243d44;margin:12px 0",
        "coverage": "font:13px/1.6 Arial,sans-serif;background:#edf4f3;padding:14px;border-left:3px solid #126271",
        "h2": "font:700 21px Georgia,serif;border-top:1px solid #cbd9d7;padding-top:25px;margin-top:30px;color:#126271",
        "h3": "font:700 18px/1.4 Georgia,serif;margin:20px 0 6px;color:#102c35",
        "link": "font:700 16px/1.5 Georgia,serif;margin:16px 0 3px;color:#126271",
        "small": "font:12px/1.6 Arial,sans-serif;color:#52696d;margin:3px 0 14px",
    }
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           f'<title>{escape(edition.email_subject)}</title></head>',
           '<body style="margin:0;background:#f3f1eb;padding:20px 10px">',
           '<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr><td align="center">',
           '<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width:680px;background:#fffdf8">',
           '<tr><td style="padding:28px 24px">']
    for kind, value, url in blocks:
        tag = {"headline": "h1", "h2": "h2", "h3": "h3"}.get(kind, "p")
        content = escape(value)
        if url:
            content = f'<a href="{escape(url, quote=True)}" style="color:inherit;text-decoration:underline">{content}</a>'
        out.append(f'<{tag} style="{styles[kind]}">{content}</{tag}>')
    out.append('</td></tr></table></td></tr></table></body></html>')
    return "\n".join(out)


def build_current_edition(live, edition_date=None):
    """Compose once from validated evidence, without rerunning legacy analytics."""
    from datetime import datetime, timezone
    import hashlib
    import json
    from src.newsletter import NewsletterEdition, LeadStory, TOCItem

    date = datetime.strptime(edition_date, "%Y-%m-%d") if edition_date else datetime.now(timezone.utc)
    # A new edition means new decisions, tally corrections or changed research.
    # Headlines are context, like the calendar and big-picture sections: a new
    # wire story alone must not mint an edition, a release and a tag.
    editorial = {k: v for k, v in live.items() if k not in EDITION_CONTEXT_KEYS}
    digest = hashlib.sha256(json.dumps({"editorial_version": 4, **editorial}, sort_keys=True).encode()).hexdigest()
    decisions = live.get("decisions", [])
    headline = "The UN dispatch"
    lede = "Current official reporting and the latest available voting research."
    if decisions:
        latest = decisions[0]
        headline = (f"{_plural(len(decisions), 'decision')}, "
                    f"{_plural(live['recorded_count'], 'recorded vote')}")
        tally = latest["tally"]
        result = (f"{tally['yes']} in favour, {tally['no']} against and {tally['abstain']} abstaining"
                  if tally else "without a recorded vote")
        lede = (f"Latest in the register: {latest['title']}. On {latest['date']}, "
                f"the General Assembly adopted {latest['symbol']} — {result}.")
    period = live.get("research_period") or "Country-level research not available"
    return NewsletterEdition(
        publication="UN-Scrupulous", edition_number=date.isocalendar().week,
        edition_date=date.date().isoformat(), edition_slug=f"{date.date()}-{digest[:12]}",
        email_subject=("UN-Scrupulous: " + headline)[:78], content_hash=digest,
        country_focus=None, dateline=date.strftime("%B %-d, %Y").upper(),
        byline="Automated political research", period_label=period,
        recent_year=int((live.get("rollcall_data_through") or str(date.year))[:4]), baseline_window={},
        headline=headline, subhead="Decisions, world affairs and what the voting record can tell us.",
        lede=lede, nut_graf="Official decisions establish what the Assembly did. The research below measures patterns in the available country votes; it does not infer diplomatic intent.",
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
