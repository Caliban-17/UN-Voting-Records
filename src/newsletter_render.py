"""
Newsletter renderers — Markdown, HTML (email-safe), and plain text.

Outputs are deliberately email-safe:
  * inline styles on every element (no external CSS; many email clients strip
    <style> blocks or rewrite class names)
  * table-based layout (Outlook 2016+ ignores flexbox / grid)
  * a single inline SVG chart (no remote image fetches; embeds cleanly)
  * a print stylesheet at the bottom so the same HTML prints to a clean
    one-or-two-page PDF

This is what makes the same edition exportable to a Substack, an academic
mailing list, an internal foreign-ministry brief, or a print archive.
"""

from __future__ import annotations

import html as _html
from typing import Iterable

from src.newsletter import (
    CoalitionMove,
    CoalitionSnapshot,
    NewsletterEdition,
    QuietConvergence,
    ResolutionSpotlight,
    StatHighlight,
    WatchItem,
)
from src.newsletter_chart import (
    alliance_arrow_svg,
    dumbbell_chart_svg,
    p5_fingerprint_svg,
    slope_chart_svg,
    top_drifts_svg,
    vote_tally_svg,
)
from src.newsletter_voice import (
    SECTION_TITLES,
    humanize_topic_full,
    strange_bedfellows_intro,
    strange_bedfellows_one_liner,
)


# ── shared helpers ──────────────────────────────────────────────────────────


def _signed_pts(delta: float) -> str:
    return f"{'+' if delta >= 0 else ''}{delta * 100:.0f} pts"


def _esc(s: str) -> str:
    return _html.escape(str(s or ""))


# ── Markdown ────────────────────────────────────────────────────────────────


def render_markdown(edition: NewsletterEdition) -> str:
    lines: list[str] = []
    lines.append(
        f"# {edition.publication} — Edition №{edition.edition_number}"
    )
    lines.append(
        f"_{edition.dateline} · {edition.byline} · "
        f"{edition.period_label}_"
    )
    lines.append("")
    lines.append(f"## {edition.headline}")
    lines.append(f"**{edition.subhead}**")
    lines.append("")
    lines.append(f"> {edition.lede}")
    lines.append("")
    lines.append(edition.nut_graf)
    lines.append("")

    # In this issue
    if edition.in_this_issue:
        lines.append("### In this issue")
        for item in edition.in_this_issue:
            lines.append(f"{item.number}. {item.title}")
        lines.append("")

    # By the numbers
    if edition.by_the_numbers:
        lines.append(f"## {SECTION_TITLES['by_the_numbers']}")
        lines.append("")
        for stat in edition.by_the_numbers:
            lines.append(f"- **{stat.value}** — {stat.label}. _{stat.context}_")
        lines.append("")

    # Lead story
    lines.append(f"## {SECTION_TITLES['shift']}")
    lines.append("")
    lines.append(edition.lead_story.body)
    lines.append("")
    lines.append(f"_Why it matters: {edition.lead_story_why_it_matters}_")
    if edition.lead_story.supporting_drifts:
        lines.append("")
        lines.append("**Supporting moves:**")
        for d in edition.lead_story.supporting_drifts:
            topics = [
                humanize_topic_full(t["topic"])
                for t in d.get("driving_topics", [])
            ][:2]
            topics = [t for t in topics if t]
            topic_str = f" — on {', '.join(topics)}" if topics else ""
            lines.append(
                f"- {d['country_a']} ↔ {d['country_b']}: "
                f"{d['baseline_agreement'] * 100:.0f}% → "
                f"{d['recent_agreement'] * 100:.0f}% "
                f"({_signed_pts(d['delta'])}){topic_str}"
            )
    lines.append("")

    # Top movers
    if edition.top_movers:
        lines.append(f"## {SECTION_TITLES['movers']}")
        lines.append("")
        lines.append(f"_{edition.top_movers_why_it_matters}_")
        lines.append("")
        for mover in edition.top_movers:
            lines.append(f"### {mover.name} ({mover.country})")
            lines.append(f"_{mover.headline}_")
            lines.append("")
            lines.append(mover.detail)
            lines.append("")

    # Coalition watch
    if edition.coalition_watch:
        lines.append(f"## {SECTION_TITLES['coalition']}")
        lines.append("")
        lines.append(f"_{edition.coalition_why_it_matters}_")
        lines.append("")
        for snap in edition.coalition_watch:
            lines.append(f"### \"{snap.topic.title()}\"")
            lines.append("")
            lines.append(snap.headline_stat)
            lines.append("")
            if snap.movers:
                lines.append("| Country | Was | Now | Lean (base → recent) |")
                lines.append("|---|---|---|---|")
                for m in snap.movers:
                    lines.append(
                        f"| {m.name} ({m.country}) | {m.from_tier} | "
                        f"**{m.to_tier}** | "
                        f"{m.mean_baseline:+.2f} → {m.mean_recent:+.2f} |"
                    )
                lines.append("")
            else:
                lines.append("_No country changed tier on this topic this period._")
                lines.append("")
            if snap.sample_titles:
                lines.append("Sample resolutions matched: " + "; ".join(
                    [f"\"{t}\"" for t in snap.sample_titles[:2]]
                ))
                lines.append("")

    # Resolution spotlight
    if edition.resolution_spotlight:
        s = edition.resolution_spotlight
        lines.append(f"## {SECTION_TITLES['spotlight']}")
        lines.append("")
        lines.append(f"**\"{s.title}\"** ({s.date})")
        if s.topic:
            lines.append(f"_Topic: {s.topic}_")
        lines.append("")
        lines.append(
            f"The most contested vote of the period — {s.margin_summary}. "
            f"Yes: {s.yes}, No: {s.no}, Abstain: {s.abstain}."
        )
        lines.append("")
        if s.breakaways:
            names = ", ".join(
                f"{b['name']} ({b['country']})" for b in s.breakaways[:12]
            )
            lines.append(f"**Voting against the majority:** {names}.")
            lines.append("")

    # Quiet convergences — the "Strange Bedfellows" gossip section.
    if edition.quiet_convergences:
        lines.append(f"## {SECTION_TITLES['convergences']}")
        lines.append("")
        # Use the top convergence as the kicker subject.
        top_c = edition.quiet_convergences[0]
        kicker_drift = {
            "country_a": top_c.country_a, "country_b": top_c.country_b,
            "baseline_agreement": top_c.baseline, "recent_agreement": top_c.recent,
            "delta": top_c.delta, "abs_delta": abs(top_c.delta),
            "driving_topics": [{"topic": t} for t in (top_c.top_topics or [])],
        }
        lines.append(strange_bedfellows_intro(kicker_drift))
        lines.append("")
        for c in edition.quiet_convergences:
            topic_str = f" on {', '.join(c.top_topics)}" if c.top_topics else ""
            lines.append(
                f"- **{c.name_a} ↔ {c.name_b}**: {c.baseline * 100:.0f}% → "
                f"{c.recent * 100:.0f}% ({_signed_pts(c.delta)}){topic_str}"
            )
        lines.append("")

    # Next to watch
    if edition.next_to_watch:
        lines.append(f"## {SECTION_TITLES['next']}")
        lines.append("")
        for w in edition.next_to_watch:
            lines.append(f"- **{w.topic}** — {w.rationale}")
        lines.append("")

    # Methodology + sources
    lines.append("---")
    lines.append("")
    lines.append("### Methodology")
    for line in edition.methodology:
        lines.append(f"- {line}")
    lines.append("")
    lines.append("### Sources")
    for s in edition.sources:
        lines.append(f"- {s}")
    lines.append("")
    fr = edition.freshness
    lines.append(
        f"_Dataset: {fr['records']:,} records, {fr['min_year']}–{fr['max_year']}. "
        f"Latest vote in source: {fr['latest_vote_date']} "
        f"({fr['days_since_latest_vote']} days ago)._"
    )
    return "\n".join(lines)


# ── Plain text (for email multipart/alternative) ────────────────────────────


def _wrap_para(text: str, width: int = 78) -> str:
    """Word-wrap a single paragraph to ``width`` columns."""
    import textwrap
    return textwrap.fill(text, width=width, break_long_words=False, replace_whitespace=False)


def render_text(edition: NewsletterEdition) -> str:
    """Plain-text version of the edition. Designed for the text/plain MIME
    fallback in multipart email — and also for piping into a terminal."""
    lines: list[str] = []

    def hr(char: str = "─", n: int = 60) -> None:
        lines.append(char * n)

    def section(title: str) -> None:
        lines.append("")
        hr("─", 60)
        lines.append(title.upper())
        hr("─", 60)
        lines.append("")

    # Masthead
    lines.append(f"{edition.publication.upper()}")
    lines.append(f"Edition №{edition.edition_number}  ·  {edition.dateline}")
    lines.append(f"{edition.byline}  ·  {edition.period_label}")
    hr("═", 60)
    lines.append("")
    lines.append(edition.headline.upper())
    lines.append("")
    lines.append(_wrap_para(edition.subhead))
    lines.append("")
    lines.append(_wrap_para(edition.lede))
    lines.append("")
    lines.append(_wrap_para(edition.nut_graf))

    # In this issue
    if edition.in_this_issue:
        section("In this issue".title())
        for item in edition.in_this_issue:
            lines.append(f"  {item.number:>2}. {item.title}")

    # By the numbers
    if edition.by_the_numbers:
        section(SECTION_TITLES["by_the_numbers"])
        for stat in edition.by_the_numbers:
            lines.append(f"  • {stat.value:>10}   {stat.label}")
            lines.append(f"            {' ':>10}   ({stat.context})")

    # Lead story
    section(f"{SECTION_TITLES['shift']}: {edition.lead_story.headline}")
    lines.append(_wrap_para(edition.lead_story.body))
    lines.append("")
    lines.append(f"Why it matters: {_wrap_para(edition.lead_story_why_it_matters)}")
    if edition.lead_story.supporting_drifts:
        lines.append("")
        lines.append("Supporting moves:")
        for d in edition.lead_story.supporting_drifts:
            topics = [
                humanize_topic_full(t["topic"])
                for t in d.get("driving_topics", [])
            ][:2]
            topics = [t for t in topics if t]
            topic_str = f" — on {', '.join(topics)}" if topics else ""
            lines.append(
                f"  • {d['country_a']} <-> {d['country_b']}: "
                f"{d['baseline_agreement'] * 100:.0f}% -> "
                f"{d['recent_agreement'] * 100:.0f}% "
                f"({_signed_pts(d['delta'])}){topic_str}"
            )

    # Top movers
    if edition.top_movers:
        section(SECTION_TITLES["movers"])
        lines.append(_wrap_para(edition.top_movers_why_it_matters))
        lines.append("")
        for mover in edition.top_movers:
            lines.append(f"  {mover.name} ({mover.country})")
            lines.append(f"    {mover.headline}")
            lines.append(f"    {_wrap_para(mover.detail, width=74).replace(chr(10), chr(10) + '    ')}")
            lines.append("")

    # Coalition watch
    if edition.coalition_watch:
        section(SECTION_TITLES["coalition"])
        lines.append(_wrap_para(edition.coalition_why_it_matters))
        lines.append("")
        for snap in edition.coalition_watch:
            lines.append(f'  "{snap.topic.title()}"')
            lines.append("  " + _wrap_para(snap.headline_stat, width=74).replace("\n", "\n  "))
            if snap.movers:
                lines.append("")
                lines.append("    Tier-jumpers:")
                for m in snap.movers:
                    lines.append(
                        f"      {m.name} ({m.country}): "
                        f"{m.from_tier} -> {m.to_tier} "
                        f"({m.mean_baseline:+.2f} -> {m.mean_recent:+.2f})"
                    )
            lines.append("")

    # Resolution spotlight
    if edition.resolution_spotlight:
        s = edition.resolution_spotlight
        section(SECTION_TITLES["spotlight"])
        lines.append(f'  "{s.title}" ({s.date})')
        if s.topic:
            lines.append(f"  Topic: {s.topic}")
        lines.append("")
        lines.append(_wrap_para(
            f"The most contested vote of the period — {s.margin_summary}. "
            f"Yes: {s.yes}, No: {s.no}, Abstain: {s.abstain}."
        ))
        if s.breakaways:
            names = ", ".join(f"{b['name']} ({b['country']})" for b in s.breakaways[:12])
            lines.append("")
            lines.append("Voting against the majority:")
            lines.append("  " + _wrap_para(names, width=74).replace("\n", "\n  "))

    # Quiet convergences
    if edition.quiet_convergences:
        section(SECTION_TITLES["convergences"])
        lines.append("Buried under the headline divergences — pairs moving closer:")
        lines.append("")
        for c in edition.quiet_convergences:
            topic_str = f" on {', '.join(c.top_topics)}" if c.top_topics else ""
            lines.append(
                f"  • {c.name_a} <-> {c.name_b}: {c.baseline * 100:.0f}% -> "
                f"{c.recent * 100:.0f}% ({_signed_pts(c.delta)}){topic_str}"
            )

    # Next to watch
    if edition.next_to_watch:
        section(SECTION_TITLES["next"])
        for w in edition.next_to_watch:
            lines.append(f"  • {w.topic}")
            lines.append(f"    {_wrap_para(w.rationale, width=74).replace(chr(10), chr(10) + '    ')}")
            lines.append("")

    # Footer
    section("Methodology & Sources")
    for line in edition.methodology:
        lines.append(f"  • {_wrap_para(line, width=74).replace(chr(10), chr(10) + '    ')}")
    lines.append("")
    lines.append("Sources:")
    for s_ in edition.sources:
        lines.append(f"  • {s_}")
    lines.append("")
    fr = edition.freshness
    lines.append(_wrap_para(
        f"Dataset: {fr['records']:,} records, {fr['min_year']}–{fr['max_year']}. "
        f"Latest vote in source: {fr['latest_vote_date']} "
        f"({fr['days_since_latest_vote']} days ago)."
    ))
    return "\n".join(lines)


# ── HTML (email-safe, table-based, inline styles) ───────────────────────────

# Inline-style fragments — kept short so render_html stays readable.
_S = {
    "wrap": (
        "margin:0;padding:0;background:#f5f3ee;font-family:Georgia,"
        "'Palatino Linotype',serif;color:#0b2238;line-height:1.55;"
    ),
    "container": (
        "max-width:680px;margin:0 auto;padding:32px 28px;background:#f8f6f1;"
    ),
    "kicker": (
        "font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;"
        "text-transform:uppercase;letter-spacing:0.16em;font-size:11px;"
        "color:#0e6f82;font-weight:700;margin-bottom:4px;"
    ),
    "masthead_meta": (
        "color:#456783;font-size:13px;font-style:italic;margin:0 0 18px;"
    ),
    "h1": (
        "font-size:30px;line-height:1.18;margin:4px 0 6px;letter-spacing:0.005em;"
        "color:#0b2238;font-weight:700;"
    ),
    "subhead": (
        "font-size:17px;line-height:1.4;color:#23425f;font-weight:600;"
        "margin:6px 0 16px;"
    ),
    "lede": (
        "font-size:17px;line-height:1.55;color:#23425f;"
        "border-left:3px solid #d46042;padding:6px 0 6px 14px;margin:18px 0;"
        "font-style:italic;"
    ),
    "nut": (
        "font-size:15.5px;line-height:1.55;color:#0b2238;margin:14px 0;"
    ),
    "toc": (
        "border:1px solid #d7d3c9;border-radius:8px;background:#fff;"
        "padding:14px 18px;margin:18px 0;"
    ),
    "toc_label": (
        "font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;"
        "text-transform:uppercase;letter-spacing:0.14em;font-size:11px;"
        "color:#456783;font-weight:700;margin-bottom:8px;"
    ),
    "toc_item": (
        "padding:3px 0;font-size:14px;color:#0b2238;"
    ),
    "h2": (
        "font-size:21px;margin:34px 0 8px;padding:0 0 6px;border-bottom:2px solid #d7d3c9;"
        "color:#0b2238;font-weight:700;"
    ),
    "h3": (
        "font-size:16px;margin:22px 0 4px;color:#0e6f82;font-weight:700;"
    ),
    "why_box": (
        "background:#fff8ea;border-left:3px solid #d38b2a;padding:8px 12px;"
        "margin:10px 0 14px;font-size:14px;color:#23425f;font-style:italic;"
    ),
    "lead_card": (
        "background:#fff;border:1px solid #d7d3c9;border-left:4px solid #d46042;"
        "border-radius:10px;padding:14px 18px;margin:12px 0;"
    ),
    "mover_card": (
        "background:#fff;border:1px solid #d7d3c9;border-left:4px solid #1f8ea5;"
        "border-radius:10px;padding:12px 16px;margin:8px 0;"
    ),
    "convergence_card": (
        "background:#f3f9f5;border:1px solid #c9e6d2;border-left:4px solid #2a9d57;"
        "border-radius:10px;padding:10px 14px;margin:8px 0;"
    ),
    "spotlight_card": (
        "background:#fff;border:1px solid #d7d3c9;border-left:4px solid #c64141;"
        "border-radius:10px;padding:14px 18px;margin:10px 0;"
    ),
    "stat_cell": (
        "background:#fff;border:1px solid #d7d3c9;border-left:4px solid #1f8ea5;"
        "border-radius:8px;padding:10px 12px;vertical-align:top;"
    ),
    "stat_value": (
        "font-size:22px;font-weight:700;color:#0b2238;"
        "font-family:'Palatino Linotype','Book Antiqua',serif;"
    ),
    "stat_label": (
        "font-size:12.5px;color:#23425f;margin-top:2px;"
    ),
    "stat_context": (
        "font-size:11.5px;color:#456783;margin-top:6px;font-style:italic;"
    ),
    "table_outer": (
        "width:100%;border-collapse:collapse;margin:8px 0 16px;font-size:14px;"
    ),
    "th": (
        "padding:6px 8px;border-bottom:2px solid #ece8e0;text-align:left;"
        "color:#456783;font-weight:700;font-size:11px;text-transform:uppercase;"
        "letter-spacing:0.06em;font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;"
    ),
    "td": (
        "padding:7px 8px;border-bottom:1px solid #ece8e0;color:#0b2238;"
    ),
    "footer": (
        "border-top:1px solid #d7d3c9;margin-top:36px;padding-top:16px;"
        "color:#456783;font-size:12.5px;line-height:1.5;"
    ),
    "footer_h4": (
        "font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;"
        "text-transform:uppercase;letter-spacing:0.12em;font-size:11px;"
        "color:#0b2238;font-weight:700;margin:14px 0 6px;"
    ),
    "chart_caption": (
        "font-size:12px;color:#456783;text-align:center;font-style:italic;"
        "margin:4px 0 18px;"
    ),
}


_PRINT_CSS = (
    "<style>@media print{"
    "body{background:#fff !important;}"
    ".no-print{display:none !important;}"
    "a[href]:after{content:'';}}"
    "</style>"
)


def _tier_pill(tier: str) -> str:
    tl = tier.lower()
    if "supporter" in tl:
        bg, fg = "#e7f5ec", "#18733e"
    elif "opposed" in tl:
        bg, fg = "#fce4e4", "#8a2222"
    else:
        bg, fg = "#fdf1d6", "#7a5a10"
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
        f'font-size:11px;font-weight:600;background:{bg};color:{fg};">{_esc(tier)}</span>'
    )


def _stat_row_html(stats: Iterable[StatHighlight]) -> str:
    # Use a table for layout — flexbox/grid don't work in Outlook 2016+.
    cells = []
    for stat in stats:
        cells.append(
            f'<td style="{_S["stat_cell"]}width:33%;">'
            f'<div style="{_S["stat_value"]}">{_esc(stat.value)}</div>'
            f'<div style="{_S["stat_label"]}">{_esc(stat.label)}</div>'
            f'<div style="{_S["stat_context"]}">{_esc(stat.context)}</div>'
            f'</td>'
        )
    # 3 per row.
    rows: list[str] = []
    for i in range(0, len(cells), 3):
        chunk = cells[i:i + 3]
        while len(chunk) < 3:
            chunk.append('<td style="width:33%;"></td>')
        rows.append('<tr>' + ''.join(chunk) + '</tr>')
    return (
        '<table role="presentation" cellpadding="0" cellspacing="6" border="0" '
        'style="width:100%;border-collapse:separate;border-spacing:6px;">'
        + ''.join(rows) + '</table>'
    )


def render_html(edition: NewsletterEdition) -> str:
    o: list[str] = []
    o.append('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    o.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    o.append(f'<title>{_esc(edition.email_subject)}</title>')
    # X-Atlas-Subject header so an email-generating wrapper can read the
    # canonical subject without re-parsing the document.
    o.append(f'<meta name="x-atlas-subject" content="{_esc(edition.email_subject)}">')
    if edition.country_focus:
        o.append(f'<meta name="x-atlas-country" content="{_esc(edition.country_focus)}">')
    o.append(_PRINT_CSS)
    o.append('</head>')
    o.append(f'<body style="{_S["wrap"]}">')
    o.append(f'<div style="{_S["container"]}">')

    # Editor-only "how to publish" callout. The wrapper is marked with
    # ``data-editor-only`` so it can be easily stripped by a Substack post-
    # ingest filter (or by hand). The user sees it on Tuesday morning in
    # Gmail; subscribers never see it because we only ship to subscribers
    # when the user pastes the Markdown into Substack and hits Publish.
    o.append(
        '<div data-editor-only="true" style="'
        'background:#fff8ea;border:1px solid #e0c07a;border-radius:8px;'
        'padding:10px 14px;margin-bottom:18px;font-size:13px;'
        'color:#7a5a10;font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;">'
        '<strong>📰 To publish this week:</strong> '
        'open Substack → Create → New post → paste the Markdown attached to '
        'this email (or copy the rendered version below). Eyeball the headline '
        'and lead story, then hit <strong>Publish</strong>. '
        '<em>Subscribers only see it after you click Publish.</em>'
        '</div>'
    )

    # Masthead
    o.append(
        f'<div style="{_S["kicker"]}">{_esc(edition.publication)} · at the UN this week · '
        f'Edition №{edition.edition_number}</div>'
    )
    o.append(f'<h1 style="{_S["h1"]}">{_esc(edition.headline)}</h1>')
    o.append(f'<p style="{_S["subhead"]}">{_esc(edition.subhead)}</p>')
    o.append(
        f'<p style="{_S["masthead_meta"]}">'
        f'{_esc(edition.dateline)} · {_esc(edition.byline)} · '
        f'{_esc(edition.period_label)}</p>'
    )

    # Lede + nut graf
    o.append(f'<p style="{_S["lede"]}">{_esc(edition.lede)}</p>')
    o.append(f'<p style="{_S["nut"]}">{_esc(edition.nut_graf)}</p>')

    # In this issue
    if edition.in_this_issue:
        o.append(f'<div style="{_S["toc"]}">')
        o.append(f'<div style="{_S["toc_label"]}">In this issue</div>')
        for item in edition.in_this_issue:
            o.append(
                f'<div style="{_S["toc_item"]}">'
                f'<strong style="color:#0e6f82;width:22px;display:inline-block;">'
                f'{item.number}.</strong> {_esc(item.title)}'
                f'</div>'
            )
        o.append('</div>')

    # By the numbers
    if edition.by_the_numbers:
        o.append(f'<h2 id="by-the-numbers" style="{_S["h2"]}">{_esc(SECTION_TITLES["by_the_numbers"])}</h2>')
        o.append(_stat_row_html(edition.by_the_numbers))

    # Top drifts chart (inline SVG — email-safe, no remote images).
    raw_drifts = edition.chart_payloads.get("top_drifts_raw") or []
    if raw_drifts:
        svg = top_drifts_svg(
            raw_drifts,
            name_lookup=edition.chart_payloads.get("name_lookup") or {},
        )
        o.append(svg)
        o.append(
            f'<p style="{_S["chart_caption"]}">'
            f'Top {len(raw_drifts)} country-pair alignment shifts in {edition.recent_year} '
            f'vs the {edition.baseline_window["start"]}–{edition.baseline_window["end"]} baseline.'
            f'</p>'
        )

    # Lead story
    o.append(
        f'<h2 id="the-shift" style="{_S["h2"]}">{_esc(SECTION_TITLES["shift"])}</h2>'
    )
    o.append(f'<div style="{_S["lead_card"]}">')
    o.append(f'<p style="margin:0 0 10px;font-size:15.5px;">{_esc(edition.lead_story.body)}</p>')
    if edition.lead_story.supporting_drifts:
        o.append('<p style="margin:6px 0 4px;font-weight:600;font-size:13px;color:#456783;">Supporting moves</p>')
        o.append('<ul style="margin:0;padding-left:20px;font-size:14px;">')
        for d in edition.lead_story.supporting_drifts:
            topics = [
                humanize_topic_full(t["topic"])
                for t in d.get("driving_topics", [])
            ][:2]
            topics = [t for t in topics if t]
            topic_str = f" — on {', '.join(topics)}" if topics else ""
            o.append(
                f'<li style="margin:2px 0;"><strong>{_esc(d["country_a"])} ↔ '
                f'{_esc(d["country_b"])}</strong>: '
                f'{d["baseline_agreement"] * 100:.0f}% → '
                f'{d["recent_agreement"] * 100:.0f}% '
                f'({_signed_pts(d["delta"])}){_esc(topic_str)}</li>'
            )
        o.append('</ul>')
    o.append('</div>')
    o.append(
        f'<div style="{_S["why_box"]}"><strong>Why it matters · </strong>'
        f'{_esc(edition.lead_story_why_it_matters)}</div>'
    )

    # Top movers
    if edition.top_movers:
        o.append(f'<h2 id="top-movers" style="{_S["h2"]}">{_esc(SECTION_TITLES["movers"])}</h2>')
        o.append(
            f'<div style="{_S["why_box"]}"><strong>Why it matters · </strong>'
            f'{_esc(edition.top_movers_why_it_matters)}</div>'
        )
        for mover in edition.top_movers:
            # Tiny P5-fingerprint chart — one row per P5 member, signed Δ.
            fingerprint = (
                p5_fingerprint_svg(mover.name, mover.p5_deltas)
                if mover.p5_deltas else ""
            )
            o.append(
                f'<div style="{_S["mover_card"]}">'
                f'<h3 style="{_S["h3"]}">{_esc(mover.name)} '
                f'<span style="color:#456783;font-weight:400;font-size:13px;">'
                f'({_esc(mover.country)})</span></h3>'
                f'<p style="margin:0 0 6px;color:#456783;font-style:italic;'
                f'font-size:13.5px;">{_esc(mover.headline)}</p>'
                f'<p style="margin:0 0 8px;font-size:15px;">{_esc(mover.detail)}</p>'
                f'{fingerprint}'
                f'</div>'
            )

    # Coalition watch
    if edition.coalition_watch:
        o.append(f'<h2 id="coalition-watch" style="{_S["h2"]}">{_esc(SECTION_TITLES["coalition"])}</h2>')
        o.append(
            f'<div style="{_S["why_box"]}"><strong>Why it matters · </strong>'
            f'{_esc(edition.coalition_why_it_matters)}</div>'
        )
        for snap in edition.coalition_watch:
            o.append(f'<h3 style="{_S["h3"]}">"{_esc(snap.topic.title())}"</h3>')
            o.append(f'<p style="margin:4px 0 10px;font-size:14.5px;">{_esc(snap.headline_stat)}</p>')
            # Visual tally — see at a glance which way the bloc leans.
            tally = snap.predicted_recent or {}
            o.append(vote_tally_svg(
                int(tally.get("yes", 0)),
                int(tally.get("no", 0)),
                int(tally.get("abstain", 0)),
                width=520,
                label=None,
            ))
            # Slope chart — the killer visualization for tier movement.
            # An almost-vertical line from top to bottom = the Argentina-
            # on-Palestine flip in pure visual form, no numbers needed.
            if snap.movers:
                base_label = (
                    f"{edition.baseline_window['start']}–"
                    f"{edition.baseline_window['end']}"
                )
                o.append(slope_chart_svg(
                    topic_title=snap.topic.title(),
                    movers=snap.movers,
                    base_label=base_label,
                    recent_label=str(edition.recent_year),
                ))
            if snap.movers:
                o.append(
                    f'<table role="presentation" style="{_S["table_outer"]}">'
                    f'<thead><tr>'
                    f'<th style="{_S["th"]}">Country</th>'
                    f'<th style="{_S["th"]}">Was</th>'
                    f'<th style="{_S["th"]}">Now</th>'
                    f'<th style="{_S["th"]}">Lean (base → recent)</th>'
                    f'</tr></thead><tbody>'
                )
                for m in snap.movers:
                    o.append(
                        f'<tr>'
                        f'<td style="{_S["td"]}">{_esc(m.name)} '
                        f'<span style="color:#456783;font-size:12px;">({_esc(m.country)})</span></td>'
                        f'<td style="{_S["td"]}">{_tier_pill(m.from_tier)}</td>'
                        f'<td style="{_S["td"]}">{_tier_pill(m.to_tier)}</td>'
                        f'<td style="{_S["td"]}">{m.mean_baseline:+.2f} → {m.mean_recent:+.2f}</td>'
                        f'</tr>'
                    )
                o.append('</tbody></table>')
            else:
                o.append(
                    '<p style="margin:0 0 12px;color:#456783;font-style:italic;'
                    'font-size:13.5px;">No country changed tier on this topic this period.</p>'
                )
            if snap.sample_titles:
                titles_str = "; ".join(f'"{_esc(t)}"' for t in snap.sample_titles[:2])
                o.append(
                    f'<p style="color:#456783;font-size:12.5px;margin:0 0 18px;">'
                    f'<em>Sample resolutions: {titles_str}</em></p>'
                )

    # Resolution spotlight
    if edition.resolution_spotlight:
        s = edition.resolution_spotlight
        o.append(f'<h2 id="resolution-spotlight" style="{_S["h2"]}">{_esc(SECTION_TITLES["spotlight"])}</h2>')
        o.append(f'<div style="{_S["spotlight_card"]}">')
        o.append(f'<h3 style="{_S["h3"]};color:#c64141;">"{_esc(s.title)}"</h3>')
        meta_bits = [s.date]
        if s.topic:
            meta_bits.append(f"topic: {s.topic}")
        o.append(
            f'<p style="color:#456783;font-size:12.5px;font-style:italic;margin:0 0 8px;">'
            f'{_esc(" · ".join(meta_bits))}</p>'
        )
        o.append(
            f'<p style="margin:0 0 8px;font-size:15px;">'
            f'The most contested vote of the period — {_esc(s.margin_summary)}.</p>'
        )
        # Visual bar to give the lopsidedness at a glance.
        o.append(vote_tally_svg(s.yes, s.no, s.abstain, width=560, label="Vote breakdown"))
        if s.breakaways:
            names = ", ".join(
                f"{_esc(b['name'])} ({_esc(b['country'])})"
                for b in s.breakaways[:12]
            )
            o.append(
                f'<p style="margin:0;font-size:14px;">'
                f'<strong>Voting against the majority:</strong> {names}.</p>'
            )
        o.append('</div>')

    # Quiet convergences
    if edition.quiet_convergences:
        o.append(f'<h2 id="quiet-convergences" style="{_S["h2"]}">{_esc(SECTION_TITLES["convergences"])}</h2>')
        top_c = edition.quiet_convergences[0]
        kicker_drift = {
            "country_a": top_c.country_a, "country_b": top_c.country_b,
            "baseline_agreement": top_c.baseline, "recent_agreement": top_c.recent,
            "delta": top_c.delta, "abs_delta": abs(top_c.delta),
            "driving_topics": [{"topic": t} for t in (top_c.top_topics or [])],
        }
        o.append(
            f'<p style="margin:0 0 10px;font-size:15px;color:#456783;">'
            f'{_esc(strange_bedfellows_intro(kicker_drift))}</p>'
        )
        # One dumbbell chart covering all pairs — more elegant than the
        # repeated dual-bar treatment and reads correctly on a single axis.
        dumbbell_rows = [
            (c.name_a, c.name_b, round(c.baseline * 100), round(c.recent * 100))
            for c in edition.quiet_convergences
        ]
        o.append(dumbbell_chart_svg(dumbbell_rows))
        # Detail cards underneath — still useful for the topic phrase.
        for c in edition.quiet_convergences:
            topic_str = f" on {', '.join(c.top_topics)}" if c.top_topics else ""
            base_i, rec_i = round(c.baseline * 100), round(c.recent * 100)
            o.append(
                f'<div style="{_S["convergence_card"]}">'
                f'<strong>{_esc(c.name_a)} ↔ {_esc(c.name_b)}</strong>: '
                f'{base_i}% → {rec_i}% '
                f'({_signed_pts(c.delta)}){_esc(topic_str)}'
                f'</div>'
            )

    # Next to watch
    if edition.next_to_watch:
        o.append(f'<h2 id="next-to-watch" style="{_S["h2"]}">{_esc(SECTION_TITLES["next"])}</h2>')
        for w in edition.next_to_watch:
            o.append(
                f'<div style="{_S["mover_card"]}">'
                f'<h3 style="{_S["h3"]}">{_esc(w.topic)}</h3>'
                f'<p style="margin:0;font-size:14.5px;">{_esc(w.rationale)}</p>'
                f'</div>'
            )

    # Footer — methodology + sources + freshness
    fr = edition.freshness
    o.append(f'<div id="methodology" style="{_S["footer"]}">')
    o.append(f'<div style="{_S["footer_h4"]}">Methodology</div><ul style="margin:6px 0 14px;padding-left:20px;">')
    for line in edition.methodology:
        o.append(f'<li style="margin:3px 0;">{_esc(line)}</li>')
    o.append('</ul>')
    o.append(f'<div style="{_S["footer_h4"]}">Sources</div><ul style="margin:6px 0 14px;padding-left:20px;">')
    for s_ in edition.sources:
        o.append(f'<li style="margin:3px 0;">{_esc(s_)}</li>')
    o.append('</ul>')
    o.append(
        f'<p style="margin:8px 0 0;"><em>Dataset: {fr["records"]:,} records, '
        f'{fr["min_year"]}–{fr["max_year"]}. Latest vote in source: '
        f'{_esc(fr.get("latest_vote_date") or "?")} '
        f'({fr.get("days_since_latest_vote", "?")} days ago).</em></p>'
    )
    o.append('</div>')

    o.append('</div></body></html>')
    return "\n".join(o)
