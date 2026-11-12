"""Published research and current context, independent of voting-data requests."""

import json
from datetime import datetime
from email.utils import format_datetime
import xml.etree.ElementTree as ET

from flask import Blueprint, Response, jsonify, render_template, request, url_for

from src.pulse import UTC, briefing, data_dir, date_value, read_snapshot, rss

bp = Blueprint("pulse", __name__)


def bloc_view(brief):
    """The map and its captions for a research brief that carries blocs."""
    blocs = brief.get("blocs")
    if not blocs:
        return None
    try:
        from app.services import country_names
        from src.world_map import bloc_lines, bloc_map_svg, bloc_title
        names = country_names() or {}
        return {"svg": bloc_map_svg(blocs, bloc_title(blocs, names)), "lines": bloc_lines(blocs, names)}
    except Exception:
        return None


def research():
    try:
        result = json.loads((data_dir() / "research.json").read_text())
        through = date_value(result.get("data_through"))
        result["data_age_days"] = (datetime.now(UTC) - through).days if through else None
        return result
    except (OSError, ValueError):
        return {"status": "insufficient", "findings": [], "message": "Research is preparing automatically. No findings are published until voting data is available."}


@bp.get("/api/pulse")
def pulse_api():
    return jsonify(briefing(read_snapshot(), topic=request.args.get("topic", ""), issue=request.args.get("issue", ""), query=request.args.get("q", "")[:200]))


@bp.get("/api/research-brief")
def research_api():
    return jsonify(research())


@bp.get("/")
@bp.get("/briefing")
def publication():
    topic, issue, query = request.args.get("topic", ""), request.args.get("issue", ""), request.args.get("q", "")[:200]
    brief = research()
    return render_template("briefing.html", research=brief, blocs=bloc_view(brief), pulse=briefing(read_snapshot(), topic=topic, issue=issue, query=query), topic=topic, issue=issue, query=query)


@bp.get("/briefing/feed.xml")
def context_feed():
    return Response(rss(read_snapshot(), url_for("pulse.publication", _external=True)), mimetype="application/rss+xml")


@bp.get("/briefing/research.xml")
def research_feed():
    root = ET.Element("rss", version="2.0")
    channel = ET.SubElement(root, "channel")
    ET.SubElement(channel, "title").text = "UN-Scrupulous | Political research"
    ET.SubElement(channel, "link").text = url_for("pulse.publication", _external=True)
    ET.SubElement(channel, "description").text = "New editions only when the underlying research changes."
    editions = []
    for path in (data_dir() / "research").glob("*.json"):
        try:
            editions.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    for edition in sorted(editions, key=lambda x: x["generated_at"], reverse=True)[:30]:
        node = ET.SubElement(channel, "item")
        ET.SubElement(node, "title").text = f"{edition['title']} — {edition['period']}"
        ET.SubElement(node, "guid", isPermaLink="false").text = edition["content_hash"]
        ET.SubElement(node, "link").text = url_for("pulse.research_edition", edition_id=edition["content_hash"][:16], _external=True)
        ET.SubElement(node, "pubDate").text = format_datetime(date_value(edition["generated_at"]))
        ET.SubElement(node, "description").text = "\n\n".join(f"{f['title']}. {f['finding']} {f['caveat']}" for f in edition["findings"])
    return Response(ET.tostring(root, encoding="utf-8", xml_declaration=True), mimetype="application/rss+xml")


@bp.get("/briefing/editions/<edition_id>")
def research_edition(edition_id):
    import re
    if not re.fullmatch(r"[a-f0-9]{16}", edition_id):
        return jsonify({"error": "Unknown edition"}), 404
    path = data_dir() / "research" / f"{edition_id}.json"
    try:
        edition = json.loads(path.read_text())
    except (OSError, ValueError):
        return jsonify({"error": "Unknown edition"}), 404
    # Archived research is immutable; current source context is labelled separately.
    return render_template("briefing.html", research=edition, blocs=bloc_view(edition), pulse=briefing(read_snapshot()), topic="", issue="", query="")


@bp.get("/newsletter")
def newsletter():
    from src.newsletter import edition_from_dict
    from src.newsletter_render import render_html
    try:
        edition = edition_from_dict(json.loads((data_dir() / "newsletter.json").read_text()))
    except (OSError, ValueError):
        return Response("The newsletter is preparing automatically. Please check again shortly.", status=503, mimetype="text/plain")
    return Response(render_html(edition), mimetype="text/html")


@bp.get("/api/decisions")
def decisions_api():
    from src.un_decisions import read_decisions
    return jsonify(read_decisions())


@bp.get("/newsletter/editions/<edition_id>")
def newsletter_edition(edition_id):
    import re
    from src.newsletter import edition_from_dict
    from src.newsletter_render import render_html
    if not re.fullmatch(r"[a-f0-9]{64}", edition_id):
        return jsonify({"error": "Unknown edition"}), 404
    try:
        payload = json.loads((data_dir() / "newsletters" / f"{edition_id}.json").read_text())
    except (OSError, ValueError):
        return jsonify({"error": "Unknown edition"}), 404
    return Response(render_html(edition_from_dict(payload)), mimetype="text/html")


def _mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def recent_editions(limit=30):
    """Archived editions, newest first. Archives are written once, so mtime
    orders them by first publication: only the newest are parsed."""
    editions = []
    for path in sorted((data_dir() / "newsletters").glob("*.json"), key=_mtime, reverse=True)[:limit * 2]:
        try:
            editions.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(editions, key=lambda e: (e["edition_date"], e["content_hash"]), reverse=True)[:limit]


ARCHIVE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<link rel="icon" href="/favicon.svg" type="image/svg+xml"><link rel="alternate icon" href="/favicon.ico">
<link rel="alternate" type="application/rss+xml" title="UN-Scrupulous" href="/newsletter/feed.xml">
<title>UN-Scrupulous: past editions</title></head>
<body style="margin:0;background:#f3f1eb;padding:28px 12px;font-family:Georgia,serif;color:#102c35">
<main style="max-width:688px;margin:0 auto;background:#fffdf8;padding:28px 24px">
<p style="font:12px Arial,sans-serif;letter-spacing:1px;text-transform:uppercase;color:#5d6f73;border-bottom:3px solid #102c35;padding-bottom:14px;margin:0">
<a href="/briefing" style="color:inherit;text-decoration:none">UN-Scrupulous</a> · Past editions</p>
{items}
<p style="font:13px/1.55 Arial,sans-serif;color:#5d6f73;border-top:1px solid #cbd5d3;padding-top:14px;margin-top:30px">
<a href="/newsletter" style="color:inherit">Latest edition</a> · <a href="/subscribe" style="color:inherit">Get it by email</a> ·
<a href="/newsletter/feed.xml" style="color:inherit">Feed for RSS readers</a></p>
</main></body></html>"""


@bp.get("/newsletter/archive")
def newsletter_archive():
    """Past editions as a page to read; the feed is for feed readers."""
    from html import escape
    from src.newsletter_story import day
    items = "".join(
        f'<article style="margin:26px 0"><p style="font:12.5px Arial,sans-serif;color:#5d6f73;margin:0 0 4px">{escape(day(e["edition_date"], year=True))}</p>'
        f'<h2 style="font-size:22px;line-height:1.2;margin:0 0 6px"><a href="/newsletter/editions/{escape(e["content_hash"])}" '
        f'style="color:#102c35;text-decoration:none">{escape(e["headline"])}</a></h2>'
        f'<p style="font-size:15.5px;line-height:1.55;color:#243d44;margin:0">{escape(e["lede"][:320])}{"…" if len(e["lede"]) > 320 else ""}</p></article>'
        for e in recent_editions(60))
    return Response(ARCHIVE.format(items=items or '<p style="font-size:17px;margin:26px 0">No editions have been published yet.</p>'),
                    mimetype="text/html")


@bp.get("/newsletter/feed.xml")
def newsletter_feed():
    root = ET.Element("rss", version="2.0")
    channel = ET.SubElement(root, "channel")
    ET.SubElement(channel, "title").text = "UN-Scrupulous | The UN newsletter"
    ET.SubElement(channel, "link").text = url_for("pulse.newsletter", _external=True)
    ET.SubElement(channel, "description").text = "Automatically published decisions, world affairs and political research."
    for edition in recent_editions():
        node = ET.SubElement(channel, "item")
        ET.SubElement(node, "title").text = edition["headline"]
        ET.SubElement(node, "guid", isPermaLink="false").text = edition["content_hash"]
        ET.SubElement(node, "link").text = url_for("pulse.newsletter_edition", edition_id=edition["content_hash"], _external=True)
        ET.SubElement(node, "pubDate").text = format_datetime(date_value(edition["edition_date"]))
        ET.SubElement(node, "description").text = edition["lede"]
    return Response(ET.tostring(root, encoding="utf-8", xml_declaration=True), mimetype="application/rss+xml")
