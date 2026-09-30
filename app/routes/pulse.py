"""Published research and current context, independent of voting-data requests."""

import json
from datetime import datetime
from email.utils import format_datetime
import xml.etree.ElementTree as ET

from flask import Blueprint, Response, jsonify, render_template, request, url_for

from src.pulse import UTC, briefing, data_dir, date_value, read_snapshot, rss

bp = Blueprint("pulse", __name__)


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
    return render_template("briefing.html", research=research(), pulse=briefing(read_snapshot(), topic=topic, issue=issue, query=query), topic=topic, issue=issue, query=query)


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
    return render_template("briefing.html", research=edition, pulse=briefing(read_snapshot()), topic="", issue="", query="")


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


@bp.get("/newsletter/feed.xml")
def newsletter_feed():
    root = ET.Element("rss", version="2.0")
    channel = ET.SubElement(root, "channel")
    ET.SubElement(channel, "title").text = "UN-Scrupulous | The UN newsletter"
    ET.SubElement(channel, "link").text = url_for("pulse.newsletter", _external=True)
    ET.SubElement(channel, "description").text = "Automatically published decisions, world affairs and political research."
    # Archives are written once, so mtime orders them by first publication:
    # parse only the newest instead of every edition ever archived.
    editions = []
    for path in sorted((data_dir() / "newsletters").glob("*.json"), key=_mtime, reverse=True)[:60]:
        try:
            editions.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    for edition in sorted(editions, key=lambda e: (e["edition_date"], e["content_hash"]), reverse=True)[:30]:
        node = ET.SubElement(channel, "item")
        ET.SubElement(node, "title").text = edition["headline"]
        ET.SubElement(node, "guid", isPermaLink="false").text = edition["content_hash"]
        ET.SubElement(node, "link").text = url_for("pulse.newsletter_edition", edition_id=edition["content_hash"], _external=True)
        ET.SubElement(node, "pubDate").text = format_datetime(date_value(edition["edition_date"]))
        ET.SubElement(node, "description").text = edition["lede"]
    return Response(ET.tostring(root, encoding="utf-8", xml_declaration=True), mimetype="application/rss+xml")
