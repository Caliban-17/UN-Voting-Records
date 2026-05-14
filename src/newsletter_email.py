"""
Email-message builder for UN-Scrupulous.

Gmail (and Outlook in particular) strips inline ``<svg>`` tags as a security
measure, so the standalone HTML render works in a browser but loses every
chart when delivered as an email. This module solves that:

1. Take the standard rendered HTML from :func:`newsletter_render.render_html`.
2. Find every inline ``<svg>…</svg>`` block.
3. Convert each one to a PNG via CairoSVG.
4. Replace the SVG markup with ``<img src="cid:<uuid>" alt="…">``.
5. Build a multipart MIME message with the HTML as the root and each PNG
   attached as a related part with the matching Content-ID.

Result: the same edition shows up in Gmail / Outlook / Apple Mail with the
charts actually rendered, no remote image fetches, and the text fallback
(text/plain) is unchanged. This is the canonical email-image embedding
pattern; every major email client supports it.
"""

from __future__ import annotations

import logging
import re
import uuid
from email.message import EmailMessage
from typing import Iterable

import cairosvg

from src.newsletter import NewsletterEdition
from src.newsletter_render import render_html, render_text

logger = logging.getLogger(__name__)


# Greedy capture across newlines — every SVG block becomes one match.
_SVG_RE = re.compile(r"<svg\b[^>]*>.*?</svg>", re.IGNORECASE | re.DOTALL)
# Pull the width attribute (if any) so we render the PNG at a sensible size.
_WIDTH_RE = re.compile(r'\bwidth="(\d+)(?:px)?"', re.IGNORECASE)
_HEIGHT_RE = re.compile(r'\bheight="(\d+)(?:px)?"', re.IGNORECASE)
_LABEL_RE = re.compile(r'\baria-label="([^"]+)"', re.IGNORECASE)


def _svg_dimensions(svg: str, fallback_width: int = 640) -> tuple[int, int | None]:
    """Read explicit width/height off the SVG element. Height may be None
    (preserve aspect ratio)."""
    w_match = _WIDTH_RE.search(svg[:500])
    h_match = _HEIGHT_RE.search(svg[:500])
    width = int(w_match.group(1)) if w_match else fallback_width
    height = int(h_match.group(1)) if h_match else None
    return width, height


def _svg_alt_text(svg: str) -> str:
    m = _LABEL_RE.search(svg[:1024])
    return m.group(1) if m else "Chart"


def _svg_to_png_bytes(svg: str) -> bytes:
    """Convert an inline ``<svg>…</svg>`` string to PNG bytes.

    We render at 2× the native size so the resulting PNG looks crisp on
    Retina / HiDPI mail clients. Email clients usually scale the image back
    via the ``width`` attribute on the ``<img>`` tag, so the on-disk PNG
    being larger isn't a problem.
    """
    native_w, _native_h = _svg_dimensions(svg)
    return cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=native_w * 2,
    )


def _replace_svgs_with_cid_images(html: str) -> tuple[str, list[tuple[str, bytes, int]]]:
    """Substitute every ``<svg>`` block with an ``<img cid:…>`` tag.

    Returns the rewritten HTML plus a list of ``(cid, png_bytes, display_w)``
    tuples so the caller can attach the corresponding image parts.
    """
    attachments: list[tuple[str, bytes, int]] = []

    def _replace(match: re.Match) -> str:
        svg = match.group(0)
        cid = uuid.uuid4().hex
        display_w, _ = _svg_dimensions(svg)
        png = _svg_to_png_bytes(svg)
        alt = _svg_alt_text(svg)
        attachments.append((cid, png, display_w))
        # Mail clients ignore inline CSS on <img>, so set width/height as attrs.
        return (
            f'<img src="cid:{cid}" alt="{alt}" '
            f'width="{display_w}" style="display:block;max-width:100%;'
            f'height:auto;border:0;outline:0;text-decoration:none;">'
        )

    new_html = _SVG_RE.sub(_replace, html)
    return new_html, attachments


def build_email_message(
    edition: NewsletterEdition,
    sender: str,
    recipient: str,
    subject_override: str | None = None,
) -> EmailMessage:
    """Build a ready-to-send multipart/alternative+related ``EmailMessage``.

    The structure is:

    * multipart/alternative
      * text/plain (the unchanged plain-text version)
      * multipart/related
        * text/html (the rewritten HTML with cid: references)
        * image/png (one part per chart, each with a matching Content-ID)
    """
    raw_html = render_html(edition)
    raw_text = render_text(edition)
    rewritten_html, attachments = _replace_svgs_with_cid_images(raw_html)
    logger.info(
        "Built email body with %d inline charts (CID-attached).",
        len(attachments),
    )

    msg = EmailMessage()
    msg["Subject"] = subject_override or edition.email_subject
    msg["From"] = sender
    msg["To"] = recipient

    # Plain-text fallback — first so clients that don't render HTML still
    # see useful content. multipart/alternative is created implicitly.
    msg.set_content(raw_text)

    # HTML alternative with related image parts.
    msg.add_alternative(rewritten_html, subtype="html")
    html_part = msg.get_payload()[-1]
    for cid, png, _w in attachments:
        html_part.add_related(
            png,
            maintype="image",
            subtype="png",
            cid=f"<{cid}>",
            filename=f"{cid}.png",
        )

    # Markdown attachment — the file the editor drags into a Substack draft
    # each Tuesday. Single source of truth for the post body; Substack's
    # editor accepts Markdown paste-and-publish.
    from src.newsletter_render import render_markdown
    msg.add_attachment(
        render_markdown(edition).encode("utf-8"),
        maintype="text",
        subtype="markdown",
        filename=f"{edition.edition_slug}.md",
    )

    return msg


def render_html_with_cid_images(
    edition: NewsletterEdition,
) -> tuple[str, list[tuple[str, bytes, int]]]:
    """Standalone helper: rewrite + return attachment payloads.

    Useful when the caller wants to build its own MIME envelope (e.g. a
    GitHub Actions workflow that posts to Beehiiv via a different transport).
    """
    raw_html = render_html(edition)
    return _replace_svgs_with_cid_images(raw_html)
