"""Tests for the email-message builder — CID-attached chart PNGs."""

from __future__ import annotations

import pytest

from src.newsletter import build_newsletter_edition
from src.newsletter_email import (
    _replace_svgs_with_cid_images,
    _svg_alt_text,
    _svg_dimensions,
    build_email_message,
)


def _df():
    from tests.test_newsletter import _milei_df
    return _milei_df()


def test_svg_dimensions_parses_width_height():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="200">…</svg>'
    w, h = _svg_dimensions(svg)
    assert w == 640
    assert h == 200


def test_svg_alt_text_reads_aria_label():
    svg = '<svg aria-label="A test chart" width="100"></svg>'
    assert _svg_alt_text(svg) == "A test chart"
    assert _svg_alt_text("<svg></svg>") == "Chart"


def test_replace_svgs_with_cid_images_strips_every_svg():
    """Every inline <svg> becomes an <img cid:…> reference."""
    df = _df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    from src.newsletter_render import render_html
    html = render_html(edition)
    svg_count_before = html.lower().count("<svg")
    rewritten, attachments = _replace_svgs_with_cid_images(html)
    assert rewritten.lower().count("<svg") == 0
    assert rewritten.count('src="cid:') == svg_count_before
    assert len(attachments) == svg_count_before


def test_replace_svgs_produces_valid_png_bytes():
    df = _df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    from src.newsletter_render import render_html
    html = render_html(edition)
    _, attachments = _replace_svgs_with_cid_images(html)
    if not attachments:
        pytest.skip("fixture produced no charts")
    cid, png_bytes, width = attachments[0]
    # PNG magic number — first 8 bytes
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(cid) >= 16
    assert width > 0


def test_build_email_message_has_text_html_and_inline_images():
    df = _df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    msg = build_email_message(
        edition, sender="from@example.com", recipient="to@example.com",
    )
    assert msg["Subject"] == edition.email_subject
    assert msg["From"] == "from@example.com"
    assert msg["To"] == "to@example.com"
    # Walk parts: expect text/plain + text/html + image/png attachments.
    types = [part.get_content_type() for part in msg.walk()]
    assert "text/plain" in types
    assert "text/html" in types
    # As long as the fixture has at least one chart, there's at least one PNG.
    if edition.chart_payloads.get("top_drifts_raw"):
        assert "image/png" in types
        # Each image part should have a Content-ID header.
        for part in msg.walk():
            if part.get_content_type() == "image/png":
                assert part["Content-ID"] is not None
