"""Round-trip tests for edition_from_dict.

The publish pipeline composes the edition once, archives it as JSON, and the
Send step reloads that JSON via ``edition_from_dict`` to build the email —
rather than recomposing. That only stays correct if a reloaded edition is
*identical* to the original: same nested structure, same rendered output, same
content_hash. These tests pin that guarantee, exercising the real archive path
(``save_edition`` -> JSON on disk -> ``edition_from_dict``) including the
``markdown`` convenience key that ``save_edition`` injects.
"""

from __future__ import annotations

import json

from src.newsletter import build_newsletter_edition, edition_from_dict, edition_to_dict
from src.newsletter_archive import save_edition
from src.newsletter_render import render_html, render_markdown, render_text
from tests.test_newsletter import _milei_df


def _edition():
    """A fully-populated edition (lead story, movers, coalition watch with
    tier-jumpers, spotlight, quiet convergences) to exercise every nested
    reconstruction path in edition_from_dict."""
    return build_newsletter_edition(_milei_df(), recent_year=2024, baseline_window_years=3)


def test_dict_roundtrip_is_identity():
    edition = _edition()
    assert edition_from_dict(edition_to_dict(edition)) == edition


def test_archived_json_roundtrip_renders_identically(tmp_path):
    """The exact workflow path: archive to disk, reload the JSON, reconstruct.
    The reconstructed edition must render byte-identically in all three formats
    and preserve the content_hash the gate/ledger keyed on."""
    edition = _edition()
    written = save_edition(edition, archive_dir=tmp_path)

    payload = json.loads(open(written["json"]).read())
    # save_edition injects a convenience 'markdown' key that is not a dataclass
    # field — reconstruction must tolerate it.
    assert "markdown" in payload
    reloaded = edition_from_dict(payload)

    assert reloaded.content_hash == edition.content_hash
    assert render_markdown(reloaded) == render_markdown(edition)
    assert render_html(reloaded) == render_html(edition)
    assert render_text(reloaded) == render_text(edition)


def test_roundtrip_preserves_nested_sections():
    """Spot-check that nested dataclass lists survive as objects, not dicts."""
    edition = _edition()
    reloaded = edition_from_dict(edition_to_dict(edition))

    assert reloaded.lead_story.headline == edition.lead_story.headline
    if edition.top_movers:
        assert reloaded.top_movers[0].p5_deltas == edition.top_movers[0].p5_deltas
        assert type(reloaded.top_movers[0]) is type(edition.top_movers[0])
    if edition.coalition_watch:
        orig_snap = edition.coalition_watch[0]
        new_snap = reloaded.coalition_watch[0]
        assert type(new_snap) is type(orig_snap)
        if orig_snap.movers:
            assert type(new_snap.movers[0]) is type(orig_snap.movers[0])
    if edition.resolution_spotlight is not None:
        assert reloaded.resolution_spotlight.rcid == edition.resolution_spotlight.rcid
