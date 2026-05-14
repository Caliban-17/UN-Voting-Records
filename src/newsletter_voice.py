"""
Editorial voice — punchier, gossipier, plain-English templates for the
newsletter. Pure functions, no LLM. The goal: keep the rigor but lose the
algorithm-speak so a journalist or policy analyst will actually read past
the first paragraph.

The right reference point is The Economist's Espresso / FT's Trade Secrets
— informed, slightly ironic, never tabloid. *"Argentina has had enough of
the non-aligned bloc"* rather than *"Argentina–Turkmenistan alignment fell
57 points."*
"""

from __future__ import annotations

import hashlib
import random


# ── Headlines ───────────────────────────────────────────────────────────────

# Headline templates indexed by drift direction. The composer picks one
# deterministically per edition (hashed on slug) so a given week always
# produces the same headline — readers can re-find an edition by URL.

_HEADLINES_DIVERGENCE = (
    "{a} cools on {b}: a {pts}-point UN voting rift",
    "{a} pulls away from {b}{topic_suffix}",
    "{a} and {b} part ways{topic_suffix}",
    "The {topic_or_session} divorce: {a} breaks with {b}",
    "{a} backs off {b}: alignment falls {pts} points",
)

_HEADLINES_CONVERGENCE = (
    "{a} cosies up to {b}: alignment up {pts} points",
    "Strange bedfellows: {a} and {b} converge{topic_suffix}",
    "{a} finds a new friend in {b}{topic_suffix}",
    "{a}–{b}: a {pts}-point UN voting thaw",
    "{a} doubles down with {b}{topic_suffix}",
)

# Subhead templates — one terse line under the headline.
_SUBHEADS_DIVERGENCE = (
    "Their UN votes lined up {baseline_pct}% of the time in the baseline period. This year, just {recent_pct}%.",
    "The pair voted together {baseline_pct}% of the time. Now {recent_pct}%. Something changed.",
    "From {baseline_pct}% agreement to {recent_pct}% — the period's biggest realignment.",
)

_SUBHEADS_CONVERGENCE = (
    "From {baseline_pct}% agreement to {recent_pct}% — the period's most unexpected alliance.",
    "Their voting record lined up {baseline_pct}% of the time, now {recent_pct}%. New geometry.",
    "An alliance that wasn't there before: {baseline_pct}% → {recent_pct}%.",
)


# ── Nut graf moods ──────────────────────────────────────────────────────────


def _every_or_n_of_n(n: int, total: int) -> str:
    """'All ten' when n == total, otherwise 'N of the top X'."""
    if n == total:
        # English number-words for small totals.
        words = {10: "ten", 9: "nine", 8: "eight", 7: "seven", 6: "six", 5: "five"}
        word = words.get(total, str(total))
        return f"all {word}"
    return f"{n} of the top {total}"


def nut_graf(n_div: int, n_conv: int, top_drift: dict | None) -> str:
    """Plain-English nut graf, calibrated to the session's overall mood."""
    if not top_drift:
        return (
            "An unusually quiet week at the UN — no significant realignments "
            "among the top ten pair-moves. Stability is itself worth noting."
        )
    total = n_div + n_conv
    if total == 0:
        return ""
    if n_div >= 7:
        mood = (
            f"This was a session of pulling apart, not coming together. "
            f"{_every_or_n_of_n(n_div, total).capitalize()} country-pair shifts "
            f"were divergences — countries walking away from old voting partners "
            f"rather than forming new ones. The geopolitics below."
        )
    elif n_conv >= 7:
        mood = (
            f"A rarer pattern: alliances hardened. "
            f"{_every_or_n_of_n(n_conv, total).capitalize()} shifts were "
            f"convergences. Bloc cohesion is up; new friendships are forming "
            f"faster than old ones are dissolving."
        )
    else:
        mood = (
            f"A session of reshuffling. {n_div} pairs drifted apart, {n_conv} "
            f"moved closer — realignment rather than uniform polarisation. "
            f"The new geometry is below."
        )
    return mood


# ── Top-mover plain-English descriptions ────────────────────────────────────


def mover_headline_for(name: str, score_pts: float) -> str:
    """The Economist-style one-liner above each top-mover card."""
    if score_pts >= 150:
        return f"{name} had the most volatile session of any UN member."
    if score_pts >= 100:
        return f"{name} is rewriting its UN voting record in real time."
    if score_pts >= 60:
        return f"{name} is on the move — broad-based realignment."
    return f"{name} is recalibrating."


def mover_detail_for(
    name: str,
    score_pts: float,
    anchor_drift: dict | None,
    name_lookup: dict[str, str],
) -> str:
    """Plain-English description of why this country is a top mover."""
    possessive = _possessive(name)
    if anchor_drift is None or score_pts < 20:
        # No single pair explains it — likely lots of small moves at once.
        return (
            f"{possessive} voting record with the five permanent Security Council "
            f"powers (US, UK, France, Russia, China) swung a combined "
            f"{score_pts:.0f} percentage points this session — across "
            f"multiple relationships, not one. The signal is breadth of "
            f"recalibration, not a single new alignment."
        )

    other = (
        anchor_drift["country_b"]
        if anchor_drift["country_a"] in name_lookup and name_lookup[anchor_drift["country_a"]] == name
        else anchor_drift["country_a"]
    )
    other_name = name_lookup.get(other, other)
    direction = (
        f"backed off {other_name}"
        if anchor_drift["delta"] < 0
        else f"moved sharply closer to {other_name}"
    )
    pair_pts = abs(anchor_drift["delta"]) * 100
    # Clean topic strings — drop the UNBIS double-dash hierarchy.
    topics = [
        humanize_topic_full(t["topic"])
        for t in (anchor_drift.get("driving_topics") or [])
    ][:2]
    topics = [t for t in topics if t]
    if topics:
        topic_phrase = (
            f" — driven by votes on {topics[0]}"
            + (f" and {topics[1]}" if len(topics) > 1 else "")
        )
    else:
        topic_phrase = ""
    return (
        f"{possessive} P5-alignment swung {score_pts:.0f} points in aggregate. "
        f"The biggest single move: it {direction} by "
        f"{pair_pts:.0f} points{topic_phrase}."
    )


# ── Strange Bedfellows section ──────────────────────────────────────────────


def strange_bedfellows_intro(top_convergence: dict | None) -> str:
    """The 'who's cosying up to whom?' kicker."""
    if not top_convergence:
        return ""
    return (
        "Not every story this session was a breakup. Quietly, away from the "
        "headline divergences, alliances are forming. The biggest:"
    )


def strange_bedfellows_one_liner(drift: dict, name_lookup: dict[str, str]) -> str:
    a = name_lookup.get(drift["country_a"], drift["country_a"])
    b = name_lookup.get(drift["country_b"], drift["country_b"])
    pts = abs(drift["delta"]) * 100
    base = round(drift["baseline_agreement"] * 100)
    rec = round(drift["recent_agreement"] * 100)
    topics = [t["topic"] for t in (drift.get("driving_topics") or [])][:1]
    topic_clause = f" — converging on {topics[0]}" if topics else ""
    return (
        f"**{a} and {b}** went from {base}% agreement to {rec}% "
        f"({pts:+.0f} points){topic_clause}."
    )


# ── Section titles — softer / journalistic ─────────────────────────────────


SECTION_TITLES = {
    "by_the_numbers": "By the Numbers",
    "shift": "This Week's Headline Split",
    "movers": "Who's Moving",
    "coalition": "If We Held the Vote Today",
    "spotlight": "The Vote of the Week",
    "convergences": "Strange Bedfellows",
    "next": "What to Watch Next",
    "methodology": "Methodology & Sources",
}


# ── Determinism helper ─────────────────────────────────────────────────────


def _pick(templates: tuple[str, ...], seed_text: str) -> str:
    """Pick one template deterministically based on a stable seed.

    Hashing the slug means re-composing the same edition always picks the
    same template — readers can bookmark and re-find an edition by URL, and
    the archived JSON is reproducible.
    """
    if not templates:
        return ""
    h = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    return templates[h % len(templates)]


# Hand-curated map: UNBISnet subject string → journalistic shorthand.
# The full UNBIS hierarchy ("TERRITORIES OCCUPIED BY ISRAEL--SETTLEMENT POLICY")
# is unreadable in a headline. These rewrites turn them into short, accurate
# noun phrases an editor would actually use. Match is case-insensitive on the
# pre-rewrite topic; the rewrite preserves the casing we want in output.
_TOPIC_SHORTHAND: dict[str, str] = {
    "territories occupied by israel--settlement policy": "Israeli settlements",
    "territories occupied by israel--natural resources": "the occupied territories",
    "territories occupied by israel--human rights--reports": "human rights in the occupied territories",
    "territories occupied by israel": "the occupied territories",
    "self-determination of peoples": "self-determination",
    "palestine question": "the Palestine question",
    "unrwa--activities": "UNRWA",
    "un--budget": "the UN budget",  # prefix match handles "un--budget (YYYY)" variants
    "un--administration": "UN administration",
    "un--general assembly (10th special sess. : 1978)": "the 10th-special-session disarmament debate",
    "nuclear weapon tests--legacy": "nuclear test legacies",
    "nuclear disarmament--conferences": "nuclear disarmament",
    "nuclear non-proliferation": "non-proliferation",
    "conventional arms--regional programmes": "regional arms control",
    "human rights advancement": "human rights",
    "human rights--reports": "human rights reporting",
    "armed conflicts prevention": "conflict prevention",
    "middle east situation": "the Middle East",
    "decolonization": "decolonisation",
    "disarmament--general and complete": "general disarmament",
    "sustainable development": "sustainable development",
    "information--international security": "information security",
}

# Words that should keep their canonical capitalization even inside an
# otherwise lower-cased headline ("on israeli settlements" reads wrong;
# "on Israeli settlements" reads right).
_PROPER_NOUNS = {
    "israel", "israeli", "palestine", "palestinian",
    "un", "unrwa", "iaea", "icc", "icj", "nato", "eu",
    "us", "usa", "uk", "russia", "russian", "china",
    "korean", "north korean", "iran", "iranian",
    "ukraine", "ukrainian", "syria", "syrian",
    "middle east",
}


def _restore_proper_nouns(text: str) -> str:
    """Title-case proper nouns in an otherwise lower-cased string."""
    if not text:
        return text
    # Word-boundary safe replacements.
    import re as _re
    out = text
    # Sort by length descending so longer multi-word forms hit first.
    for pn in sorted(_PROPER_NOUNS, key=len, reverse=True):
        pattern = _re.compile(rf"\b{_re.escape(pn)}\b", _re.IGNORECASE)
        # Capitalize each word in the matched form.
        replacement = " ".join(w.capitalize() for w in pn.split())
        out = pattern.sub(replacement, out)
    return out


def _short_topic(topic: str | None) -> str:
    """Shorten UNBISnet subject strings for headline use.

    Three-stage pipeline:
      1. Check the hand-curated shorthand map for a known rewrite.
      2. Fall back to "first segment before ``--``" with proper-noun
         capitalization preserved.
      3. Clamp to six words and lowercase for inline-in-sentence use.
    """
    if not topic:
        return ""
    t = str(topic).strip()
    # Hand-curated shorthand wins.
    key = t.lower()
    if key in _TOPIC_SHORTHAND:
        return _TOPIC_SHORTHAND[key]
    # Strip suffix after first UNBIS double-dash; also normalize spacing.
    head = t.split("--", 1)[0].strip()
    if head.isupper():
        head = head.capitalize()
    words = head.split()
    if len(words) > 6:
        head = " ".join(words[:6]) + "…"
    # Lower for inline use, then re-capitalize proper nouns.
    return _restore_proper_nouns(head.lower()) if head else ""


def humanize_topic_full(topic: str | None) -> str:
    """Like ``_short_topic`` but for the full topic — used in body text /
    'driven by' lists, where the full hierarchy is appropriate but the
    formatting should still be polite ("Territories occupied by Israel —
    settlement policy" not "TERRITORIES OCCUPIED BY ISRAEL--SETTLEMENT POLICY").
    """
    if not topic:
        return ""
    raw = str(topic).strip()
    key = raw.lower()
    if key in _TOPIC_SHORTHAND:
        # For body text, capitalize the shorthand's first letter.
        sh = _TOPIC_SHORTHAND[key]
        return sh[0].upper() + sh[1:] if sh else ""
    # Try prefix-match: "un--budget (2025)" matches "un--budget" entry too.
    for prefix_key, replacement in _TOPIC_SHORTHAND.items():
        if key.startswith(prefix_key + " "):
            # Reattach any trailing parenthetical / year for context.
            suffix = raw[len(prefix_key):].strip()
            return f"{replacement[0].upper() + replacement[1:]} {suffix}".strip()
    # Convert UNBIS "--" → " — " (em-dash) so it reads as a hierarchy
    # rather than a typographic error.
    parts = [p.strip() for p in raw.split("--") if p.strip()]
    if not parts:
        return raw
    out = " — ".join(parts)
    # Capitalize the first word only if it's all-caps (UNBIS convention).
    if out.split() and out.split()[0].isupper():
        first = parts[0].capitalize()
        rest = parts[1:]
        out = first + (" — " + " — ".join(p.lower() for p in rest) if rest else "")
    return _restore_proper_nouns(out)


def _possessive(name: str) -> str:
    """English possessive — handles names ending in 's' correctly.

    'Marshall Islands' → 'Marshall Islands''   (no extra s)
    'Argentina'        → "Argentina's"
    """
    if not name:
        return name
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    """Singular/plural agreement helper."""
    return singular if n == 1 else (plural or f"{singular}s")


def pick_headline(drift: dict, name_lookup: dict[str, str], seed: str) -> str:
    a = name_lookup.get(drift["country_a"], drift["country_a"])
    b = name_lookup.get(drift["country_b"], drift["country_b"])
    pts = abs(drift["delta"]) * 100
    topics = [t["topic"] for t in (drift.get("driving_topics") or [])][:1]
    short_topic = _short_topic(topics[0]) if topics else ""
    topic_phrase = f" on {short_topic}" if short_topic else ""
    template = _pick(
        _HEADLINES_DIVERGENCE if drift["delta"] < 0 else _HEADLINES_CONVERGENCE,
        seed_text=seed,
    )
    return template.format(
        a=a,
        b=b,
        pts=f"{pts:.0f}",
        topic_suffix=topic_phrase,
        topic_or_session=(short_topic or "session"),
    )


def pick_subhead(drift: dict, seed: str) -> str:
    template = _pick(
        _SUBHEADS_DIVERGENCE if drift["delta"] < 0 else _SUBHEADS_CONVERGENCE,
        seed_text=seed,
    )
    return template.format(
        baseline_pct=round(drift["baseline_agreement"] * 100),
        recent_pct=round(drift["recent_agreement"] * 100),
    )
