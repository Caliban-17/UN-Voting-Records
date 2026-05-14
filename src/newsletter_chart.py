"""
Inline SVG renderer for the newsletter's top-drifts chart.

Hand-rolled SVG so the markup is self-contained, dependency-free, and
embeds cleanly in HTML email (no remote images, no JS, no kaleido / orca).
"""

from __future__ import annotations

import html as _html

# Palette aligned with the Atlas brand (sea-green for up, ember-red for down).
_COLOR_UP = "#2a9d57"
_COLOR_DOWN = "#c64141"
_COLOR_AXIS = "#456783"
_COLOR_GRID = "#d7d3c9"
_COLOR_TEXT = "#0b2238"
_FONT = "Georgia, 'Palatino Linotype', serif"


def _esc(s: str) -> str:
    return _html.escape(str(s or ""))


# Tier ladder used by the slope chart. Top of the chart = most supportive,
# bottom = most opposed. Must match the tiers in src.coalition.TIER_THRESHOLDS.
_TIER_LADDER = (
    "Champion supporter",
    "Reliable supporter",
    "Leans supporter",
    "Fence-sitter",
    "Leans opposed",
    "Reliable opposed",
    "Champion opposed",
)


def slope_chart_svg(
    topic_title: str,
    movers: list,  # list[CoalitionMove]-shaped duck type, dot-attr access
    width: int = 560,
    base_label: str = "baseline",
    recent_label: str = "this year",
) -> str:
    """Two-column slope chart — tier ladder on Y, baseline on the left,
    recent on the right, one connecting line per country.

    The visual reading: a Champion supporter who fell to Fence-sitter
    becomes a sharp red downward stroke from the top of the chart to the
    middle. Argentina's flip from Champion supporter (top) to Champion
    opposed (bottom) on Palestine is an almost-vertical red line down the
    full height — the kind of visual that the bar chart can't deliver.
    """
    if not movers:
        return ""

    # Sizing — generous rows so 4-deep label stacks fit cleanly.
    top_margin = 40
    bottom_margin = 28
    tier_count = len(_TIER_LADDER)
    row_h = 44
    height = top_margin + bottom_margin + (tier_count - 1) * row_h + 8

    left_col_x = 150
    right_col_x = width - 150
    dot_r = 5

    # Y positions per tier — top of ladder = top of chart.
    def _tier_y(tier_name: str) -> float:
        try:
            idx = _TIER_LADDER.index(tier_name)
        except ValueError:
            idx = tier_count // 2
        return top_margin + idx * row_h

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="Tier-jumper slope chart for {_esc(topic_title)}" '
        f'style="font-family:Georgia,\'Palatino Linotype\',\'Times New Roman\',serif;">'
    )

    # Title — small kicker above the chart.
    parts.append(
        f'<text x="{width / 2:.0f}" y="14" font-size="12" font-weight="700" '
        f'fill="{_COLOR_TEXT}" text-anchor="middle">'
        f'Tier movement on "{_esc(topic_title)}"</text>'
    )

    # Column headers.
    parts.append(
        f'<text x="{left_col_x}" y="{top_margin - 14}" font-size="11" '
        f'font-weight="700" fill="{_COLOR_AXIS}" text-anchor="middle" '
        f'letter-spacing="0.05em">{_esc(base_label.upper())}</text>'
    )
    parts.append(
        f'<text x="{right_col_x}" y="{top_margin - 14}" font-size="11" '
        f'font-weight="700" fill="{_COLOR_AXIS}" text-anchor="middle" '
        f'letter-spacing="0.05em">{_esc(recent_label.upper())}</text>'
    )

    # Tier-row guidelines + side labels.
    for i, tier in enumerate(_TIER_LADDER):
        y = top_margin + i * row_h
        # Faint horizontal guide
        parts.append(
            f'<line x1="{left_col_x}" y1="{y}" x2="{right_col_x}" y2="{y}" '
            f'stroke="{_COLOR_GRID}" stroke-width="0.5" stroke-dasharray="2 4"/>'
        )
        # Left margin tier label
        parts.append(
            f'<text x="{left_col_x - dot_r - 8}" y="{y + 4}" '
            f'font-size="11" fill="{_COLOR_AXIS}" text-anchor="end">'
            f'{_esc(tier)}</text>'
        )
        # Right margin tier label
        parts.append(
            f'<text x="{right_col_x + dot_r + 8}" y="{y + 4}" '
            f'font-size="11" fill="{_COLOR_AXIS}" text-anchor="start">'
            f'{_esc(tier)}</text>'
        )

    # Movers: connecting lines first (so dots and labels sit on top of them).
    # Sort so the most-moved entries draw last and visually dominate.
    sorted_movers = sorted(
        movers,
        key=lambda m: abs(_TIER_LADDER.index(m.to_tier) - _TIER_LADDER.index(m.from_tier))
        if m.to_tier in _TIER_LADDER and m.from_tier in _TIER_LADDER else 0,
    )
    for m in sorted_movers:
        if m.from_tier not in _TIER_LADDER or m.to_tier not in _TIER_LADDER:
            continue
        y1 = _tier_y(m.from_tier)
        y2 = _tier_y(m.to_tier)
        # Up the ladder (lower idx) = green; down = red.
        idx_from = _TIER_LADDER.index(m.from_tier)
        idx_to = _TIER_LADDER.index(m.to_tier)
        line_color = _COLOR_UP if idx_to < idx_from else _COLOR_DOWN
        parts.append(
            f'<line x1="{left_col_x}" y1="{y1:.1f}" '
            f'x2="{right_col_x}" y2="{y2:.1f}" '
            f'stroke="{line_color}" stroke-width="2" opacity="0.75"/>'
        )

    # Collect unique endpoint positions so we draw ONE dot per tier-side,
    # not one per mover. This avoids the "last-mover-wins colour" bug.
    from collections import defaultdict
    left_groups: dict[float, list] = defaultdict(list)
    right_groups: dict[float, list] = defaultdict(list)
    for m in sorted_movers:
        if m.from_tier not in _TIER_LADDER or m.to_tier not in _TIER_LADDER:
            continue
        left_groups[_tier_y(m.from_tier)].append(m)
        right_groups[_tier_y(m.to_tier)].append(m)

    # Neutral slate-grey dots at endpoints — they mark POSITIONS, the lines
    # carry direction. Avoids misleading colours when multiple movers share
    # a tier endpoint (e.g. three countries all sitting at "Reliable opposed").
    dot_color = _COLOR_TEXT
    for y in set(left_groups) | set(right_groups):
        if y in left_groups:
            parts.append(
                f'<circle cx="{left_col_x}" cy="{y:.1f}" r="{dot_r}" '
                f'fill="{dot_color}"/>'
            )
        if y in right_groups:
            parts.append(
                f'<circle cx="{right_col_x}" cy="{y:.1f}" r="{dot_r}" '
                f'fill="{dot_color}"/>'
            )

    # Country code labels — stacked vertically when multiple share a tier.
    # 14px line-height (vs the previous 11) so labels don't overlap when
    # CairoSVG falls back to a wider font.
    label_lh = 14
    max_visible = 4  # if more than this share a tier, show "+N more"
    for y, group in left_groups.items():
        visible = group[:max_visible]
        for j, m in enumerate(visible):
            parts.append(
                f'<text x="{left_col_x + dot_r + 6}" '
                f'y="{y + 4 + j * label_lh:.1f}" '
                f'font-size="11" font-weight="700" fill="{_COLOR_TEXT}" '
                f'text-anchor="start">{_esc(m.country)}</text>'
            )
        if len(group) > max_visible:
            parts.append(
                f'<text x="{left_col_x + dot_r + 6}" '
                f'y="{y + 4 + max_visible * label_lh:.1f}" '
                f'font-size="10" fill="{_COLOR_AXIS}" '
                f'text-anchor="start">+{len(group) - max_visible} more</text>'
            )
    for y, group in right_groups.items():
        visible = group[:max_visible]
        for j, m in enumerate(visible):
            parts.append(
                f'<text x="{right_col_x - dot_r - 6}" '
                f'y="{y + 4 + j * label_lh:.1f}" '
                f'font-size="11" font-weight="700" fill="{_COLOR_TEXT}" '
                f'text-anchor="end">{_esc(m.country)}</text>'
            )
        if len(group) > max_visible:
            parts.append(
                f'<text x="{right_col_x - dot_r - 6}" '
                f'y="{y + 4 + max_visible * label_lh:.1f}" '
                f'font-size="10" fill="{_COLOR_AXIS}" '
                f'text-anchor="end">+{len(group) - max_visible} more</text>'
            )

    parts.append('</svg>')
    return "".join(parts)


def dumbbell_chart_svg(
    pairs: list[tuple[str, str, int, int]],  # (name_a, name_b, baseline_pct, recent_pct)
    width: int = 560,
    row_height: int = 32,
    label_col_w: int = 220,
) -> str:
    """Dumbbell chart for Strange Bedfellows.

    Each row: pair name on the left, then a 0–100% axis with two dots
    (baseline in muted slate, recent in sea-green) connected by a heavy
    line. Reads as "this is how much the alliance moved" without forcing
    the reader to subtract two numbers.
    """
    if not pairs:
        return ""

    top_margin = 32
    bottom_margin = 30
    n = len(pairs)
    height = top_margin + bottom_margin + n * row_height

    plot_left = label_col_w + 8
    plot_right = width - 64           # leave room for trailing pct
    plot_w = plot_right - plot_left

    def _x(pct: int) -> float:
        return plot_left + (max(0, min(100, pct)) / 100.0) * plot_w

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="Alliance change dumbbell chart" '
        f'style="font-family:Georgia,\'Palatino Linotype\',\'Times New Roman\',serif;">'
    )
    # Title
    parts.append(
        f'<text x="{plot_left}" y="14" font-size="12" font-weight="700" '
        f'fill="{_COLOR_TEXT}">Alliance shift · agreement %</text>'
    )

    # Axis ticks at 0/25/50/75/100
    for tick in (0, 25, 50, 75, 100):
        x = _x(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top_margin - 6}" '
            f'x2="{x:.1f}" y2="{height - bottom_margin + 4}" '
            f'stroke="{_COLOR_GRID}" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - bottom_margin + 18}" '
            f'font-size="10" fill="{_COLOR_AXIS}" text-anchor="middle">{tick}%</text>'
        )

    # Each pair as a dumbbell row.
    for i, (name_a, name_b, base, rec) in enumerate(pairs):
        y = top_margin + 16 + i * row_height
        label = f"{name_a} — {name_b}"
        if len(label) > 32:
            label = label[:30].rstrip() + "…"
        parts.append(
            f'<text x="{label_col_w}" y="{y + 4}" font-size="12" '
            f'fill="{_COLOR_TEXT}" text-anchor="end">{_esc(label)}</text>'
        )
        x_base = _x(base)
        x_rec = _x(rec)
        # Heavy connecting line
        parts.append(
            f'<line x1="{x_base:.1f}" y1="{y}" x2="{x_rec:.1f}" y2="{y}" '
            f'stroke="{_COLOR_UP}" stroke-width="3" opacity="0.55"/>'
        )
        # Baseline dot (muted) and recent dot (sea-green)
        parts.append(
            f'<circle cx="{x_base:.1f}" cy="{y}" r="5" '
            f'fill="{_COLOR_AXIS}" opacity="0.55"/>'
        )
        parts.append(
            f'<circle cx="{x_rec:.1f}" cy="{y}" r="6" fill="{_COLOR_UP}"/>'
        )
        # Trailing recent percent on the far right
        parts.append(
            f'<text x="{plot_right + 8}" y="{y + 4}" font-size="11" '
            f'font-weight="700" fill="{_COLOR_TEXT}" text-anchor="start">'
            f'{rec}%</text>'
        )

    parts.append('</svg>')
    return "".join(parts)


def vote_tally_svg(
    yes: int,
    no: int,
    abstain: int,
    width: int = 480,
    height: int = 56,
    label: str | None = None,
) -> str:
    """Single horizontal stacked bar showing Yes/No/Abstain proportions.

    Used in the Resolution Spotlight to give the reader an immediate visual
    of how lopsided (or close) the vote was, without forcing them to do the
    arithmetic in their head.
    """
    total = max(1, yes + no + abstain)
    bar_y = 24 if label else 12
    bar_h = height - bar_y - 8

    y_w = max(2, round(yes / total * (width - 2)))
    n_w = max(2, round(no / total * (width - 2))) if no else 0
    a_w = max(0, (width - 2) - y_w - n_w)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="Yes {yes}, No {no}, Abstain {abstain}." '
        f'style="font-family:Georgia,\'Palatino Linotype\',serif;">'
    )
    if label:
        parts.append(
            f'<text x="0" y="14" font-size="11" fill="{_COLOR_AXIS}" '
            f'text-transform="uppercase">{_esc(label)}</text>'
        )

    # Yes (sea-green)
    parts.append(
        f'<rect x="1" y="{bar_y}" width="{y_w}" height="{bar_h}" fill="#2a9d57"/>'
    )
    if y_w > 30:
        parts.append(
            f'<text x="{1 + y_w / 2:.0f}" y="{bar_y + bar_h * 0.7:.0f}" '
            f'font-size="13" font-weight="600" fill="#ffffff" '
            f'text-anchor="middle">Yes {yes}</text>'
        )

    # No (ember-red)
    if n_w > 0:
        parts.append(
            f'<rect x="{1 + y_w}" y="{bar_y}" width="{n_w}" height="{bar_h}" '
            f'fill="#c64141"/>'
        )
        if n_w > 30:
            parts.append(
                f'<text x="{1 + y_w + n_w / 2:.0f}" y="{bar_y + bar_h * 0.7:.0f}" '
                f'font-size="13" font-weight="600" fill="#ffffff" '
                f'text-anchor="middle">No {no}</text>'
            )

    # Abstain (warm-amber)
    if a_w > 0:
        parts.append(
            f'<rect x="{1 + y_w + n_w}" y="{bar_y}" width="{a_w}" '
            f'height="{bar_h}" fill="#d38b2a"/>'
        )
        if a_w > 40:
            parts.append(
                f'<text x="{1 + y_w + n_w + a_w / 2:.0f}" '
                f'y="{bar_y + bar_h * 0.7:.0f}" font-size="13" font-weight="600" '
                f'fill="#ffffff" text-anchor="middle">Abstain {abstain}</text>'
            )

    parts.append('</svg>')
    return "".join(parts)


def p5_fingerprint_svg(
    country_name: str,
    p5_deltas: dict[str, float],
    width: int = 500,
    height: int = 160,
    bar_height: int = 16,
    bar_gap: int = 6,
) -> str:
    """Five-bar 'fingerprint' showing this country's signed Δ with each P5 member.

    Layout (left → right):
        [country code] [bar — left or right of zero line] [signed value]

    Bars use a symmetric x-axis around the zero line. Up-moves render in
    sea-green, down-moves in ember-red. Sized so the labels remain legible
    when the PNG is downscaled by Gmail / Outlook.
    """
    if not p5_deltas:
        return ""

    # Fixed P5 order so the chart reads consistently across countries.
    order = ["USA", "GBR", "FRA", "RUS", "CHN"]
    rows = [(code, p5_deltas.get(code)) for code in order if code in p5_deltas]
    if not rows:
        return ""

    # More generous spacing than v1 — every label has room.
    title_h = 28
    label_col_w = 58              # country codes
    value_col_w = 60              # signed numeric labels on the right
    plot_left = label_col_w + 12
    plot_right = width - value_col_w - 8
    plot_w = plot_right - plot_left
    zero_x = plot_left + plot_w / 2
    max_abs = max(1.0, max(abs(v) for _, v in rows if v is not None) * 100.0)
    scale = (plot_w / 2.0) / max_abs

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="P5 alignment fingerprint for {_esc(country_name)}" '
        f'style="font-family:Georgia,\'Palatino Linotype\',\'Times New Roman\',serif;">'
    )
    # Title (bigger, more breathing room)
    parts.append(
        f'<text x="0" y="14" font-size="12" font-weight="700" fill="{_COLOR_AXIS}" '
        f'letter-spacing="0.04em">'
        f'P5 fingerprint · Δ percentage points</text>'
    )

    # Zero baseline — slightly heavier than v1 so it reads as the spine.
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{title_h - 2}" '
        f'x2="{zero_x:.1f}" y2="{height - 6}" '
        f'stroke="{_COLOR_AXIS}" stroke-width="1"/>'
    )

    for i, (code, value) in enumerate(rows):
        y = title_h + i * (bar_height + bar_gap)
        if value is None:
            continue
        pts = value * 100.0
        if pts >= 0:
            x = zero_x
            w = max(2.0, pts * scale)
            fill = _COLOR_UP
        else:
            x = zero_x - max(2.0, abs(pts) * scale)
            w = max(2.0, abs(pts) * scale)
            fill = _COLOR_DOWN
        # Country code label — left column, right-aligned at plot_left - 8.
        parts.append(
            f'<text x="{plot_left - 8}" y="{y + bar_height * 0.72:.1f}" '
            f'font-size="13" font-weight="700" fill="{_COLOR_TEXT}" '
            f'text-anchor="end">{_esc(code)}</text>'
        )
        # Bar
        parts.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_height}" '
            f'fill="{fill}" rx="2" ry="2"/>'
        )
        # Value label — always in a fixed column on the right, never collides
        # with country codes or the bars themselves.
        sign = "+" if pts >= 0 else "−"  # use proper minus sign for typography
        parts.append(
            f'<text x="{plot_right + 8}" y="{y + bar_height * 0.72:.1f}" '
            f'font-size="12" font-weight="600" fill="{_COLOR_TEXT}" '
            f'text-anchor="start">{sign}{abs(pts):.0f}</text>'
        )

    parts.append('</svg>')
    return "".join(parts)


def alliance_arrow_svg(
    baseline_pct: int,
    recent_pct: int,
    width: int = 320,
    height: int = 36,
) -> str:
    """Small 'before → after' bar for the Strange Bedfellows section.

    Two segments side-by-side: baseline alignment in muted slate, recent
    alignment in sea-green (because this section only shows convergences).
    """
    base_w = max(8, int((baseline_pct / 100.0) * (width - 80)))
    rec_w = max(8, int((recent_pct / 100.0) * (width - 80)))

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="Alignment moved from {baseline_pct}% to {recent_pct}%." '
        f'style="font-family:Georgia,\'Palatino Linotype\',serif;">'
    )
    # Baseline bar
    parts.append(
        f'<rect x="0" y="6" width="{base_w}" height="10" '
        f'fill="{_COLOR_AXIS}" opacity="0.45" rx="2" ry="2"/>'
    )
    parts.append(
        f'<text x="{base_w + 4}" y="14" font-size="9" fill="{_COLOR_AXIS}">'
        f'{baseline_pct}% baseline</text>'
    )
    # Recent bar
    parts.append(
        f'<rect x="0" y="20" width="{rec_w}" height="10" '
        f'fill="{_COLOR_UP}" rx="2" ry="2"/>'
    )
    parts.append(
        f'<text x="{rec_w + 4}" y="28" font-size="9" font-weight="600" fill="{_COLOR_TEXT}">'
        f'{recent_pct}% recent</text>'
    )
    parts.append('</svg>')
    return "".join(parts)


def top_drifts_svg(
    drifts: list[dict],
    name_lookup: dict[str, str] | None = None,
    width: int = 640,
    bar_height: int = 22,
    bar_gap: int = 6,
    label_width: int = 230,
    right_margin: int = 60,
    top_margin: int = 36,
    bottom_margin: int = 36,
) -> str:
    """Return an SVG ``<svg>…</svg>`` string for a horizontal-bar drift chart.

    Each bar is the percentage-point delta. Positive (convergence) bars are
    green and extend right of the zero line; negative (divergence) bars are
    red and extend left. Bars are sorted from largest absolute delta at the
    top down. Country labels are full names where ``name_lookup`` provides them.
    """
    if not drifts:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="60" '
            f'role="img" aria-label="No drifts to display.">'
            f'<text x="{width // 2}" y="32" text-anchor="middle" font-family="{_FONT}" '
            f'fill="{_COLOR_AXIS}" font-size="13" font-style="italic">'
            f'No drifts to display.</text></svg>'
        )

    name_lookup = name_lookup or {}
    rows: list[tuple[str, float]] = []
    for d in drifts:
        a_name = name_lookup.get(d["country_a"], d["country_a"])
        b_name = name_lookup.get(d["country_b"], d["country_b"])
        # Em-dash (U+2014) renders reliably in CairoSVG's default font; the
        # original ↔ (U+2194) is missing from many fallback fonts and shows
        # as a tofu rectangle in email clients.
        label = f"{a_name} — {b_name}"
        if len(label) > 38:
            label = label[:36].rstrip() + "…"
        rows.append((label, float(d["delta"]) * 100.0))

    # Symmetric x-axis around 0 so both directions are readable.
    max_abs = max(abs(v) for _, v in rows)
    max_abs = max(10.0, max_abs * 1.05)

    height = top_margin + bottom_margin + len(rows) * (bar_height + bar_gap)
    plot_left = label_width
    plot_right = width - right_margin
    plot_w = plot_right - plot_left
    zero_x = plot_left + plot_w / 2.0
    scale = (plot_w / 2.0) / max_abs

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="Top alignment shifts in percentage points." '
        f'style="font-family:{_FONT};color:{_COLOR_TEXT};">'
    )
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')

    # Title
    parts.append(
        f'<text x="{plot_left}" y="20" font-size="13" font-weight="700" '
        f'fill="{_COLOR_TEXT}">Top alignment shifts (percentage points)</text>'
    )

    # Vertical gridlines + tick labels every 10 pts
    for tick in range(-int(max_abs // 10) * 10, int(max_abs // 10) * 10 + 1, 10):
        x = zero_x + tick * scale
        parts.append(
            f'<line x1="{x:.1f}" y1="{top_margin - 4}" '
            f'x2="{x:.1f}" y2="{height - bottom_margin + 4}" '
            f'stroke="{_COLOR_GRID}" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - bottom_margin + 18}" font-size="10" '
            f'fill="{_COLOR_AXIS}" text-anchor="middle">{tick:+d}</text>'
        )

    # Zero baseline
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{top_margin - 4}" '
        f'x2="{zero_x:.1f}" y2="{height - bottom_margin + 4}" '
        f'stroke="{_COLOR_AXIS}" stroke-width="1"/>'
    )

    # Bars. Value labels live OUTSIDE the bar at the zero-line side — for a
    # negative bar that's the bar's right edge (= zero), for a positive bar
    # the bar's left edge (= zero). This keeps every label in one tidy column
    # next to the y-axis and never lets them collide with the country names.
    for i, (label, value) in enumerate(rows):
        y = top_margin + i * (bar_height + bar_gap)
        bar_y = y
        if value >= 0:
            x = zero_x
            w = value * scale
            fill = _COLOR_UP
            # Outside-right of bar (away from zero).
            value_x = x + w + 6
            value_anchor = "start"
        else:
            x = zero_x + value * scale
            w = abs(value) * scale
            fill = _COLOR_DOWN
            # Outside-left of bar (away from zero) — but this would collide
            # with country names. Instead, anchor the value just OUTSIDE the
            # bar's RIGHT edge (which IS the zero line), text-anchor="start".
            value_x = zero_x + 6
            value_anchor = "start"
        # Country label — right-aligned in the left margin.
        parts.append(
            f'<text x="{plot_left - 8}" y="{bar_y + bar_height * 0.7:.1f}" '
            f'font-size="12" fill="{_COLOR_TEXT}" text-anchor="end">'
            f'{_esc(label)}</text>'
        )
        # Bar.
        parts.append(
            f'<rect x="{x:.1f}" y="{bar_y}" width="{w:.1f}" height="{bar_height}" '
            f'fill="{fill}" rx="3" ry="3"/>'
        )
        # Value label.
        sign = "+" if value >= 0 else ""
        parts.append(
            f'<text x="{value_x:.1f}" y="{bar_y + bar_height * 0.7:.1f}" '
            f'font-size="11" font-weight="600" fill="{_COLOR_TEXT}" '
            f'text-anchor="{value_anchor}">{sign}{value:.0f} pts</text>'
        )

    parts.append('</svg>')
    return "".join(parts)
