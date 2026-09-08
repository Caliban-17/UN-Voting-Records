# CLAUDE.md — UN-Scrupulous / UN Alignment Atlas

Guide for AI assistants working in this repository. It describes what is
actually here; when the code and this file disagree, trust the code and fix
this file.

## What this is

Two products on one dataset of UN General Assembly roll-call votes
(1946 → present, ~950k rows from the UN Digital Library):

1. **A Flask web app** ("What is the UN voting on, and who stands where?") —
   a single-page site that opens on **The Big Picture** (six whole-record
   readings: agenda, divisions, a power and the world, Washington–Beijing
   scatter, landmark votes, the annual Cuba vote), then country profiles, an
   alignment map, coalition builder, drift feed, network/bloc views, and the
   older clustering/PCA/prediction tools under "Tools".
2. **A weekly newsletter pipeline** ("Weekly Atlas") — composes an edition
   from the data, emails it to a Substack inbox on a schedule, and only
   publishes when the underlying data has actually moved.

**Stack**: Python 3.12 (CI/Docker; local venv may be 3.13), Flask, pandas,
scikit-learn, networkx, Plotly (server-side figures and the browser), a
vanilla-JS frontend, GitHub Actions for data refresh + publishing.

There is **no Streamlit app**. `src/app.py` does not exist.

## Repository map

```
web_app.py                 Entry point: create_app() + load_data() at import
app/
  __init__.py              Application factory, CORS, blueprint registration
  middleware.py            Per-endpoint rate limit, analysis-slot semaphore, CSP/security headers
  services.py              Global DataFrame, name/year helpers, caches, background job queue
  routes/
    core.py                "/" , /health, /api/data/*, /api/country/*, /api/countries,
                           /api/drift*, /api/newsletter/*, /api/events, /api/coalition,
                           /api/methods, /api/insights, /api/report
    analysis.py            /api/analysis/{clustering,soft-power,abstention,pivotality,
                           bloc-timeline,compare,divergence-report}
    visualization.py       /api/visualization/{network,pca,issue-timeline,soft-power-trends}
    prediction.py          /api/prediction/{train,predict,issues}
    jobs.py                /api/jobs/{<id>,train-model,soft-power-trends,network-animation}
    story.py               /api/story/{agenda,division,alignment,scatter,landmarks,
                           resolution/<rcid>/map,recurring/<key>,country/<code>,
                           this-week,calendar,emergency,lenses}  (whole-record, cached)
src/                       Analysis + newsletter library (no Flask imports here)
  config.py                Paths, VOTE_ENCODING, column maps, env-driven settings
  story_analysis.py        The Big Picture: themes, division, alignment, scatter, landmarks
  session_calendar.py      The Assembly's year: phases, the computed opening date (rule 1),
                           typical dates of recurring votes, votes just landed
  lenses.py                Through which lens: explained variance by partition, cohesion,
                           agenda shares, norm cascades, per-lens indices, eras
  lenses_partitions.py     Curated year-aware partitions: alliance camps, income tiers
                           (World Bank CSV in data/), feminist-foreign-policy cohort
  regional_groups.py       UN regional groups by ISO-3 (+ historical states, lineages)
  data_processing.py       CSV/parquet loading, schema normalisation, vote matrix, entropy
  country_display.py       ISO-3 -> editorial names ("Iran", "North Korea") + title_case_name()
  country_profile.py       Allies/opponents/P5 alignment for one country
  drift_analysis.py        Pairwise alignment drifts (the "What Changed" feed + newsletter lead)
  coalition.py             Coalition builder for a topic
  divergence_analysis.py   Where two countries split, by resolution
  main.py                  Vote matrix -> cosine similarity -> agglomerative clustering
  similarity_utils.py      Numerically stable cosine similarity
  cluster_naming.py        Auto-names clusters
  network_analysis.py / network_viz.py   Voting network + Plotly renderings
  soft_power.py            PageRank/betweenness/eigenvector composite
  pivotality_analysis.py   Who lands on the prevailing side of divided votes
  abstention_analysis.py   Abstention rates by country and topic
  sankey_analysis.py       Bloc-membership timeline
  model.py                 Random-forest vote predictor
  data_fetcher_github.py   Recorded votes from DGACM's GitHub extracts (the refresh source)
  data_fetcher_marc.py     UN DL MARC-XML fetcher — behind a WAF bot challenge since 2026-09; opt-in
  data_fetcher.py          Older Playwright scraper; only its merge/dedup logic is still used
  cache_utils.py           LRUCache, cached_api decorator, model registry
  newsletter.py            Edition composer (NewsletterEdition, content_hash, pick_recent_year,
                           edition_to_dict / edition_from_dict)
  newsletter_render.py     Markdown / email-safe HTML / plain-text renderers (+ country-code glossary)
  newsletter_chart.py      Inline SVG charts for the newsletter
  newsletter_voice.py      Headline/subhead/prose templates (deterministic per edition)
  newsletter_email.py      MIME message builder (deterministic Message-ID, PNG chart via cairosvg)
  newsletter_ledger.py     Committed ledger of published editions (data/published_ledger.json)
  newsletter_archive.py    Disk archive of composed editions (data/editions/, gitignored)
static/js/app.js           The whole frontend (single file, no build step)
static/css/style.css       Styles; Okabe-Ito colour tokens (--ok / --warn / --bad)
static/vendor/             Self-hosted Plotly 2.27, axios 1.20 and the world topojson
templates/index.html       The one page; Plotly + axios served from static/vendor/
scripts/refresh_data.py    Fetch new votes and merge them into the CSV (see "Data")
tests/                     pytest suite (see "Testing")
data/                      CSV + parquet cache (gitignored), known_events.json and
                           published_ledger.json (committed)
docs/                      Older planning/summary docs; may be stale
.github/workflows/         tests, refresh-data, publish-newsletter, validate-events
```

## Runtime architecture

- `web_app.py` builds the app with `create_app()` and calls
  `app.services.load_data()` **at import time**, so the DataFrame exists before
  the first request under both the dev server and Gunicorn. Data load is the
  slow step (a 360 MB CSV; a parquet cache beside it makes reloads fast).
- `app/services.py` owns `df_global` plus helpers every route uses:
  `get_df()`, `get_year_bounds()`, `data_freshness()`, `country_names()`
  (ISO-3 → display name, cached per process), `normalize_country_code()`,
  `validate_year_range()`, the background job store, and LRU registries for
  expensive payloads.
- `app/middleware.py`: a per-(client, method, path) sliding-window rate limit
  on `/api/*`, a `with_analysis_slot` semaphore for heavy analyses (returns
  429 when busy), and a strict Content-Security-Policy. If you add a new CDN
  or inline script, update the CSP or it will be silently blocked.
- Routes follow one shape: validate input → `make_error(msg, 400)` on bad
  input → call a `src/` function → `jsonify`. Unexpected exceptions go through
  `make_server_error()` which logs the traceback and returns a generic 500.
- The frontend keeps a single `state` object, fetches `/api/countries` once
  at startup, and decodes ISO-3 codes with `nameFor(code)`. All Plotly traces
  take colours from the `PALETTE` constant (colour-blind safe); do not
  introduce green/red.
- Tabs load lazily: `runAnalysis()` loads only the open tab, `loadDashboard()`
  runs the clustering/PCA/timeline/insights set once per window, and the Big
  Picture (`loadBigPicture()` and the `renderStory*` functions at the end of
  app.js) fetches its six endpoints in parallel. Every story card follows one
  shape — kicker, a **finding** computed from the numbers, caption, chart,
  the server's `takeaway`, its `caveat` — so findings never drift from data.
- Navigation is grouped (The story · Countries · Blocs · Signals · Tools);
  the year window drives the country/bloc/signal views, never the Big Picture.

## Newsletter pipeline (read before touching it)

Composition lives in `src/newsletter.py` (`build_newsletter_edition`). The
edition carries a deterministic `content_hash`; the publish gate compares it
with the committed ledger and skips the email when nothing changed.

Contracts that regression tests enforce
(`tests/test_publish_workflow_consistency.py`, `tests/test_newsletter_*.py`):

- **Compose once, reuse.** The workflow's Send step must load the archived
  edition via `edition_from_dict` and must never call
  `build_newsletter_edition` again. Recomposing let the emailed edition drift
  from the gated one and re-sent the same issue weekly.
- **`recent_year` is auto-picked**, never `df["year"].max()`.
  `pick_recent_year` skips a sparse in-progress year
  (`MIN_RECENT_RESOLUTIONS = 20`), so early in a session the edition anchors
  on the last complete year and stays stable until the new year qualifies.
  Editions therefore freeze during the off-season; that is by design.
- **Resolve the newest `data-*` release by `publishedAt`, never
  `createdAt`.** A release's `createdAt` is the date of the commit it points
  at; weekly releases cut from an unchanged `main` all tie, and the old
  `sort_by(.createdAt) | last` returned the oldest of them.
- Rendering is separate from hashing: glossary, names and prose changes in
  `newsletter_render.py` do not affect `content_hash`.
- `NewsletterEdition.big_picture` ("The bigger picture" section) is whole-record
  context computed from `story_analysis`; it is deliberately outside
  `content_hash` and is optional, so old archives load with `{}`.
- `NewsletterEdition.calendar` ("The week ahead") comes from `session_calendar`
  and is forward-looking, so it stays outside `content_hash`. The ledger stores a
  numeric `big_picture` snapshot with each published edition; the next edition for
  the same focus and year reports each stat as up/down/unchanged since it.
- `NewsletterEdition.this_week` ("This Week in the Assembly") holds the recorded
  votes of the latest fortnight *in the data*. In season its rcids are added to
  `content_hash` so a week with new votes always publishes; off-season the key
  is absent and every hash is unchanged.
- Topic phrases keep initialisms upper-case (`_ACRONYMS` in
  `newsletter_voice.py`: HIV/AIDS, UNRWA, …); add there, not to `_PROPER_NOUNS`.

Schedule: `refresh-data.yml` Monday 22:00 UTC pulls new votes via MARC-XML
and, if rows changed, publishes a GitHub Release `data-YYYY-MM-DD` with the
full CSV. `publish-newsletter.yml` Tuesday 09:00 UTC downloads the newest
release, refuses data older than 90 days, composes the global edition plus a
curated per-country matrix (`COUNTRY_EDITIONS`), gates on the ledger, emails
via SMTP, then commits the ledger back to `main` as `atlas-bot`.

Consequence: **`main` moves without you.** Pull before pushing; ledger commits
land most Tuesdays in season.

## Data

- Source CSV: `data/2025_03_31_ga_voting_corr1.csv` (override with
  `UN_VOTING_DATA_PATH` in `.env`). Gitignored. Locally it is chmod 444 and
  `scripts/refresh_data.py` never overwrites it without `--promote`.
- Source columns: `undl_id, ms_code, ms_name, ms_vote, date, session, title,
  subjects, resolution, agenda_title, undl_link` (plus the modern aliases
  `rcid/country_code/vote` on merged files). `_apply_post_load_transformations`
  in `data_processing.py` normalises both schemas.
- Processed columns used everywhere: `rcid, country_identifier (ISO-3),
  country_name, vote, date, year, issue, primary_topic`.
- `VOTE_ENCODING`: Y=1, N=-1, A=0, X/blank=None (not voting / absent).
- Country names: `country_name` is title-cased with
  `country_display.title_case_name` (keeps "and/of/the" lowercase, fixes
  `People'S`). Editorial short names come from `COUNTRY_DISPLAY_NAMES`
  overrides; add an entry there when a long form reads badly.
- `data/known_events.json` annotates time-series charts; a workflow
  validates its shape on every change.
- The parquet cache rebuilds itself when a required column is missing;
  delete it to force a rebuild after schema changes.

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python web_app.py                # http://localhost:5001  (PORT env overrides)
gunicorn -w 2 -b 0.0.0.0:5001 web_app:app   # production shape (gunicorn not pinned)
docker compose up -d             # http://localhost:8080
```

The desktop-app preview config (`.claude/launch.json`, name `un-atlas`) runs
`venv/bin/python web_app.py` on port 5001.

### Testing

```bash
pytest                    # pytest.ini adds -m "not slow"
pytest -m slow            # model-training / long-iteration tests
flake8 app src web_app.py scripts   # CI lint gate; keep it at zero
```

- `tests/conftest.py` skips the data-dependent Flask tests automatically when
  the real CSV is absent (CI never has it). Everything else runs on synthetic
  fixtures.
- `tests/test_data_fetcher_marc.py` and `tests/test_data_fetcher_merge.py`
  hit the network and are excluded in CI.
- `tests/test_publish_workflow_consistency.py` parses the workflow YAML and
  needs `pyyaml` (declared in requirements.txt).

### Conventions

- Configuration and magic numbers live in `src/config.py`; env vars are read
  there, not in routes.
- Module-level `logger = logging.getLogger(__name__)`; log before and after
  expensive steps. Never `print` in library code.
- `src/` functions return `None` / empty frames on failure and log the
  reason; routes translate that into 4xx/5xx JSON.
- Type hints and docstrings on public functions. PEP 8, 88-column lines
  (`.flake8` ignores E501/E203).
- Commit messages: conventional prefixes (`feat(ui):`, `fix(newsletter):`,
  `ci:`, `chore(ledger):`), present tense, explain the *why*.
- Tests go beside the code they cover: `tests/test_<module>.py`.

## Adding things

- **New analysis**: function in `src/<topic>_analysis.py` (pure pandas, no
  Flask) → tests → route in the matching blueprint under `app/routes/` → a
  renderer in `static/js/app.js` and a card in `templates/index.html`. Use
  `nameFor()` for any country code shown to a reader, `resolveCountryCode()`
  for inputs (they accept a name or a code, backed by the `countryOptions`
  datalist), and `xaxis/yaxis automargin` on Plotly layouts.
- **New newsletter section**: extend `NewsletterEdition` and
  `edition_to_dict` / `edition_from_dict` together (round-trip test in
  `tests/test_newsletter_roundtrip.py`), then all three renderers.
- **New workflow step that reads data**: copy the "Fetch UN voting CSV from
  latest data release" step verbatim (publishedAt sort, filename
  normalisation).

## Definitions the Big Picture relies on

- Recorded (roll-call) votes only; consensus adoptions are absent by construction.
- "Taking a side" = voting Yes or No. Agreement = share of shared side-takings
  that matched. Divided vote = winning side under 90% of Yes+No; contested =
  under two-thirds (the Charter test, rarely failed, reported not charted).
- Isolated vote = a country took a side with at most two other members.
- Lineages: RUS continues SUN; DEU continues GER; CZE continues CSK; SRB
  continues YUG/SCG (`regional_groups.LINEAGE`).
- Landmark votes are looked up by resolution symbol; tallies come from the
  data and are checked against the historical record in
  `tests/test_story_analysis.py`.

## Gotchas

- The Digital Library answers scripts with `x-amzn-waf-action: challenge` (HTTP
  202, empty body; 403 for headless browsers). Do not build a challenge bypass:
  robots.txt disallows `/search`. The refresh uses DGACM's GitHub extracts;
  weekly in-season currency needs the library's authenticated API.

- The app loads the whole CSV into memory per process; two Gunicorn workers
  mean two copies.
- `numpy` 1.26 built from source on Python 3.13 emits spurious
  `RuntimeWarning: divide by zero encountered in matmul` during PCA. The
  output is finite; use 3.12 locally to silence it.
- Plotly, axios and the map topojson are vendored under `static/vendor/`; the
  CSP allows `script-src 'self'` only, so a CDN script tag needs a CSP change.
- `/` renders `index.html` with `?v=<hash>` on app.js and style.css; edit the
  files and reload, no manual cache-busting needed.
- `docs/*.md` predate the Flask rewrite in places; treat them as history.
