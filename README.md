# UN Alignment Atlas

> **Who does this country vote with at the UN, and how has that shifted?**

A country-first dashboard for the UN General Assembly voting record. Pick a country, see its top five allies and top five opponents in any window, watch how its alignment with each of the P5 (US, UK, France, Russia, China) has moved year over year, and pull up the resolutions where it broke ranks with its closest partner.

Deep-dive tabs (network graph, soft-power rankings, bloc-membership sankey, vote predictor) sit behind the headline view as supporting tools.

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment config
cp .env.example .env              # edit as needed

# 4. Run
python web_app.py                 # → http://localhost:5001
```

Place your UN voting CSV at `data/2025_03_31_ga_voting_corr1.csv` (or configure `UN_VOTING_DATA_PATH` in `.env`).

### Docker

```bash
docker-compose up -d
# → http://localhost:8080
```

---

## Features

### 🏁 Country Profile (default view)

- Top 5 allies and top 5 opponents in the selected window, each tagged with a **percentile chip** (`p92` = "this alignment is higher than 92% of all country pairs this window") so every number carries its own baseline
- **Bloc-alignment strip** — three headline percentages (Western / Eastern / Non-aligned) make two profiles immediately comparable at a glance
- Per-year alignment with the P5 — one chart that tells the story of the country's bloc drift
- **This country's biggest shifts** — drift cards for the most recent year vs the rest of the window, each tagged with the *named topics* where disagreement rose most
- **Biggest splits with top ally** — surfaces the resolutions where the country and its closest partner voted opposite
- Click "Compare →" on any ally/opponent or drift card to jump to the Compare tab pre-filled
- Shareable via `?country=XXX&start=YYYY&end=YYYY` in the URL hash

### ⚡ What Changed (drift feed)

Newsroom-ready cards showing the world's biggest alignment shifts:

- *"NIC ↔ PAN: 96% → 50% (−46 pts). Driven by: Armed conflicts prevention, Middle East situation."*
- Configurable recent year + baseline window + direction filter (convergence / divergence / both)
- Topic attribution surfaces the **delta in disagreement rate per topic** — only "newly diverged" topics show up, so chronic disagreements don't pollute the cards
- Filter by country (`?country=USA`) to see the world from one country's vantage point
- **Digest panel** turns the top-5 shifts into a 3-paragraph press-release-style summary (`/api/drift/digest`) — copy-paste-ready for daily briefings, Slack posts, or email digests

### 📰 Weekly Atlas — Newsletter Composer

Reproducible, deterministic, **LLM-free** UN-voting newsletter, ready to paste into Substack, email, or Slack:

- **Headline + lede** — the single biggest realignment of the period, with named topic drivers
- **By the Numbers** — 5 punchy stat cards, each with a comparison context (not standalone)
- **The Shift** — narrative paragraph with top-5 supporting drifts
- **Top Movers** — countries ranked by aggregate |Δ| with the P5, each with a one-sentence story
- **Coalition Watch** — for each watched topic (`nuclear`, `palestine`, `climate`, `human rights`, `sanctions` by default), a tier-jumper table showing which countries flipped between baseline and recent
- **Resolution Spotlight** — smallest-margin recent vote with named breakaways
- **Quiet Convergences** — under-the-radar pairs moving *closer* in a period dominated by divergence
- **Methodology** — short, honest, every-edition

Endpoint `GET /api/newsletter/weekly?recent_year=&baseline_window=&topics=&country=&format=` emits:
- `format=json` — full structured payload + embedded markdown + stable `email_subject` + filesystem-safe `edition_slug`
- `format=markdown` — copy-paste into Substack / Beehiiv / Notion
- `format=html` — self-contained inline-style HTML; paste into Gmail/Mailchimp
- `format=text` — plain-text MIME fallback (wrapped to 78 columns)

**Per-country editions:** add `?country=ARG` and the composer scopes drifts to that country, re-frames the byline (*"Country edition — Argentina (ARG)"*), and prefixes the email subject with the ISO-3 code.

**Edition archive:** every composition can be persisted to `data/editions/{year}/{slug}.{md,html,json,txt}`:

| Endpoint | What it does |
|---|---|
| `POST /api/newsletter/archive?…` | Compose + write all four formats. Idempotent on slug. |
| `GET /api/newsletter/archive` | List archived editions, newest first. |
| `GET /api/newsletter/archive/{year}/{slug}.{md\|html\|json\|txt}` | Retrieve one. Slugs are filesystem-safe; path-traversal is rejected. |
| `GET /api/newsletter/retrospective?year=YYYY` | Aggregate top-drift pair counts across the archive — the "drift of the year" retrospective. |

The archive root defaults to `data/editions/` and can be overridden with `ATLAS_ARCHIVE_DIR`. Editions are gitignored — they're reproducible from the source CSV.

**Weekly publish flow** ([.github/workflows/publish-newsletter.yml](.github/workflows/publish-newsletter.yml)):

Substack deprecated publish-via-email for new publications in 2024, so the workflow takes the editor-in-the-loop approach:

1. **Tuesday 09:00 UTC** the cron fires.
2. The composer pulls the latest CSV from your GH Release backend.
3. It composes the edition; the content-hash gate skips publishing if nothing changed since last week.
4. If there's new news, it emails **you** at `dominicbramwell@gmail.com` with subject *"📰 Ready to publish: UN-Scrupulous №N"*.
5. The email has the rendered HTML + a `.md` file attached.
6. You open Substack → Create → New post → paste the Markdown → eyeball → hit **Publish**. **30 seconds.**
7. Subscribers only see it once you click Publish — full editorial control, full human-in-the-loop.

This is also the design Bloomberg / Reuters / Politico use for their "morning briefs" — automated compose, human approve.

For the older spec (when Substack's publish-via-email worked), the workflow path below is still relevant:

1. Pulls the UN voting CSV from your data backend (plug in S3 / GH Release / HF dataset)
2. Composes the edition with the same Python the API uses
3. Saves all four formats to `data/editions/{year}/{slug}.*`
4. Ships to one of four configurable destinations:

| destination | secrets needed |
|---|---|
| `commit` | (none — uses `GITHUB_TOKEN`) |
| `email` / `substack-inbox` | `ATLAS_SMTP_HOST`, `ATLAS_SMTP_PORT`, `ATLAS_SMTP_USER`, `ATLAS_SMTP_PASS`, `ATLAS_SMTP_TO`, `ATLAS_SMTP_FROM` |
| `beehiiv` | `ATLAS_BEEHIIV_API_KEY`, `ATLAS_BEEHIIV_PUBLICATION_ID` |
| `buttondown` | `ATLAS_BUTTONDOWN_API_KEY` |

Sample output (real data, 2024 edition):

> **Argentina–Turkmenistan alignment fell 57 points**
>
> The most consequential UN voting realignment of 2024 was between Argentina and Turkmenistan. Their alignment fell from 93% in 2021–2023 to 36% this year — a 57-point swing on Territories occupied by Israel–settlement policy and Self-determination of peoples.
>
> **Coalition Watch — "Palestine"**: Argentina jumped tiers from **Champion supporter (+0.93)** to **Champion opposed (−1.00)**.
>
> **Quiet Convergence**: Argentina ↔ Israel: 40% → 82% (+42 pts).

### 🤝 Coalition Builder

*If a fresh resolution about [topic] were tabled today — who lobbies whom?*

Type a topic ("nuclear", "palestine", "climate", "sanctions", "human rights"), get back:

- **Predicted Yes / No / Abstain tally** across all 193 member states
- **Seven tiers** from "Champion supporter" to "Champion opposed", with "Fence-sitter" surfaced as the lobbying-target list
- Per-country detail: mean lean, n-votes on the topic, Y/N/A breakdown, abstain rate
- Sample of matched resolutions so you can spot-check the topic query

### 🏷 Auto-named clusters

Clustering output replaces opaque IDs ("Cluster 3") with labels:

> *"Israel-aligned bloc — decolonization"* · 23 members
> *"Russia-led group"* · 1 member
> *"Bangladesh-aligned bloc — nuclear disarmament"* · 47 members

Each cluster is named by its most central member (highest mean intra-cluster similarity) and its **signature topic** — the topic where this cluster's mean vote diverges most sharply from the rest of the world.

### 📅 Event annotations

The P5 alignment time-series chart overlays hand-curated events from [data/known_events.json](data/known_events.json) (Russia invades Ukraine 2022, Berlin Wall 1989, COVID 2020, etc.) so alignment drops get a named cause. Edit the file by hand — a GitHub Action ([.github/workflows/validate-events.yml](.github/workflows/validate-events.yml)) validates the schema on PR.

### 🌐 Network Graph

- Force-directed graph of country voting similarity
- Configurable similarity threshold and layout algorithm
- Louvain community detection (voting blocs colour-coded)
- Toggle node labels on/off

### 💪 Soft Power Rankings

- Composite score: `0.4×PageRank + 0.3×Betweenness + 0.2×Eigenvector + 0.1×VoteSwaying`
- Historical yearly trend charts
- Heavy computation offloaded to background job with progress polling

### 🔀 Divergence & Comparison

- Pairwise country similarity with anomaly breakdown
- Cluster divergence report over a configurable time window
- ISO-3 country code validation with fuzzy suggestions

### 📊 PCA Projection

- 2D variance projection of the country × vote matrix
- Labelled as "projection, not causal ideology axes"

### 🤖 Vote Prediction

- Logistic regression trained on a configurable train/test split
- Brier score, reliability curve, and majority-vote baselines shown after training
- Training runs as a background job; results cached by split key

### 📋 Methods Panel

- Every API response includes a `meta` object with vote encoding, algorithm names, and analytical caveats
- Dedicated `/api/methods` endpoint

---

## Project Structure

```
UN Voting Records/
├── web_app.py              Flask application (routes, governance, jobs)
├── src/
│   ├── config.py           Env-var driven configuration
│   ├── data_processing.py  CSV loading + vote encoding
│   ├── main.py             Similarity, clustering, ML pipeline
│   ├── similarity_utils.py Numerically safe cosine similarity
│   ├── network_analysis.py NetworkX graph + centrality
│   ├── soft_power.py       Influence scoring + trend tracking
│   ├── divergence_analysis.py Anomaly + divergence reports
│   ├── network_viz.py      Plotly figure generators
│   ├── model.py            Vote predictor (logistic regression)
│   └── cache_utils.py      Thread-safe LRU cache + @cached_api
├── static/
│   ├── js/app.js           Dashboard JS (state, rendering, job polling)
│   └── css/style.css       Design system
├── templates/index.html    Single-page dashboard
├── tests/                  72 tests (pytest), 0 warnings
├── docs/                   Performance + implementation guides
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Publishing cadence — what's realistic

UN General Assembly voting is **highly seasonal**:

| Period | Activity |
|---|---|
| **Sept – Dec** | Regular session — 5–20+ recorded votes per week. Heavy news cycle. |
| **Jan – Aug** | Mostly off-session. Some weeks have zero new recorded votes. August is the quietest month. |

**Recommended cadence:**
- **Sept – Dec**: refresh + publish weekly (Mon 22:00 UTC refresh, Tue 09:00 UTC publish). This is what the workflows are set up to do by default.
- **Jan – Aug**: refresh weekly, but **let the dedup gate decide whether to publish**. If no new votes have landed since last edition, the content-hash gate will skip the SMTP step and log a warning. Result: zero duplicate editions across the off-season.

**The dedup gate** (`.github/workflows/publish-newsletter.yml`, `Skip-if-unchanged gate` step):
1. After composing the edition, computes its `content_hash` — a SHA-256 of the drift list, coalition tier-jumpers, and spotlight rcid. The hash is **deterministic**: same data + same recent_year + same window → identical hash.
2. Walks the archive backwards, finds the most recent edition with the same `country_focus`.
3. If the hashes match, the publish step is skipped with a `::warning::` annotation. The archive is still written so we can see "we composed an identical edition this week" in the history.

This means you can leave the weekly cron running year-round without manual intervention — it'll publish only when something actually changed.

## Keeping data fresh

The platform expects a UN voting CSV at `data/2025_03_31_ga_voting_corr1.csv` (or wherever `UN_VOTING_DATA_PATH` points). The CSV covers 1946 → the export date.

**Check freshness:** `GET /api/data/freshness` returns `{loaded, records, min_year, max_year, latest_vote_date, days_since_latest_vote, is_stale}`. On startup, the server logs a `WARNING` if the latest vote is more than 60 days old.

**Refresh locally:**

```bash
# Default: pulls everything since the CSV's latest vote date, writes the
# merged result to a date-stamped sibling file. SOURCE CSV IS NEVER TOUCHED.
python scripts/refresh_data.py

# Override the window (e.g. 90 days)
python scripts/refresh_data.py --days 90

# Dry-run — fetch + merge but skip the write
python scripts/refresh_data.py --dry-run

# Atomically swap in the new file as the source (DANGEROUS)
python scripts/refresh_data.py --promote
```

**How it works:** the fetcher uses UN Digital Library's MARC-XML API
(`?of=xm` on the search endpoint). Each search response includes the
full voting data for 50 records, so backfilling 9 months takes ~10
minutes via plain HTTP. No Playwright, no Chromium, no browser.

**Schema-aware merge:** the new rows use the historical
`undl_id/ms_code/ms_vote` columns plus the modern `rcid/country_code/vote`
aliases. The dedup key is `(undl_id, ms_code)`, which is populated in
both sources. There is a hard safety gate that **refuses to overwrite the
source if the merged frame is smaller than the input** — this is the
bug that destroyed 880k rows in an earlier draft, now caught by 4
regression tests.

**Refresh in CI:** [.github/workflows/refresh-data.yml](.github/workflows/refresh-data.yml) is wired as a `workflow_dispatch` template. To run it on a schedule, plug in your data-storage backend (S3 / GCS / HF dataset / Git LFS — the raw CSV is gitignored) at the marked spots and uncomment the cron line. The shape of the workflow is correct; only the download/upload steps need your storage credentials.

**Cache rebuild:** the parquet cache (`data/*.parquet`) auto-rebuilds whenever it's missing a required column. Delete it by hand to force a full rebuild on any schema change.

---

## Configuration

Key `.env` settings:

```bash
# Data
UN_VOTING_DATA_PATH=data/2025_03_31_ga_voting_corr1.csv
MAX_ROWS_TO_LOAD=0                  # 0 = no limit; set e.g. 100000 for dev

# Analysis
DEFAULT_SIMILARITY_THRESHOLD=0.7
TEMPORAL_DECAY_FACTOR=0.95
PAGERANK_WEIGHT=0.4
BETWEENNESS_WEIGHT=0.3
EIGENVECTOR_WEIGHT=0.2
VOTE_SWAYING_WEIGHT=0.1

# Governance
MAX_CONCURRENT_ANALYSIS=2          # max simultaneous heavy requests
MAX_API_REQUESTS_PER_MIN=180
CORS_ALLOWED_ORIGINS=*             # comma-separated for production
```

---

## Temporal Weighting

```
weight = 0.95^(current_year − vote_year)
```

| Year    | Weight |
| ------- | ------ |
| current | 1.00   |
| −1      | 0.95   |
| −5      | 0.77   |
| −10     | 0.60   |
| −30     | 0.21   |

---

## Vote Encoding

| Vote    | Value                             |
| ------- | --------------------------------- |
| Yes     | +1                                |
| Abstain | 0                                 |
| No      | −1                                |
| Missing | excluded from pairwise comparison |

---

## API Overview

| Route                                  | Method   | Notes                                       |
| -------------------------------------- | -------- | ------------------------------------------- |
| `/api/country/<code>/profile`          | GET      | Allies, opponents, P5 alignment, bloc strip, drift, splits |
| `/api/drift`                           | GET      | Top alignment shifts with named topic drivers |
| `/api/drift/digest`                    | GET      | 3-paragraph narrative summary of the top drifts |
| `/api/coalition`                       | GET      | Coalition Builder — predicted tally + tiered country list |
| `/api/newsletter/weekly`               | GET      | Weekly Atlas edition — JSON / Markdown / HTML / text     |
| `/api/newsletter/archive`              | GET/POST | List or write archived editions                          |
| `/api/newsletter/archive/<year>/<slug>.<fmt>` | GET | Retrieve one archived edition                            |
| `/api/newsletter/retrospective`        | GET      | Drift-of-the-year aggregation across the archive         |
| `/api/events`                          | GET      | Hand-curated event annotations for time-series overlays |
| `/api/data/summary`                    | GET      | Dataset statistics                          |
| `/api/data/freshness`                  | GET      | Records, year span, days since latest vote  |
| `/api/analysis/clustering`             | POST     | Clusters + bootstrapped stability (ARI/NMI) |
| `/api/analysis/soft-power`             | POST     | Centrality scores                           |
| `/api/visualization/network`           | POST     | Plotly JSON                                 |
| `/api/visualization/pca`               | POST     | Plotly JSON                                 |
| `/api/visualization/issue-timeline`    | POST     | Plotly JSON                                 |
| `/api/visualization/soft-power-trends` | POST     | Plotly JSON                                 |
| `/api/analysis/compare`                | POST     | Pairwise similarity + anomalies             |
| `/api/analysis/divergence-report`      | POST     | Cluster divergence                          |
| `/api/prediction/train`                | POST     | Sync training                               |
| `/api/prediction/predict`              | POST     | Predict vote distribution                   |
| `/api/prediction/issues`               | GET      | Issue catalogue                             |
| `/api/jobs/train-model`                | POST     | Async training → `job_id`                   |
| `/api/jobs/soft-power-trends`          | POST     | Async trends → `job_id`                     |
| `/api/jobs/<job_id>`                   | GET      | Poll job or retrieve result                 |
| `/api/insights`                        | GET/POST | Automated dataset insights                  |
| `/api/methods`                         | GET      | Methods + caveats metadata                  |

---

## Testing

```bash
pytest tests/ -v                        # run all 72 tests
pytest tests/ -v --cov=src              # with coverage
pytest tests/test_network_analysis.py   # single module
```

---

## Performance Tips

- Set `MAX_ROWS_TO_LOAD=100000` for 80% load-time reduction during development
- Narrow the year range in the dashboard to reduce matrix size
- Convert the CSV to Parquet for ~5–7 s load time vs 30 s for the full file

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for details.

---

## Key Analytical Caveats

- **PCA plots** are variance projections, not causal ideological axes
- **Network centrality** measures structural position in a similarity graph — it is not direct causal influence without additional tests (e.g., pivotality)
- **Abstentions** are encoded as 0 (neutral), but can reflect strategic non-commitment; treat divergence scores accordingly
- **Agenda composition** shifts year-to-year, which can affect apparent divergence independently of relationship change

---

## License

MIT
