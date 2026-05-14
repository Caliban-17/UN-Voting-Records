# UN Voting Intelligence Platform — Implementation Summary

## Project Vision

> Treat UN nations as nodes; visually display alliances; calculate relative soft power over time; show which issues caused divergence.

All of the above is implemented and running on a self-contained Flask application at `http://localhost:5001`.

---

## Architecture

```
web_app.py              Flask app — API routes, rate limiting, concurrency
src/
  data_processing.py   CSV loading, vote encoding, preprocessing
  main.py              Similarity matrix, clustering, ML pipeline
  similarity_utils.py  Numerically safe cosine-similarity helper
  network_analysis.py  Graph build (NetworkX), centrality, communities
  soft_power.py        Composite influence scoring, trend tracking
  divergence_analysis.py  Anomaly detection, before/after comparison
  network_viz.py       Plotly figures (network, soft-power trends, etc.)
  model.py             Logistic regression vote predictor
  cache_utils.py       Thread-safe LRUCache, @cached_api decorator
  config.py            Env-var driven configuration
static/
  js/app.js            Vanilla JS — state, fetching, DOM rendering, jobs
  css/style.css        Design system, layout, responsive styles
templates/index.html   Single-page dashboard shell
tests/                 72 tests, 0 warnings
```

---

## Key Backend Modules

### `network_analysis.py`

- Country nodes, cosine-similarity edges
- Temporal exponential weighting (`0.95^Δyear`)
- Community detection: Louvain, Label Propagation
- 5 centrality metrics: PageRank, Betweenness, Eigenvector, Degree, Closeness

### `soft_power.py`

- Composite score: `0.4×PageRank + 0.3×Betweenness + 0.2×Eigenvector + 0.1×VoteSwaying`
- Historical trend computation (year-by-year)

### `divergence_analysis.py`

- Vote anomaly detection (specific votes that diverged from expected bloc behaviour)
- Issue-level similarity before/after a pivot period
- Full divergence report per cluster/year

### `similarity_utils.py`

- `compute_cosine_similarity_matrix()` — drops zero-norm rows before calling sklearn, eliminating all `RuntimeWarning: divide by zero` messages

### `cache_utils.py`

- Thread-safe `LRUCache` with optional TTL
- `@cached_api` decorator for route-level response caching keyed on path + JSON payload
- `model_registry` — stores trained models keyed by `{train_end}_{test_start}`

---

## API Surface

| Method | Route                                  | Purpose                                        |
| ------ | -------------------------------------- | ---------------------------------------------- |
| GET    | `/api/data/summary`                    | Dataset statistics                             |
| POST   | `/api/analysis/clustering`             | Agglomerative clustering + stability (ARI/NMI) |
| POST   | `/api/analysis/soft-power`             | Centrality scores                              |
| POST   | `/api/visualization/network`           | Plotly network graph                           |
| POST   | `/api/visualization/pca`               | 2D PCA projection                              |
| POST   | `/api/visualization/issue-timeline`    | Issue frequency over time                      |
| POST   | `/api/visualization/soft-power-trends` | Historical soft-power line chart               |
| POST   | `/api/analysis/compare`                | Pairwise country similarity + anomalies        |
| POST   | `/api/analysis/divergence-report`      | Cluster divergence report                      |
| POST   | `/api/prediction/train`                | Synchronous model training                     |
| POST   | `/api/prediction/predict`              | Vote prediction for an issue                   |
| GET    | `/api/prediction/issues`               | Deduplicated issue catalogue                   |
| POST   | `/api/jobs/train-model`                | Async training job (returns `job_id`)          |
| POST   | `/api/jobs/soft-power-trends`          | Async trends job                               |
| GET    | `/api/jobs/<job_id>`                   | Poll job status / retrieve result              |
| GET    | `/api/insights`                        | Automated data insights                        |
| GET    | `/api/methods`                         | Methods + caveats metadata                     |

---

## Security & Governance

- **CSP headers**: script/style/connect sources allowlisted
- **Rate limiting**: per-client sliding window (default 180 req/min)
- **Concurrency cap**: max 2 simultaneous heavy analyses (HTTP 429 if busy)
- **Input validation**: strict ISO-3 country codes with fuzzy suggestions; year bounds derived from data
- **DOM safety**: frontend renders all text via `textContent` (no `innerHTML` from user/server data)

---

## Analytical Value

| Feature                | What it computes                                   | Caveat surfaced to user                      |
| ---------------------- | -------------------------------------------------- | -------------------------------------------- |
| PCA projection         | Variance maximising 2D view                        | "projection, not causal ideology axes"       |
| Network centrality     | Structural position in similarity graph            | "not direct influence without further tests" |
| Cluster stability      | Bootstrapped ARI/NMI over resampled columns        | Shown alongside cluster output               |
| Prediction calibration | Brier score, reliability curve, majority baselines | Shown on training completion                 |
| Methods panel          | Vote encoding, algorithm names, caveats            | Always visible in UI                         |

---

## Test Coverage

```
pytest tests/ -v   →   72 passed, 0 warnings
```

Suites: `test_data_processing`, `test_main`, `test_network_analysis`,
`test_app_safeguards`, `test_enhancements`, `test_web_app`, `test_quick`.

---

## How to Run

```bash
# Local development
source venv/bin/activate
python web_app.py
# → http://localhost:5001

# Production (Gunicorn)
gunicorn -w 2 -b 0.0.0.0:5001 web_app:app
```

Configuration via `.env`:

```bash
TEMPORAL_DECAY_FACTOR=0.95
DEFAULT_SIMILARITY_THRESHOLD=0.7
PAGERANK_WEIGHT=0.4
MAX_CONCURRENT_ANALYSIS=2
MAX_API_REQUESTS_PER_MIN=180
CORS_ALLOWED_ORIGINS=http://localhost:5001
```
