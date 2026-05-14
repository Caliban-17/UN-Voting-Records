# Performance Optimization Guide

## CPU Usage Profile

**Phase 1: Initial Load** (one-time, 10–30 seconds)

- CSV parsing and cleaning: heavy CPU
- After: dataset held in memory for the lifetime of the process

**Phase 2: Analysis Requests** (per-request)

- Similarity matrix + clustering: 2–5 seconds
- Network graph build: 1–3 seconds
- Results cached by the LRU cache (TTL 5 min for API, 1 h for models)

**Phase 3: Subsequent Requests** (cache hit)

- Same parameters → near-instant response from LRU cache

**Phase 4: Idle**

- Flask worker: minimal CPU (~1–5%)

---

## Quick Optimizations

### Option 1: Limit Dataset Size (Fastest) ⭐

For development/testing, cap the rows loaded at startup:

```bash
# .env
MAX_ROWS_TO_LOAD=100000   # ~2-3 years of recent data
```

**Result**: 2–3 s load time (vs 30 s), 80% CPU reduction.

### Option 2: Sample Data File

```bash
cd data
head -100000 2025_03_31_ga_voting_corr1.csv > un_votes_sample.csv
```

Update `.env`:

```bash
UN_VOTING_DATA_PATH=un_votes_sample.csv
```

### Option 3: Narrow the Analysis Window

In the dashboard:

- Shorter year ranges (5 years instead of 20+)
- Higher similarity threshold (0.8 → fewer network edges)
- Avoid requesting very wide soft-power trend windows (uses background job)

---

## Recommended Settings by Use Case

### Development/Testing (Fastest)

```bash
MAX_ROWS_TO_LOAD=50000
MAX_WORKERS=1
CHUNK_SIZE=5000
```

**Load time**: 2–3 seconds

### Normal Use (Balanced)

```bash
MAX_ROWS_TO_LOAD=200000
MAX_WORKERS=2
CHUNK_SIZE=10000
```

**Load time**: 5–10 seconds

### Full Historical Analysis

```bash
MAX_ROWS_TO_LOAD=0   # no limit
MAX_WORKERS=2
CHUNK_SIZE=10000
```

**Load time**: 10–30 seconds (one-time, then cached)

---

## Concurrency Governance

The server enforces two resource guards:

| Setting                    | Default | Env var                     |
| -------------------------- | ------- | --------------------------- |
| Max concurrent analyses    | 2       | `MAX_CONCURRENT_ANALYSIS`   |
| Analysis slot wait timeout | 0.25 s  | `ANALYSIS_SLOT_TIMEOUT_SEC` |
| Rate limit (requests/min)  | 180     | `MAX_API_REQUESTS_PER_MIN`  |
| Background job workers     | 2       | `MAX_JOB_WORKERS`           |

Requests that exceed the concurrency cap receive HTTP 429 ("Server busy").
Heavy operations (model training, soft-power trends) run as background jobs
polled via `/api/jobs/<id>`.

---

## Monitoring

```bash
# Check process CPU
top -pid $(pgrep -f "web_app.py")
```

**Normal levels:**

- Initial load: 50–100% for 10–30 s
- Idle: 1–5%
- Analysis endpoint: 20–50% for 2–5 s

---

## Long-Term Optimizations

### Convert to Parquet (80% faster load)

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/2025_03_31_ga_voting_corr1.csv')
df.to_parquet('data/un_votes.parquet')
"
```

Update `config.py`:

```python
UN_VOTES_CSV_PATH = DATA_DIR / 'un_votes.parquet'
```

### Use a Database

Import into SQLite/PostgreSQL to query only the needed slice of rows
rather than loading the full dataset into RAM.

---

## Current Status

✅ LRU caching — API responses cached 5 min, models 1 h  
✅ Concurrency cap — max 2 simultaneous heavy analyses  
✅ Background jobs — training and trend endpoints are async  
⚠️ Full CSV still loaded on startup — use `MAX_ROWS_TO_LOAD` to reduce
