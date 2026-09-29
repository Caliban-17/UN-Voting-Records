# Automated political research

## Editorial contract

The newsletter combines a concise current-affairs digest with findings calculated from
voting data. Its research addresses two questions: the world through the voting record, and the UN as an institution. Official
news is a separate context stream. Source excerpts never become causal explanations
for quantitative changes merely because they mention the same country or subject.

Every finding carries its period, observed figures, denominator, method and caveat.
The first research edition covers agreement with the US, China and Russia, minority
opposition, agenda composition and abstention. These measurements describe the General
Assembly's recorded votes; they cannot by themselves measure influence, motives,
institutional effectiveness, Security Council behaviour, or the whole UN system.

## Comparison design

- Use current year-to-date when it contains at least 20 recorded votes and the same
  calendar dates across the preceding three years supply at least 20 baseline votes.
  Otherwise use the latest past calendar year with at least 20 recorded votes and eligible
  years in the preceding three-year window. A closed calendar year is not a guarantee
  of source completeness: first/last dates and counts are shown explicitly.
- Compare alignment on the same country panel. Agreement excludes abstentions and
  absences; abstentions have their own measure. Missing participation can still change
  denominators. The baseline pools observations across its years.
- Report broad-topic-standardised changes alongside raw alignment changes. This holds
  baseline topic weights constant on shared themes. It does not hold resolution text,
  within-topic composition or attendance constant and is not a causal adjustment.
- Treat the largest agenda shift as exploratory. Neither confidence intervals nor
  statistical significance are claimed for these descriptive population summaries.
- Group exact duplicate country/resolution rows before calculating. Source corrections
  can change a published result and create a new edition.
- Flag voting data more than 60 days old. Collection time, article publication time,
  research publication time and voting coverage are separate concepts.

## Automatic publication

The app's background worker collects immediately and subsequently every 30 minutes.
`PULSE_AUTO_REFRESH=0` disables it in controlled environments. `PULSE_REFRESH_SECONDS`
sets the interval (minimum five minutes); `PULSE_DATA_DIR` changes the output directory.
Run the app under a persistent process supervisor or Docker for unattended operation.
The app checks the complete voting dataset daily, safely promotes verified updates,
and reloads changes automatically. Failed checks retry after one hour; source outcomes
are stored in `data/pulse/votes-refresh.json`. The scheduled
GitHub workflow always downloads the newest data release before
analysis, so its published research follows refreshed source data automatically.

`scripts/refresh_pulse.py --require-research` does one complete pass and exports a
self-contained newsletter in HTML, Markdown, text and canonical JSON, alongside the
research briefing and source diagnostics. `/newsletter` serves the exact saved edition.
`/newsletter/feed.xml` provides stable edition identifiers and links to immutable archives.
`.github/workflows/publish-research.yml` runs it every six hours and after the existing
data-refresh workflow succeeds. The `newsletter-current` release has current artifacts;
each changed editorial hash gets an immutable `newsletter-<hash>` release. Poll timestamps
and health changes do not trigger a new edition. Source corrections do. No mail, Substack draft or human
approval gate is required. This is downloadable publication, not public website hosting.
The interactive site is served by the existing Flask deployment.

Locally, research archives are under `data/pulse/research/<hash>.json`, and the app serves
them at `/briefing/editions/<hash>`. The research RSS keeps stable GUIDs and original
dates, publishing only changed research. Daily source snapshots are retained for 90
days; the rolling source window holds 30 days and the page shows the last seven days.

## Failure behaviour

- Fetch feeds concurrently with timeouts, bounded size and retries.
- Reject malformed feeds, unsafe article URLs, undated items and future-dated items.
- Parse source HTML to plain text; escape it at rendering. No remote scripts run.
- Preserve last verified items through a source failure. Show failure, last successful
  collection and latest article dates. A feed with no items newer than 30 days is quiet,
  even when HTTP succeeds. A collection older than two hours is stale.
- Use an operating-system writer lock and atomic JSON replacement. Concurrent readers
  see a complete prior or new snapshot. Failures are retried on the next worker pass.
- If all sources fail, the CLI saves diagnostic state and exits non-zero; the hosted
  workflow does not replace the public release with a failed collection.
- No research is manufactured when there are insufficient data. A research-required CI
  run fails rather than publishing an empty edition.

## Sources and boundaries

Source URLs and areas of coverage are in `src/pulse.py`. The feeds cover public UN
reporting, headquarters meetings, Geneva briefings and health. WHO's feed may carry old
items; the source-health panel exposes this. There is no access to private negotiations.
Issue dossiers group related articles, not proven accounts of one event; exact matching
headlines retain all source links. Keyword topic labels are navigational aids.

The live tests confirmed working HTTP/XML responses from all four sources. Upstream
freshness is assessed separately. New adapters for budgets, mandates, treaty actions or
other institutional records should bring explicit schema, date and coverage contracts
before generating research claims from them.


## Faster decision coverage

The official public e-deleGATE GA resolutions registers provide adoption dates, symbols,
titles and aggregate Yes/No/Abstain totals ahead of the batch country-voting extracts.
The worker polls the current and preceding regular sessions every 30 minutes; hosted
publication polls every six hours. Country-level CSV refresh runs daily at 04:00 UTC.
Unrecognised action text, impossible totals, future dates, empty responses and unexpected
register shrinkage fail that source and preserve its previous verified records.

`data/pulse/decisions.json` is a separate decision ledger. It includes resolutions adopted
without a vote and never converts their adoption into 193 invented Yes votes. Aggregate
recorded results also never create country rows. Full roll calls must arrive through the
validated voting-data pipeline before entering alignment research.

The 30-day newsletter includes a bounded selection of official headlines, the GA decision
ledger for that period, an institutional adoption indicator, and three research findings.
Its sources and coverage section distinguishes old upstream data from collection failure.
WHO currently returns an older feed and is labelled quiet; this is partial UN-system
coverage, not a claim to monitor every body. The live iGov country-vote endpoints returned
empty results during the September 2026 investigation and are not used as a data source.

As verified on 29 September 2026, the registers contain 12 resolutions after the country
CSV's 28 July cutoff: seven recorded results and five adopted without a vote. The latest
register adoption is 17 September 2026. These are source coverage dates, not a guarantee
that no later UN activity occurred.
