# ◆ OPEN-TERMINAL

A free, open-source financial and OSINT intelligence terminal in Python. Mirrors
the core Bloomberg workflows — equities, macro, maritime, aviation, news — using
only free APIs, public data and open standards.

No paid subscriptions. No API keys required to start.

```bash
pip install -r requirements.txt
python -m nltk.downloader -d .venv/share/nltk_data vader_lexicon
streamlit run app.py
```

The second line installs the sentiment lexicon. Skip it and everything still
runs — headlines just score `+0.00` across the board. See
[Headline sentiment](#headline-sentiment).

---

## What it does

| Module | Command | Capability |
|---|---|---|
| **Equity** | `AAPL EQUITY` | Candlesticks + EMA/RSI/MACD/Bollinger/ATR, as-reported financials from SEC XBRL, peer comparables, options chains, per-ticker news |
| **Macro** | `YCRV` | Full 11-tenor Treasury curve, inversion detection, composite recession score, FRED series browser, World Bank cross-country data |
| **Maritime** | `SUEZ SHIP` | AIS vessel positions, 7 chokepoint congestion monitors, MMSI/IMO lookup, custom area scans |
| **Aviation** | `EUROPE FLY` | Live ADS-B state vectors, fleet watchlists, aircraft tracks, airport throughput |
| **News** | `NEWS` | 21 RSS feeds + GDELT, sentiment scoring, trending-term extraction |

Command bar accepts Bloomberg-style syntax: `<SUBJECT> <FUNCTION>`. Type `HELP`
for the full reference.

---

## Data sources

Everything here is free. The credentials are all optional.

| Source | Key needed? | Notes |
|---|---|---|
| yfinance | No | Prices, fundamentals, options. Unofficial; 15-min delayed |
| SEC EDGAR | No | Filings + XBRL company facts. Requires a descriptive User-Agent |
| FRED | Optional | Keyless CSV endpoint is used automatically when no key is set |
| OpenSky | Optional | 400 credits/day anonymous, 4000 with free OAuth2 credentials |
| AISStream | **Yes** (free) | The only module that genuinely needs a key to function |
| GDELT | No | Global news index, 65+ languages |
| World Bank | No | Cross-country macro |

Copy `.env.example` to `.env` and fill in whichever you want:

```
FRED_API_KEY=...          # fredaccount.stlouisfed.org/apikeys
OPENSKY_CLIENT_ID=...     # opensky-network.org -> Account -> API Client
OPENSKY_CLIENT_SECRET=...
AISSTREAM_API_KEY=...     # aisstream.io
SEC_USER_AGENT=Open-Terminal/1.0 (research; you@example.com)
```

---

## Architecture

```
app.py                  Streamlit entry point, command parser, routing
config.py               Credentials, TTLs, chokepoints, feeds, watchlists
data_fetchers/
  equities.py           yfinance + SEC EDGAR XBRL + indicators
  macro.py              FRED (3-tier fallback) + World Bank + curve analysis
  maritime.py           AIS: local SDR -> aisstream -> scrapers
  aviation.py           OpenSky OAuth2 + state vectors
  news.py               RSS + GDELT + VADER/FinBERT sentiment
ui/
  terminal_theme.py     Bloomberg palette, CSS, Plotly template
  components.py         Tape, tiles, charts, news feed, tables
  maps.py               Folium + Plotly dark geo rendering
utils/
  cache.py              SQLite TTL cache + requests-cache sessions
  rate_limiter.py       Backoff, token buckets, circuit breakers
```

### Resilience

Free data sources fail constantly. Three layers handle it:

- **Caching** — SQLite TTL cache (daily bars 1h, scrapes 15m, fundamentals 24h).
  If an upstream dies and a stale entry exists, the stale value is served with
  its age rather than an error.
- **Backoff** — every request retries with exponential backoff and jitter,
  honouring `Retry-After`. Per-provider token buckets keep request rates well
  inside published limits.
- **Circuit breakers** — after 3 consecutive failures a provider is marked down
  and calls fail instantly instead of paying the retry budget. Without this, a
  FRED outage made the yield curve (11 tenors × 5 attempts × 20s) take eleven
  minutes to render nothing. Tripped breakers surface in the sidebar.

---

## Tests

```bash
pytest                # 186 offline tests, ~3s
pytest -m network     # 6 live tests against SEC EDGAR
pytest -m "" -q       # everything
```

Network tests are excluded by default. Free upstreams go down constantly
(OpenSky 503s regularly), and a suite that goes red because of someone else's
outage trains you to ignore failures.

| File | Covers |
|---|---|
| `test_indicators.py` | RSI/ATR against a textbook Wilder loop, EMA/MACD/Bollinger identities, empty and short-frame edges |
| `test_xbrl.py` | SEC period labelling, duration/form filtering, restatements, live accounting identities |
| `test_resilience.py` | Circuit breaker states and fail-fast timing, retry/backoff, token bucket, cache TTL and stale-on-error |
| `test_domain.py` | Command parser, ticker normalisation, AIS sentinels and MMSI flags, yield-curve analysis, sentiment |

The two indicator/XBRL files are **regression suites, not coverage padding**.
Both bugs they guard were invisible in the UI — the chart drew a plausible
oscillator, the statement table rendered real numbers under wrong headings.
Both were verified by mutation: reintroducing the original bug makes the
relevant suite fail.

If you change `rsi()`, `atr()` or `_extract_xbrl_series()`, run these first.

---

## Notes from building this

Things that were verified rather than assumed, and are easy to get wrong:

**SEC XBRL period labelling.** A fact's `fy`/`fp` fields identify *the report the
fact appeared in*, not the period the number describes. Apple's FY2023 revenue
appears in the FY2023, FY2024 and FY2025 10-Ks. Keying on `fy` and keeping the
newest filing shifts an entire income statement by two years — it reported
$383.285B as FY2025 revenue when the real figure is $416.161B. Periods are
therefore derived from each fact's own `end` date, with duration filtering to
separate annual from quarterly tags. Validated against the accounting
identities: gross profit ties, and the balance sheet balances.

**XBRL tag migration.** Filers switch tags mid-history, so candidate tags must
be *merged per period*, not tried until one hits. Microsoft reported cost of
revenue under `CostOfRevenue` through FY2017 and `CostOfGoodsAndServicesSold`
from FY2020; Apple used `PaymentsOfDividendsCommonStock` in FY2016-17 and
`PaymentsOfDividends` from FY2020. Returning on the first tag with any data
produced a decade-old series with every recent column blank — the line looked
simply unavailable. Merging gives MSFT a continuous 14-year cost line.

**Some blanks are correct.** Apple stopped disclosing goodwill separately after
FY2017 and interest expense after FY2023; no tag carries them. Microsoft
publishes no aggregate D&A tag at all — only `Depreciation` and
`AmortizationOfIntangibleAssets` separately. Those are deliberately *not*
substituted: a depreciation-only figure under a "Depreciation & Amortization"
heading is an understated number that looks authoritative, which is worse than
a blank.

**Tickers can outlive their registrant.** XOM currently resolves to
"ExxonMobil Holdings Corp" (CIK 2115436), a post-reorganisation entity that has
filed no 10-K and carries no `us-gaap` taxonomy — the financial history sits
under the predecessor CIK. Rather than render an empty table, this case is
detected and reported with the entity name and a pointer to the Yahoo source.

**RSI seeding.** Wilder's RSI starts from a *simple* mean of the first 14
changes, then smooths recursively. `ewm(adjust=False)` implements the recursion
correctly but seeds from the first single observation, which biases the series
for dozens of bars — on Wilder's own published example that gives 50.7 instead
of 70.5, enough to flip an overbought reading to neutral. `_wilder_average()`
plants the correct seed; output matches a textbook loop to ~1e-14. ATR had the
same bug.

**FRED and User-Agent.** `fredgraph.csv` accepts the TLS connection and then
never responds if you send a Chrome User-Agent, but answers in 0.3s with a plain
one. Conversely SEC and BLS return 403 to a browser UA and require a descriptive
contact string. User-Agent is selected per host.

**OpenSky auth.** Basic auth is deprecated; the API moved to OAuth2
client-credentials. Anonymous access still works at a quarter of the quota and
10-second resolution instead of 5.

**Free-data limits worth knowing.** Prices are delayed. AIS coverage is thin
outside major lanes. OpenSky's receiver network is sparse over oceans and much
of the global south — an absent aircraft is not evidence it isn't flying.
Headline sentiment measures the tone of the writing, not market impact
("beats estimates, shares fall" scores positive). None of this is investment
advice.

---

## On scraping

Open-Terminal prefers official free APIs over scraping wherever one exists —
SEC's XBRL API rather than parsing 10-K HTML, FRED's CSV endpoint rather than
the website, aisstream rather than MarineTraffic.

Playwright scrapers for MarineTraffic and VesselFinder are implemented but
**ship disabled**. Both sites prohibit automated collection in their Terms of
Service and both sit behind Cloudflare, so those paths are brittle by design and
will break without warning. Enabling them (`OPENTERM_ALLOW_SCRAPERS=1`) is a
decision you make deliberately.

For maritime data the genuinely better options are a free aisstream.io key, or a
~$25 RTL-SDR dongle running `rtl_ais` — your own antenna, no rate limits, no
terms of service, sub-second latency.

The aviation watchlists ship institutional airframes only: national carriers,
cargo fleets, government transports from public registries. Aircraft positions
are broadcast unencrypted and are legally receivable, but sustained tracking of
a *named private individual's* aircraft is treated very differently in law
across jurisdictions and is restricted by OpenSky's own terms. Nothing stops you
adding entries; think about that one first.

---

## Optional extras

```bash
playwright install chromium              # only if enabling scrapers
pip install transformers torch           # FinBERT instead of VADER (~3GB)
pip install openbb                       # auto-detected if present (~1GB)
```

## Headline sentiment

The NEWS module scores headlines with VADER, whose lexicon ships as data
rather than code. NLTK will fetch it on first use into `~/nltk_data`, so on
most machines nothing is needed. Two things make that worth pinning down:

**The failure is silent.** A missing lexicon does not raise — every headline
scores a flat `+0.00`, which looks like a market with no opinion rather than a
broken install. If the whole tape reads neutral, this is why.

**NLTK 3.10 validates data paths.** It ships a `pathsec` layer that refuses
any data file resolving outside an allowed root. Where the OS redirects the
home directory — sandboxed, containerised or packaged-app environments — the
lexicon lands in `~/nltk_data` but *resolves* somewhere else, and NLTK raises
`PermissionError`. The app catches it and reports the misleading "resource not
found, try re-downloading it" message; re-downloading writes to the same
redirected path and fails again.

Installing into a directory already on `nltk.data.path` sidesteps both:

```bash
python -m nltk.downloader -d .venv/share/nltk_data vader_lexicon
```

It lives inside the virtualenv, so a rebuilt venv needs it again. Verify with:

```bash
python -c "from nltk.sentiment.vader import SentimentIntensityAnalyzer as S; print(len(S().lexicon), 'entries')"
```

7502 entries means it loaded. A traceback means it did not.

Swapping VADER for FinBERT (see [Optional extras](#optional-extras)) removes
the dependency entirely — the model ships through `transformers`, not as
downloaded corpus data.

## Environment flags

| Flag | Effect |
|---|---|
| `OPENTERM_DEBUG=1` | Full tracebacks in the UI, debug logging |
| `OPENTERM_OFFLINE=1` | Render from cache only, zero network calls |
| `OPENTERM_ALLOW_SCRAPERS=1` | Enable the ToS-restricted scraper fallbacks |

## Requirements

Python 3.11+ (developed and tested on 3.14).

## License

MIT. Data belongs to its respective providers — check their terms before
redistributing anything you pull.
