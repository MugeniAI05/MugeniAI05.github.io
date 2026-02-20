---
layout: post
title: Building a Production Quantitative Volatility Pipeline for SPY Options Research
image: "/posts/vol_analysis.png"
tags: [Quantitative Finance, ETL Pipeline, Time Series, GARCH, Options Pricing, Python, SQLite]
---

Our goal was to build a production-grade end-to-end quantitative data pipeline for SPY, the world's most liquid ETF, capable of ingesting raw market data, enforcing automated data quality standards, and producing statistically rigorous volatility analysis suitable for a quant research team.

The full source code, including the ETL, validation, analysis, scheduler, and alert system, is available on GitHub: **[spy-volatility-pipeline](https://github.com/MugeniAI05/spy-volatility-pipeline)**

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth / Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Stage 1 — ETL Pipeline](#etl-pipeline)
    - [Database Schema](#db-schema)
    - [Retry Wrapper](#retry-wrapper)
    - [Price History Ingestion](#price-ingestion)
    - [Options Chain Ingestion](#options-ingestion)
    - [ETL Log Output](#etl-output)
- [03. Stage 2 — Data Validation & Quality Checks](#validation)
    - [Check 1 — Price Spike Detection](#check-spikes)
    - [Check 2 — Zero / Negative Prices](#check-zero)
    - [Check 3 — OHLC Consistency](#check-ohlc)
    - [Check 4 — Missing Date Gaps](#check-gaps)
    - [Check 5 — Bid-Ask Inversions & Wide Spreads](#check-bidask)
    - [Check 6 — Implied Volatility Outliers](#check-iv)
    - [Check 7 — Sparse Expirations](#check-sparse)
    - [Check 8 — Data Freshness](#check-freshness)
    - [Validation Summary Report](#validation-summary)
- [04. Stage 3 — Time Series & Volatility Analysis](#analysis)
    - [Data Loading & Log Returns](#data-loading)
    - [Rolling Realized Volatility](#rvol)
    - [GARCH(1,1) Model](#garch)
    - [Mean-Reversion & ADF Testing](#adf)
    - [Implied Volatility Surface](#iv-surface)
    - [Volatility Risk Premium](#vrp)
- [05. Results Summary](#results-summary)
- [06. Growth & Next Steps](#growth-next-steps)

---

# Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Quantitative research and risk management teams at exchanges and asset managers depend on reliable, clean, and analytically-rich financial data pipelines. Raw market data from any vendor, paid or free, arrives with gaps, outliers, structural inconsistencies, and quality issues that must be detected and handled before any model can be trusted.

This project builds a three-stage production pipeline around SPY, the SPDR S&P 500 ETF Trust. SPY was chosen deliberately: it is the world's most liquid equity instrument, with deep and continuous options markets, making it an ideal subject for demonstrating end-to-end quant infrastructure. The pipeline covers 2 years of daily OHLCV price history and 5 near-term options expiration dates.

The project is structured as a proper Python package — not notebooks. All logic lives in importable modules under `src/`, with a master orchestrator (`run_pipeline.py`), a daily scheduler (`scheduler.py`), a centralised config (`config.py`), and an email alert system (`src/notifier.py`). The pipeline runs automatically every weekday at 4:15 PM local time, 15 minutes after market close.

### Actions <a name="overview-actions"></a>

We first needed to compile the necessary financial data from Yahoo Finance via the `yfinance` library, storing it in a well-designed SQLite database with four tables: `price_history`, `options_chain`, `etl_log`, and `analysis_results`. The pipeline is broken into three modular stages that mirror real quant infrastructure:

* **Stage 1 — ETL Pipeline**: Data ingestion from Yahoo Finance → SQLite with retry logic, upserts, and full audit logging
* **Stage 2 — Data Validation**: 8 automated quality checks covering price spikes, OHLC integrity, bid-ask inversions, IV outliers, data freshness, and more
* **Stage 3 — Volatility Analysis**: Rolling realized vol (10/21/63d), GARCH(1,1) fitting, ADF mean-reversion testing, implied volatility surface construction, and Volatility Risk Premium estimation

Every run writes results to the `analysis_results` table, building a time series of all computed metrics so you can track how vol, GARCH parameters, and VRP evolve over time. The alert system emails you when any stage crashes, validation finds anomalies, data goes stale, or VRP exceeds a configurable threshold.

### Results <a name="overview-results"></a>

The pipeline has been running live since February 17, 2026. Below are results from the most recent run on **February 20, 2026**, which completed in **3.7 seconds**.

**ETL Ingestion:**
* 502 daily price rows loaded for SPY (2-year history)
* 969 options contracts loaded across 5 expiration dates (2026-02-20 through 2026-02-26)
* 0 failed ingestion steps; all data upserted cleanly with audit trail

**Data Validation:**
* 70 total anomalies flagged across 8 checks
* 4 price spikes detected — all confirmed real market events (April 2025 tariff shock)
* 61 wide bid-ask spreads — all concentrated in deep OTM near-expiry contracts, expected
* 5 IV outliers — deep OTM contracts filtered before analysis
* 0 structural data errors (OHLC violations, inversions, date gaps)
* Data confirmed fresh (0 days stale as of run date)

**Volatility Analysis:**

| Metric | Value |
|---|---|
| 10-day Realized Vol | 13.91% |
| 21-day Realized Vol | 11.66% |
| 63-day Realized Vol | 11.28% |
| GARCH Long-run Vol | 14.86% |
| GARCH Current Conditional Vol | 12.67% |
| GARCH Persistence (α + β) | 0.9318 |
| Near-term ATM IV | 49.63% |
| VRP (ATM IV − 21d RVol) | +37.97% (rich options) |

### Growth / Next Steps <a name="overview-growth"></a>

While the pipeline is fully functional and produces statistically meaningful outputs, several natural extensions would take this to production-grade infrastructure:

From an engineering standpoint, the Yahoo Finance source would be replaced with a professional vendor (Bloomberg, Refinitiv), ETL runs would be scheduled via Apache Airflow, and SQLite would be replaced by Postgres or Snowflake for concurrent access and scale.

From a modelling standpoint, EGARCH or GJR-GARCH could be added to capture the leverage effect (vol rises asymmetrically on negative shocks). The IV surface could be fitted with a parametric model (SVI or Heston) for cleaner interpolation. A Greeks computation layer (delta, gamma, vega, theta) and a backtested VRP-harvesting strategy would complete the research toolkit.

---

# Data Overview <a name="data-overview"></a>

All data is stored in a local SQLite database (`quant.db`). The schema was designed to support idempotent upserts — re-running the ETL never creates duplicate rows. A fourth table, `analysis_results`, records every run's computed metrics as a time series.

After the ETL stage, we have the following data available for analysis:

| **Variable** | **Source Table** | **Description** |
|---|---|---|
| date | price_history | Trading date (YYYY-MM-DD), unique per ticker |
| open / high / low / close | price_history | OHLC prices (unadjusted) |
| adj_close | price_history | Dividend-adjusted close — used for return calculations |
| volume | price_history | Daily share volume |
| expiration | options_chain | Option expiry date |
| option_type | options_chain | 'call' or 'put' |
| strike | options_chain | Strike price |
| bid / ask | options_chain | Market quotes at ingestion time |
| implied_vol | options_chain | Annualised Black-Scholes implied volatility |
| open_interest | options_chain | Outstanding contracts |
| in_the_money | options_chain | Boolean 0/1 flag |
| rvol_10d / rvol_21d / rvol_63d | analysis_results | Rolling realized vol per run |
| garch_persistence / garch_longrun_vol | analysis_results | GARCH parameters per run |
| atm_iv / vrp | analysis_results | IV and VRP per run |

---

# Stage 1 — ETL Pipeline <a name="etl-pipeline"></a>

The ETL lives in [`src/etl.py`](https://github.com/MugeniAI05/spy-volatility-pipeline/blob/main/src/etl.py). It is invoked by the master orchestrator `run_pipeline.py` or can be run standalone. The code is broken into four key sections:

* Database Schema & Setup
* Retry Wrapper
* Price History Ingestion
* Options Chain Ingestion

### Database Schema <a name="db-schema"></a>

All four tables are created on first run if they do not already exist. The `UNIQUE` constraints on `price_history` and `options_chain` ensure that `INSERT OR REPLACE` upserts work cleanly on re-runs, making the pipeline idempotent. The `analysis_results` table accumulates one row per run, building a historical record of all computed metrics.

```python

# config.py — central configuration, the only file you need to edit
DB_PATH          = "quant.db"
TICKER           = "SPY"
PRICE_PERIOD     = "2y"
PRICE_INTERVAL   = "1d"
MAX_EXPIRATIONS  = 5
RETRY_ATTEMPTS   = 3
RETRY_DELAY      = 5              # seconds between retries
SCHEDULE_TIME    = "16:15"        # 15 min after market close, Mon–Fri

def create_tables(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS price_history (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker      TEXT    NOT NULL,
        date        TEXT    NOT NULL,
        open        REAL, high REAL, low REAL, close REAL, adj_close REAL,
        volume      INTEGER,
        ingested_at TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, date)
    );

    CREATE TABLE IF NOT EXISTS options_chain (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker        TEXT NOT NULL,
        expiration    TEXT NOT NULL,
        option_type   TEXT NOT NULL,
        strike        REAL NOT NULL,
        last_price    REAL, bid REAL, ask REAL,
        volume        INTEGER, open_interest INTEGER,
        implied_vol   REAL, in_the_money INTEGER,
        ingested_at   TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, expiration, option_type, strike)
    );

    CREATE TABLE IF NOT EXISTS etl_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_time TEXT DEFAULT (datetime('now')),
        ticker TEXT, step TEXT, status TEXT, rows_loaded INTEGER, message TEXT
    );

    CREATE TABLE IF NOT EXISTS analysis_results (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        run_date            TEXT DEFAULT (datetime('now')),
        ticker              TEXT,
        rvol_10d            REAL, rvol_21d REAL, rvol_63d REAL,
        garch_omega         REAL, garch_alpha REAL, garch_beta REAL,
        garch_persistence   REAL, garch_longrun_vol REAL, garch_current_vol REAL,
        atm_iv              REAL, vrp REAL, ou_halflife REAL
    );
    """

```

### Retry Wrapper <a name="retry-wrapper"></a>

All API calls are wrapped in a generic retry function. This is a production pattern that guards against rate limits and transient network failures — critical when relying on any external data vendor, free or paid.

```python

def retry_fetch(fn, *args, label="fetch", **kwargs):
    """Call fn(*args, **kwargs) up to RETRY_ATTEMPTS times on failure."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            log.warning(f"[{label}] Attempt {attempt}/{RETRY_ATTEMPTS} failed: {exc}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"[{label}] All {RETRY_ATTEMPTS} attempts failed.")

```

### Price History Ingestion <a name="price-ingestion"></a>

Price data is pulled using `yfinance`, normalised to lowercase column names, forward-filled for any missing OHLC values, and upserted into SQLite row-by-row.

```python

def ingest_price_history(engine, ticker=TICKER):
    log.info(f"Ingesting price history for {ticker} ({PRICE_PERIOD})...")

    def _fetch():
        t = yf.Ticker(ticker)
        return t.history(period=PRICE_PERIOD, interval=PRICE_INTERVAL, auto_adjust=False)

    try:
        df = retry_fetch(_fetch, label="price_history")
    except RuntimeError as e:
        log_etl_event(engine, ticker, "price_history", "FAILED", message=str(e))
        return pd.DataFrame()

    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["ticker"] = ticker
    df["date"]   = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    df[["open", "high", "low", "close", "adj_close"]] = (
        df[["open", "high", "low", "close", "adj_close"]].ffill().bfill()
    )
    df["volume"] = df["volume"].fillna(0).astype(int)

    rows_loaded = 0
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("INSERT OR REPLACE INTO price_history "
                     "(ticker, date, open, high, low, close, adj_close, volume) "
                     "VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume)"),
                row.to_dict(),
            )
            rows_loaded += 1
        conn.commit()

    log.info(f"  Loaded {rows_loaded} price rows for {ticker}.")
    log_etl_event(engine, ticker, "price_history", "OK", rows=rows_loaded)
    return df

```

### Options Chain Ingestion <a name="options-ingestion"></a>

Five near-term expiration dates are pulled. For each expiration, both calls and puts are ingested, tagged with `option_type`, and upserted. Column names from yfinance's camelCase convention are normalised to snake_case.

```python

def ingest_options_chain(engine, ticker=TICKER):
    log.info(f"Ingesting options chain for {ticker}...")

    t           = retry_fetch(lambda: yf.Ticker(ticker), label="options_ticker")
    expirations = t.options[:MAX_EXPIRATIONS]
    log.info(f"  Pulling {len(expirations)} expiration dates: {list(expirations)}")

    all_frames = []
    for exp in expirations:
        try:
            chain = retry_fetch(t.option_chain, exp, label=f"chain_{exp}")
            for opt_type, frame in [("call", chain.calls), ("put", chain.puts)]:
                frame = frame.copy()
                frame["ticker"]      = ticker
                frame["expiration"]  = exp
                frame["option_type"] = opt_type
                all_frames.append(frame)
        except Exception as e:
            log.warning(f"  Skipping expiration {exp}: {e}")

    df = pd.concat(all_frames, ignore_index=True)
    df = df.rename(columns={
        "contractSymbol":   "contract_symbol",
        "lastPrice":        "last_price",
        "openInterest":     "open_interest",
        "impliedVolatility":"implied_vol",
        "inTheMoney":       "in_the_money",
    })
    # ... fill missing, upsert rows

```

### ETL Log Output <a name="etl-output"></a>

```
2026-02-20 18:05:00  INFO      ======================================================================
2026-02-20 18:05:00  INFO        SPY VOLATILITY PIPELINE — 2026-02-20 18:05:00
2026-02-20 18:05:00  INFO      ======================================================================
2026-02-20 18:05:00  INFO      ▶  STAGE 1 / 3: ETL
2026-02-20 18:05:00  INFO      Tables verified / created.
2026-02-20 18:05:00  INFO      Ingesting price history for SPY (2y)...
2026-02-20 18:05:00  INFO        Loaded 502 price rows for SPY.
2026-02-20 18:05:00  INFO      Ingesting options chain for SPY...
2026-02-20 18:05:00  INFO        Pulling 5 expiration dates:
                                 ['2026-02-20', '2026-02-23', '2026-02-24', '2026-02-25', '2026-02-26']
2026-02-20 18:05:01  INFO        Loaded 969 option rows for SPY.
2026-02-20 18:05:01  INFO      ETL COMPLETE — Price rows: 502, Option rows: 969
```

---

# Stage 2 — Data Validation & Quality Checks <a name="validation"></a>

The validation stage lives in [`src/validate.py`](https://github.com/MugeniAI05/spy-volatility-pipeline/blob/main/src/validate.py). Every ingestion run is followed by 8 automated quality checks. The checks are designed to catch *real* data problems in production, not hypothetical ones.

**Thresholds used (all configurable in `config.py`):**

| Parameter | Value | Rationale |
|---|---|---|
| `ZSCORE_SPIKE_THRESHOLD` | 4.0 | Daily return beyond 4σ is statistically extreme |
| `IQR_MULTIPLIER` | 3.0 | Conservative outlier fence for price levels |
| `MAX_BID_ASK_SPREAD_PCT` | 0.50 | Flag if (ask−bid)/mid > 50% |
| `MAX_IV_THRESHOLD` | 5.0 | IV > 500% is almost certainly garbage data |
| `MIN_OPTION_ROWS_PER_EXPIRY` | 5 | Flag expirations with suspiciously few strikes |

### Check 1 — Price Spike Detection (Z-Score) <a name="check-spikes"></a>

Daily log returns are z-scored against the full sample mean and standard deviation. Any return with |z| > 4.0 is flagged as an anomaly.

```python

def check_price_spikes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date")
    df["daily_return"] = df["close"].pct_change()
    mean_ret = df["daily_return"].mean()
    std_ret  = df["daily_return"].std()
    df["zscore"] = (df["daily_return"] - mean_ret) / std_ret

    spikes = df[df["zscore"].abs() > ZSCORE_SPIKE_THRESHOLD].copy()
    spikes["anomaly_type"] = "price_spike"
    spikes["detail"] = spikes.apply(
        lambda r: f"Return={r['daily_return']:.2%}, Z={r['zscore']:.2f}", axis=1
    )
    log.info(f"[CHECK 1] Price spikes (|z| > {ZSCORE_SPIKE_THRESHOLD}): {len(spikes)} found")
    return spikes[["date", "close", "daily_return", "zscore", "anomaly_type", "detail"]]

```

This check found **4 spikes**, all confirmed as real market events clustering around April 3–10, 2025 — the Trump tariff announcement and subsequent whipsaw:

| Date | Close | Return | Z-Score | Context |
|---|---|---|---|---|
| 2025-04-03 | $536.03 | −4.93% | −4.84 | Tariff announcement |
| 2025-04-04 | $507.68 | −5.85% | −5.74 | Selloff continuation |
| 2025-04-09 | $560.98 | +10.50% | +10.11 | Tariff pause announced |
| 2025-04-10 | $536.84 | −4.38% | −4.31 | Reversal |

The pipeline correctly identifies these as statistically extreme without falsely flagging the surrounding elevated-but-normal volatility period.

### Check 2 — Zero / Negative Prices <a name="check-zero"></a>

```python

def check_zero_negative_prices(df: pd.DataFrame) -> pd.DataFrame:
    bad = df[
        (df["close"] <= 0) | (df["open"] <= 0) |
        (df["high"]  <= 0) | (df["low"]  <= 0)
    ].copy()
    bad["anomaly_type"] = "zero_or_negative_price"
    bad["detail"]       = "One or more OHLC fields <= 0"
    log.info(f"[CHECK 2] Zero/negative prices: {len(bad)} found")
    return bad[["date", "close", "anomaly_type", "detail"]]

```

Result: **0 found** ✓

### Check 3 — OHLC Consistency <a name="check-ohlc"></a>

Financial price data must satisfy ordering constraints: High ≥ Low, High ≥ Open, High ≥ Close, Low ≤ Open, Low ≤ Close. Violations indicate a data vendor error or processing bug.

```python

def check_ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
    bad = df[
        (df["high"] < df["low"])   | (df["high"] < df["open"]) |
        (df["high"] < df["close"]) | (df["low"]  > df["open"]) |
        (df["low"]  > df["close"])
    ].copy()
    bad["anomaly_type"] = "ohlc_inconsistency"
    bad["detail"]       = "OHLC relationship violated"
    log.info(f"[CHECK 3] OHLC inconsistencies: {len(bad)} found")
    return bad[["date", "open", "high", "low", "close", "anomaly_type", "detail"]]

```

Result: **0 found** ✓

### Check 4 — Missing Date Gaps <a name="check-gaps"></a>

Consecutive trading dates are compared. Normal gaps are 3 calendar days (weekend) or 4 days (weekend + holiday). Gaps larger than 5 days suggest missing data.

```python

def check_missing_dates(df: pd.DataFrame, max_gap_days=5) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["prev_date"] = df["date"].shift(1)
    df["gap_days"]  = (df["date"] - df["prev_date"]).dt.days
    gaps = df[df["gap_days"] > max_gap_days].copy()
    gaps["anomaly_type"] = "missing_date_gap"
    gaps["detail"]       = gaps["gap_days"].apply(lambda g: f"Gap of {g} calendar days")
    log.info(f"[CHECK 4] Date gaps > {max_gap_days} days: {len(gaps)} found")
    return gaps[["date", "gap_days", "anomaly_type", "detail"]]

```

Result: **0 found** ✓

### Check 5 — Bid-Ask Inversions & Wide Spreads <a name="check-bidask"></a>

A bid-ask *inversion* (bid > ask) is a hard data error. A *wide spread* (spread > 50% of mid-price) is not an error but a liquidity concern.

```python

def check_bid_ask_inversions(df: pd.DataFrame) -> pd.DataFrame:
    inversions = df[df["bid"] > df["ask"]].copy()
    inversions["anomaly_type"] = "bid_ask_inversion"

    valid = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])].copy()
    valid["mid"]        = (valid["bid"] + valid["ask"]) / 2
    valid["spread_pct"] = (valid["ask"] - valid["bid"]) / valid["mid"].replace(0, np.nan)
    wide = valid[valid["spread_pct"] > MAX_BID_ASK_SPREAD_PCT].copy()
    wide["anomaly_type"] = "wide_bid_ask_spread"

    result = pd.concat([inversions, wide], ignore_index=True)
    log.info(f"[CHECK 5] Bid-ask inversions: {len(inversions)} | Wide spreads: {len(wide)}")
    return result[["expiration", "option_type", "strike", "bid", "ask", "anomaly_type", "detail"]]

```

Result: **0 inversions, 61 wide spreads** — all concentrated in deep OTM near-expiry contracts, expected behaviour for weekly options. ✓

### Check 6 — Implied Volatility Outliers <a name="check-iv"></a>

Options with zero IV indicate missing vendor data. Options with IV > 500% are deep OTM contracts where Black-Scholes breaks down.

```python

def check_iv_outliers(df: pd.DataFrame) -> pd.DataFrame:
    zero_iv = df[df["implied_vol"] <= 0].copy()
    zero_iv["anomaly_type"] = "zero_implied_vol"

    high_iv = df[df["implied_vol"] > MAX_IV_THRESHOLD].copy()
    high_iv["anomaly_type"] = "extreme_implied_vol"

    result = pd.concat([zero_iv, high_iv], ignore_index=True)
    log.info(f"[CHECK 6] IV outliers — zero: {len(zero_iv)}, extreme: {len(high_iv)}")
    return result[["expiration", "option_type", "strike", "implied_vol", "anomaly_type", "detail"]]

```

Result: **0 zero-IV, 5 extreme IV** (deep OTM contracts). Filtered before analysis. ✓

### Check 7 — Sparse Expirations <a name="check-sparse"></a>

```python

def check_sparse_expirations(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["expiration", "option_type"]).size().reset_index(name="strike_count")
    sparse = counts[counts["strike_count"] < MIN_OPTION_ROWS_PER_EXPIRY].copy()
    sparse["anomaly_type"] = "sparse_expiration"
    log.info(f"[CHECK 7] Sparse expirations: {len(sparse)} found")
    return sparse

```

Result: **0 found** ✓

### Check 8 — Data Freshness <a name="check-freshness"></a>

```python

def check_data_freshness(price_df: pd.DataFrame) -> bool:
    latest    = price_df["date"].max()
    staleness = (pd.Timestamp.today() - latest).days
    if staleness > MAX_STALE_DAYS:
        log.warning(f"[CHECK 8] Data is STALE — {staleness} days since {latest.date()}")
        return False
    log.info(f"[CHECK 8] Data freshness OK — latest: {latest.date()} ({staleness} days ago)")
    return True

```

Result: **Data is fresh** — latest date is 2026-02-20, 0 days stale. ✓

### Validation Summary Report <a name="validation-summary"></a>

```
2026-02-20 18:05:01  INFO      ▶  STAGE 2 / 3: VALIDATION
2026-02-20 18:05:01  INFO      [CHECK 1] Price spikes (|z| > 4.0): 4 found
2026-02-20 18:05:01  WARNING     Spike on 2025-04-03: Return=-4.93%, Z=-4.84
2026-02-20 18:05:01  WARNING     Spike on 2025-04-04: Return=-5.85%, Z=-5.74
2026-02-20 18:05:01  WARNING     Spike on 2025-04-09: Return=10.50%, Z=10.11
2026-02-20 18:05:01  WARNING     Spike on 2025-04-10: Return=-4.38%, Z=-4.31
2026-02-20 18:05:01  INFO      [CHECK 2] Zero/negative prices: 0 found
2026-02-20 18:05:01  INFO      [CHECK 3] OHLC inconsistencies: 0 found
2026-02-20 18:05:01  INFO      [CHECK 4] Date gaps > 5 days: 0 found
2026-02-20 18:05:01  INFO      [CHECK 8] Data freshness OK — latest: 2026-02-20 (0 days ago)
2026-02-20 18:05:01  INFO      [CHECK 5] Bid-ask inversions: 0 | Wide spreads: 61
2026-02-20 18:05:01  INFO      [CHECK 6] IV outliers — zero: 0, extreme: 5
2026-02-20 18:05:01  INFO      [CHECK 7] Sparse expirations: 0 found
```

**Data Quality Summary**

| Check | Result |
|---|---|
| Price Spikes | 4 ISSUE(S) — all real market events |
| Zero/Negative Prices | PASS |
| OHLC Inconsistencies | PASS |
| Missing Date Gaps | PASS |
| Bid-Ask Inversions | PASS |
| Wide Bid-Ask Spreads | 61 ISSUE(S) — all deep OTM near-expiry, expected |
| IV Outliers | 5 ISSUE(S) — deep OTM, filtered before analysis |
| Sparse Expirations | PASS |
| Data Freshness | PASS (0 days stale) |
| **TOTAL ANOMALIES** | **70** |

---

# Stage 3 — Time Series & Volatility Analysis <a name="analysis"></a>

The analysis stage lives in [`src/analyze.py`](https://github.com/MugeniAI05/spy-volatility-pipeline/blob/main/src/analyze.py). It runs five quantitative models and writes all computed metrics to the `analysis_results` table so results accumulate into a historical time series across daily runs.

### Data Loading & Log Returns <a name="data-loading"></a>

Log returns are used in preference to simple returns throughout. They are additive over time, more normally distributed, and are the standard input for GARCH and other financial time series models.

```python

def load_prices() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT date, close, adj_close, volume FROM price_history ORDER BY date ASC", conn
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    price_col    = "adj_close" if df["adj_close"].notna().sum() > 10 else "close"
    df["price"]  = df[price_col]
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    return df.dropna(subset=["log_return"])

```

### Rolling Realized Volatility <a name="rvol"></a>

Annualized realized volatility is computed using rolling standard deviation of log returns over three standard windows: 10-day (short-term), 21-day (monthly), and 63-day (quarterly).

```python

def rolling_realized_vol(df: pd.DataFrame) -> pd.DataFrame:
    result = df[["log_return"]].copy()
    for w in RVOL_WINDOWS:   # (10, 21, 63) from config
        result[f"rvol_{w}d"] = df["log_return"].rolling(w).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    log.info("[RVOL] Rolling realized volatility computed:")
    latest = result.iloc[-1]
    for w in RVOL_WINDOWS:
        log.info(f"  {w:>3}d RVol = {latest[f'rvol_{w}d']:.2%}")
    return result

```

```
2026-02-20 18:05:01  INFO      [RVOL] Rolling realized volatility computed:
2026-02-20 18:05:01  INFO         10d RVol = 13.91%
2026-02-20 18:05:01  INFO         21d RVol = 11.66%
2026-02-20 18:05:01  INFO         63d RVol = 11.28%
```

The 3-panel chart below shows price history, rolling realized vol, and GARCH conditional vol:

![SPY Volatility Analysis](vol_analysis.png)

The April 2025 tariff shock is unmistakable across all three panels. The 10-day RVol spiked above 70% annualized — a level not seen in SPY since the COVID crash. The multi-week elevated regime visible in the 63-day line reflects how long the uncertainty persisted even after the immediate spike subsided. By February 2026, all three windows have converged back to the 11–14% range, consistent with the GARCH long-run estimate.

### GARCH(1,1) Model <a name="garch"></a>

A GARCH(1,1) model captures time-varying conditional variance. The model is estimated via maximum likelihood using the `arch` library.

**σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}**

```python

def fit_garch(df: pd.DataFrame):
    from arch import arch_model

    returns_pct = df["log_return"] * 100   # arch expects percentage returns
    model  = arch_model(returns_pct.dropna(), vol="Garch", p=1, q=1,
                        mean="Constant", dist="normal")
    result = model.fit(disp="off")

    cond_vol    = result.conditional_volatility / 100 * np.sqrt(TRADING_DAYS_PER_YEAR)
    params      = result.params
    alpha       = params.get("alpha[1]", np.nan)
    beta        = params.get("beta[1]",  np.nan)
    omega       = params.get("omega",    np.nan)
    persistence = alpha + beta
    longrun_vol = np.sqrt(omega / (1 - persistence)) / 100 * np.sqrt(TRADING_DAYS_PER_YEAR)

    log.info(f"  omega={omega:.6f}, alpha={alpha:.4f}, beta={beta:.4f}")
    log.info(f"  Persistence (alpha+beta) = {persistence:.4f}")
    log.info(f"  Long-run vol = {longrun_vol:.2%}")
    log.info(f"  Current conditional vol  = {cond_vol.iloc[-1]:.2%}")

    return result, cond_vol, {"omega": omega, "alpha": alpha, "beta": beta,
                               "persistence": persistence, "longrun_vol": longrun_vol,
                               "current_vol": cond_vol.iloc[-1]}

```

```
2026-02-20 18:05:02  INFO      [GARCH] GARCH(1,1) model fitted:
2026-02-20 18:05:02  INFO        omega=0.059753, alpha=0.1205, beta=0.8112
2026-02-20 18:05:02  INFO        Persistence (alpha+beta) = 0.9318
2026-02-20 18:05:02  INFO        Long-run vol = 14.86%
2026-02-20 18:05:02  INFO        Current conditional vol  = 12.67%
```

| Parameter | Value | Meaning |
|---|---|---|
| ω (omega) | 0.059753 | Long-run variance intercept |
| α (alpha) | 0.1205 | 12.1% of last period's shock² feeds into today's variance |
| β (beta) | 0.8112 | 81.1% of last period's conditional variance persists |
| α + β (persistence) | 0.9318 | Vol shocks decay slowly |
| Long-run vol | 14.86% | Unconditional annualized volatility |
| Current conditional vol | 12.67% | Model's current estimate — below long-run, calm regime |

A persistence of 0.9318 confirms that SPY volatility is highly persistent. The long-run vol of 14.86% aligns closely with the historical average of the VIX index — a crucial sanity check. The current conditional vol of 12.67% sitting below the long-run level confirms we are in a calm regime post the April 2025 shock.

### Mean-Reversion & ADF Testing <a name="adf"></a>

The ADF test determines whether a series is stationary (mean-reverting) or has a unit root (random walk). We expect log prices to be non-stationary and log returns to be stationary — and verify both.

```python

def mean_reversion_analysis(df: pd.DataFrame) -> dict:
    from scipy import stats

    log_price = np.log(df["price"].dropna())
    ret       = df["log_return"].dropna()

    # ADF on log price level
    y = log_price.diff().dropna()
    x = log_price.shift(1).dropna()
    x, y = x.align(y, join="inner")
    slope, _, _, p_level, _ = stats.linregress(x, y)

    # ADF on returns
    y2 = ret.diff().dropna()
    x2 = ret.shift(1).dropna()
    x2, y2 = x2.align(y2, join="inner")
    slope2, _, _, p_returns, _ = stats.linregress(x2, y2)

    # OU half-life
    lagged   = ret.shift(1).dropna()
    current  = ret.iloc[1:]
    ou_slope = np.polyfit(lagged, current, 1)[0]
    half_life = -np.log(2) / np.log(abs(ou_slope)) if ou_slope < 0 else np.inf

    log.info(f"  Log price  — slope: {slope:.4f}, p-value: {p_level:.4f}")
    log.info(f"  Log returns — slope: {slope2:.4f}, p-value: {p_returns:.4f}")
    log.info(f"  OU half-life (log returns): {half_life:.1f} trading days")

    return {"adf_level_pval": p_level, "adf_returns_pval": p_returns,
            "ou_half_life_days": half_life}

```

```
2026-02-20 18:05:02  INFO      [MEAN-REV] ADF-style regression results:
2026-02-20 18:05:02  INFO        Log price  — slope: -0.0058, p-value: 0.1974 (NON-stationary ✓)
2026-02-20 18:05:02  INFO        Log returns — slope: -1.0912, p-value: 0.0000 (Stationary ✓)
2026-02-20 18:05:02  INFO        OU half-life (log returns): 0.3 trading days
```

| Series | p-value | Conclusion |
|---|---|---|
| Log price level | 0.1974 | Fail to reject unit root → **random walk** |
| Log returns | 0.0000 | Strongly reject unit root → **stationary** |
| OU half-life (returns) | — | **0.3 trading days** — shocks dissipate almost instantaneously |

These results confirm market efficiency. An OU half-life of 0.3 days for returns means SPY, as the world's most liquid ETF, is extremely difficult to exploit via daily-frequency mean-reversion strategies.

### Implied Volatility Surface <a name="iv-surface"></a>

The IV surface maps implied volatility across moneyness (strike / ATM strike) and days-to-expiration. ATM strike is approximated as the strike with minimum IV per expiration. Moneyness is binned into 5% buckets from 0.80 to 1.20.

```python

def implied_vol_surface(options_df: pd.DataFrame) -> pd.DataFrame:
    calls = options_df[options_df["option_type"] == "call"].copy()
    calls = calls[(calls["implied_vol"] > 0.01) & (calls["implied_vol"] < 5.0)]

    def atm_strike(group):
        return group.loc[group["implied_vol"].idxmin(), "strike"]

    atm   = calls.groupby("expiration").apply(atm_strike).rename("atm_strike")
    calls = calls.merge(atm, on="expiration")
    calls["moneyness"] = calls["strike"] / calls["atm_strike"]

    bins   = np.arange(0.80, 1.25, 0.05)
    labels = [f"{b:.2f}" for b in bins[:-1]]
    calls["moneyness_bin"] = pd.cut(calls["moneyness"], bins=bins, labels=labels, include_lowest=True)

    today        = pd.Timestamp.today().normalize()
    calls["dte"] = (calls["expiration"] - today).dt.days

    surface = calls.pivot_table(index="moneyness_bin", columns="dte",
                                 values="implied_vol", aggfunc="mean")
    return surface.dropna(how="all", axis=0).dropna(how="all", axis=1)

```

```
2026-02-20 18:05:02  INFO      [IV SURFACE] Surface shape: (8, 7) (moneyness bins × expirations)
```

![Implied Volatility Surface](iv_surface.png)

**Two important patterns are visible:**

**1. Volatility Skew:** IV is significantly higher at low moneyness (0.85–0.90) than ATM (1.00), reflecting the classic put skew — market participants pay a premium for downside protection. The 0.85 bin shows IV of 47–60% vs 13–25% ATM.

**2. Downward Term Structure:** IV is higher for near-term options than longer-dated ones across most moneyness levels, reflecting elevated short-term uncertainty. This is typical in calm-to-normal markets and tends to invert sharply during crises.

### Volatility Risk Premium <a name="vrp"></a>

The VRP is the spread between implied volatility and realized volatility. A persistently positive VRP means options are systematically overpriced relative to realized vol — sellers of options collect this premium over time.

```python

def vol_risk_premium(price_df: pd.DataFrame, options_df: pd.DataFrame) -> dict:
    rvol_21 = (price_df["log_return"].rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)).iloc[-1]

    near_exp = options_df["expiration"].min()
    atm_iv   = options_df[
        (options_df["expiration"] == near_exp) &
        (options_df["implied_vol"] > 0.01) &
        (options_df["implied_vol"] < 5.0)
    ]["implied_vol"].mean()

    vrp = atm_iv - rvol_21
    log.info(f"  21d Realized Vol  = {rvol_21:.2%}")
    log.info(f"  Near-term ATM IV  = {atm_iv:.2%}")
    log.info(f"  VRP (IV - RVol)   = {vrp:+.2%}  ({'rich' if vrp > 0 else 'cheap'} options)")

    if abs(vrp) > ALERT_VRP_THRESHOLD:
        log.warning(f"  ⚠  VRP exceeds threshold ({ALERT_VRP_THRESHOLD:.0%}): {vrp:+.2%}")

    return {"rvol_21d": rvol_21, "atm_iv": atm_iv, "vrp": vrp}

```

```
2026-02-20 18:05:02  INFO      [VRP] Volatility Risk Premium:
2026-02-20 18:05:02  INFO        21d Realized Vol  = 11.66%
2026-02-20 18:05:02  INFO        Near-term ATM IV  = 49.63%
2026-02-20 18:05:02  INFO        VRP (IV - RVol)   = +37.97%  (rich options)
2026-02-20 18:05:02  WARNING     ⚠  VRP exceeds threshold (20%): +37.97%
```

The VRP alert fired, which in a live deployment would have triggered an email to the configured address. The elevated magnitude (+37.97%) is driven by the nearest expiration being the same day (1 DTE), where the mean IV across the full strike chain includes many wide-spread OTM contracts. A cleaner production implementation would use only the ATM strike's IV rather than a mean across all contracts.

---

# Results Summary <a name="results-summary"></a>

| Stage | Key Output | Value |
|---|---|---|
| ETL | Price rows loaded | 502 |
| ETL | Options contracts loaded | 969 |
| ETL | Pipeline runtime | 3.7 seconds |
| Validation | Total anomalies | 70 (all correctly classified) |
| Validation | Structural errors | 0 |
| RVol | 10d / 21d / 63d | 13.91% / 11.66% / 11.28% |
| GARCH(1,1) | Persistence α+β | 0.9318 |
| GARCH(1,1) | Long-run vol | 14.86% |
| GARCH(1,1) | Current cond. vol | 12.67% (below long-run — calm regime) |
| ADF | Log price p-value | 0.197 (non-stationary ✓) |
| ADF | Log return p-value | 0.000 (stationary ✓) |
| OU | Returns half-life | 0.3 days |
| IV Surface | Shape | (8, 7) — 8 moneyness bins × 7 expirations |
| VRP | IV − RVol | +37.97% (rich options, alert triggered) |

The GARCH model independently recovered a long-run vol estimate of 14.86% — extremely close to the VIX's long-run historical average — without any external calibration. This is a strong sanity check that the model is working correctly on real market data.

---

# Growth & Next Steps <a name="growth-next-steps"></a>

**Engineering:**
- Replace Yahoo Finance with a professional vendor (Bloomberg BLPAPI, Refinitiv, CBOE DataShop) and schedule via Apache Airflow with Slack alerting
- Replace SQLite with Postgres or Snowflake for concurrent access and larger data volumes
- Add a Greeks computation layer using vectorised Black-Scholes (delta, gamma, vega, theta)

**Modelling:**
- Extend GARCH to EGARCH or GJR-GARCH to capture the leverage effect — vol rises more aggressively on negative return shocks than positive ones of equal magnitude
- Fit a parametric IV surface model (SVI or Heston) for arbitrage-free interpolation across the full strike/expiry grid
- Backtest a systematic VRP-harvesting strategy (sell ATM straddles, delta-hedge daily) over the 2-year price history to quantify the realised Sharpe ratio
- Add term structure modelling to forecast the forward vol curve under different regimes

The full source code is on GitHub: **[MugeniAI05/spy-volatility-pipeline](https://github.com/MugeniAI05/spy-volatility-pipeline)**
