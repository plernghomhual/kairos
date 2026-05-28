# Kairos — Causal Economic Signal Engine
## Design Spec · 2026-05-28

---

## What It Is

Kairos is open-source financial intelligence infrastructure. It detects market-moving signals before they appear in prices by modeling the actual causal chain: real-world events → narrative spread → price impact.

Most systems measure prices. Kairos measures everything that causes prices to move, before prices move.

**Core thesis:** Every market move has a real-world precursor that leaves a trace in public data. Prices are the last thing to move. Everything else moves first.

**"Too dangerous to be free":** Gives retail traders the same causal intelligence that institutional quant funds pay $100M/year to extract. Fully explainable — every signal cites a named economic principle.

---

## Theoretical Foundation

All model choices are grounded in proven academic work:

| Principle | Author | Applied To |
|-----------|--------|------------|
| Narrative Economics — narratives spread like epidemics | Shiller (Nobel 2013) | Layer 2: SIR model on news graph |
| Financial Instability Hypothesis — stability breeds instability | Minsky | Layer 3: regime stage detection |
| Reflexivity — prices affect fundamentals in feedback loops | Soros | Layer 3: regime amplification |
| Manias, Panics, Crashes — bubbles follow 5 measurable stages | Kindleberger | Layer 3: Minsky stage labeling |
| Efficient Market gaps — semi-strong vs strong form | Fama | Exploiting narrative lag window |
| Adaptive Markets — strategies evolve with regimes | Andrew Lo | Regime-conditional signal weighting |

---

## MVP Scope

**Asset:** BTC (Bitcoin)
**Signal type:** Narrative momentum → price move prediction
**Proving threshold:** Signal fires before 7 of 10 known historical BTC moves >15% (2020–2024 backtest)
**Interface:** CLI + local JSON signal stream + local FastAPI

---

## Architecture

Three signal layers feed an XGBoost ensemble. All processing is local. All outputs are explainable.

```
[Data Sources]
Reddit (PRAW) · CryptoCompare News · CoinGecko · FRED · RSS
        ↓ async ingest every 5 minutes
[DuckDB — local single-file store: kairos.db]
        ↓
Layer 1 — Reality
  Kalman Filter         clean price/volume time series
  Isolation Forest      flag statistical anomalies
        ↓
Layer 2 — Narrative
  Entity Graph          networkx graph of news entities + Reddit mentions
  SIR Epidemic Model    model narrative spread velocity (dI/dt)
  Tipping Point Flag    detect when narrative crosses critical threshold
        ↓
Layer 3 — Regime
  HMM (3 states)        accumulation / distribution / transition
  Bayesian Causal DAG   encode known economic causal relationships
        ↓
  XGBoost Ensemble      combine all features → signal score
        ↓
  SignalEvent (JSON)    structured output with citations
```

---

## Signal Event Output

Every signal is a structured JSON object. No black boxes.

```json
{
  "asset": "BTC",
  "direction": "bullish",
  "confidence": 0.74,
  "regime": "accumulation",
  "narrative_velocity": 2.3,
  "narrative_tipping_point": true,
  "mechanism": "narrative_momentum → retail_fomo → price",
  "estimated_hours_to_price": 36,
  "citations": ["Shiller 2017 - Narrative Economics", "Minsky Stage 2"],
  "triggered_at": "2026-05-28T14:22:00Z"
}
```

---

## File Structure

```
kairos/
├── pyproject.toml
├── .env.example
├── kairos/
│   ├── cli.py                    # Typer CLI
│   ├── config.py                 # Settings, API keys, thresholds
│   ├── db.py                     # DuckDB connection + schema
│   ├── ingest/
│   │   ├── reddit.py             # PRAW: r/Bitcoin, r/CryptoCurrency
│   │   ├── news.py               # CryptoCompare + RSS
│   │   ├── price.py              # CoinGecko OHLCV
│   │   ├── macro.py              # FRED: DFF, M2, UNRATE
│   │   └── scheduler.py          # Async 5-minute ingest loop
│   ├── signals/
│   │   ├── kalman.py             # Layer 1: Kalman filter
│   │   ├── anomaly.py            # Layer 1: Isolation Forest
│   │   ├── narrative.py          # Layer 2: SIR + entity graph
│   │   ├── regime.py             # Layer 3: HMM regime detector
│   │   ├── causal.py             # Layer 3: Bayesian causal DAG
│   │   └── ensemble.py           # XGBoost signal ensemble
│   ├── models/
│   │   └── signal_event.py       # SignalEvent dataclass
│   ├── backtest/
│   │   └── runner.py             # Historical backtest harness
│   └── api/
│       └── server.py             # FastAPI local server
└── tests/
    ├── test_db.py
    ├── test_kalman.py
    ├── test_narrative.py
    ├── test_regime.py
    ├── test_ensemble.py
    └── test_backtest.py
```

---

## CLI Interface

```bash
kairos watch BTC              # live signal stream to terminal
kairos backtest BTC 2022      # run against 2022 historical data, print hit rate
kairos explain last           # why did the last signal fire?
kairos serve                  # start local FastAPI on :8000
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Ingestion | Python asyncio, PRAW, httpx |
| Storage | DuckDB (single `.db` file, zero config) |
| Signal processing | numpy, scipy, hmmlearn, networkx, scikit-learn |
| ML ensemble | XGBoost |
| API | FastAPI + uvicorn |
| CLI | Typer |
| Testing | pytest |
| Future perf layer | Rust ingestion worker |

---

## License

AGPL-3.0 — free for everyone. If a commercial product is built on top, it must be open-sourced. Protects the project from being absorbed by hedge funds without contribution.

---

## Success Criteria

1. Backtest hit rate ≥ 70% on 2020–2024 BTC data (7/10 major moves predicted)
2. Signal fires at least 12 hours before price move
3. Every signal output includes mechanism + citations
4. Full pipeline runs locally with zero cloud dependency
5. `kairos watch BTC` produces live terminal output within 60 seconds of startup
