# Kairos Pre-Production Audit — Fix Handoff (2026-07-04)

Source: full audit, 5 parallel reviewers (data/ingest, security/API, trading/money, backtest/models, tests/CI/ops), all findings verified against source. Audit-only session — nothing below has been fixed yet.

Baseline at audit time: 364 tests passing, ruff clean, no secrets tracked.

Work top to bottom. Each item: what's broken, evidence, why it matters, fix, and a test to add/change. Re-run the narrowest relevant test after each fix; full suite (`make test`) before calling any batch done.

---

## 1. CRITICAL — GitHub webhook events silently never persisted

**File:** `kairos/ingest/github.py:490-507`, `kairos/db.py:61-186`

`_persist_event()` does:
```python
conn.execute(
    "INSERT INTO github_events (event_type, repo, payload, received_at) VALUES (?, ?, ?, ?)",
    [...],
)
```
`github_events` is never created in `create_schema()` (db.py). Every insert raises a DuckDB catalog/binder error, caught by the broad `except Exception` in `_record_event` (github.py:483-486), which only logs a WARNING. Events survive only in the in-memory `_RECENT_EVENTS` deque (maxlen 10,000) — lost on every restart. No `tests/test_ingest_github.py` exists, so this was never caught.

**Fix:**
- Add a `github_events` table to `create_schema()` in `db.py` (columns matching the insert: `event_type VARCHAR, repo VARCHAR, payload JSON, received_at TIMESTAMP`, plus a sensible primary key / index).
- Add `tests/test_ingest_github.py` exercising `_persist_event` end-to-end (insert then read back), so a future schema/insert mismatch fails loudly.
- Decide whether historical webhook data is unrecoverable (in-memory deque only) or if there's a way to backfill from GitHub's API — flag to product/user, don't silently drop this decision.

---

## 2. HIGH — Regime mislabeling on model fallback

**File:** `kairos/signals/ensemble.py:577-608`

```python
model = self._models.get(regime) if self._fitted.get(regime, False) else None
if model is None:
    fitted = self.fitted_regimes()
    if not fitted:
        raise RuntimeError(...)
    model = self._models[fitted[0]]   # falls back silently to a DIFFERENT regime's model
...
if regime == "lv_down":               # confidence tweak gated on REQUESTED regime, not the model used
    ...
return SignalEvent(..., regime=regime, ...)   # SignalEvent reports the requested regime, not fitted[0]
```

When a regime's sub-model has insufficient training data (`fit_raw` skips regimes with `<10` rows), `predict()` substitutes a different regime's model but labels + confidence-adjusts the output as if it were the originally requested regime. No log, no warning. Corrupts anything keyed on `SignalEvent.regime` downstream (position sizing gates, ATR multipliers, regime-conditioned reporting).

**Fix:**
- When falling back to `fitted[0]`, either (a) set `regime=fitted[0]` on the returned `SignalEvent` so it reflects reality, or (b) log a warning noting the fallback and the mismatch, and gate the `lv_down` confidence-adjustment block on the model actually used, not the requested string.
- Add a test that forces a regime-fallback path (train with `<10` rows in one regime) and asserts either the returned `regime` matches the model used, or a warning fires.

---

## 3. HIGH — Timing-unsafe API key comparison

**File:** `kairos/api/server.py:67`

```python
def _require_auth(x_api_key: str = Header(default="")) -> None:
    if not _api_key:
        raise HTTPException(status_code=503, detail="KAIROS_API_KEY is required")
    if x_api_key != _api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
```
Plain `!=` short-circuits on the first differing byte — a network timing side-channel that can be used to brute-force `KAIROS_API_KEY`.

**Fix:**
```python
import hmac
...
if not hmac.compare_digest(x_api_key, _api_key):
    raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
```
Also add a test hitting `/signals` with a wrong (but correctly-shaped) key, asserting 401 — currently untested (`tests/test_api.py` only tests valid-key success and missing-key 503).

---

## 4. HIGH — No lint job in CI

**File:** `.github/workflows/ci.yml`

Only the test matrix (fast/slow × py3.11/3.12) runs. No `ruff check` / `ruff format --check` step anywhere in CI. Lint is enforced only via local pre-commit, which is optional to install (`make pre-commit-install` is a manual target) and skippable (`--no-verify`).

Compounding: `.pre-commit-config.yaml:15` runs ruff with `--select "E,W,F,I,N" --ignore "N803,N806,N814,E741,F841,E712,E402"` while `Makefile:18` runs plain `ruff check kairos/ tests/` (pyproject defaults, no overrides) — the two configs can silently diverge and nothing catches it.

**Fix:**
- Add a `lint` job to `ci.yml` running `ruff check kairos/ tests/` and `ruff format --check kairos/ tests/` (match the Makefile invocation, not the pre-commit override, or reconcile the two configs to use one source of truth — prefer `pyproject.toml`'s `[tool.ruff]` as the single config and drop the pre-commit override).
- Make it a required check.

---

## 5. HIGH (SUSPECTED — verify before assuming fixed) — Circuit-breaker half-open concurrency race

**File:** `kairos/circuit_breaker.py:67, 181`

Reviewer flagged: the half-open probe-slot check and in-flight counter increment don't appear to be atomic across the `await` boundary — two concurrent calls could both observe HALF_OPEN and both proceed past the slot check before either increments. Similarly, the recovery-probe task creation (line 181) may do a check-then-create outside the lock, risking duplicate recovery tasks under rapid flapping failures.

This was **not fully re-verified** — the reviewing agent hit a tooling outage mid-trace and could not confirm lock scope with certainty on a final pass. Treat as "trace and confirm" rather than settled.

**Fix:**
- Read `circuit_breaker.py` end to end, specifically the lock scope around: (a) the HALF_OPEN state check + probe-slot accounting, (b) recovery-task scheduling.
- If the race is real: make the state-check + slot-increment atomic (single lock acquisition spanning both), and make recovery-task creation a check-then-create protected by the same lock or an explicit "already scheduled" flag set under the lock.
- Add a concurrency test: fire N concurrent calls at a HALF_OPEN breaker with `half_open_max_requests=1`, assert only one actually proceeds through the guarded callable.

---

## MEDIUM

### 6. Docker never build-tested pre-merge, never smoke-tested pre-publish
**Files:** `.github/workflows/docker-publish.yml:3-6, 47-54`
- Triggers only on push to main/tags — a broken Dockerfile is only discovered post-merge.
- Goes straight from `build-push-action` to `push: true` with no smoke step (e.g. `docker run --rm <image> --help`).
**Fix:** Add a `docker build` step to `ci.yml` on PRs (build only, no push). Add a smoke-run step to `docker-publish.yml` before the push step.

### 7. Docker image never actually starts the API server
**Files:** `Dockerfile:44`, `docker-compose.yml:9,21`
`CMD ["kairos", "--asset", "BTC", "--no-tui"]` runs the one-shot CLI, never `create_app`/uvicorn. Yet `EXPOSE 8000` + `ports: ["8000:8000"]` imply the API is reachable — it isn't, nothing binds that port by default.
**Fix:** Decide the intended deployment shape — either add a second compose service / CMD variant that runs the API server, or drop the `EXPOSE`/port mapping and document that the API is opt-in (separate invocation).

### 8. Alerting env vars missing from docker-compose
**File:** `docker-compose.yml:12-20` vs `kairos/config.py:8-36`
Compose forwards only `FRED_API_KEY, GITHUB_TOKEN, SOLANA_RPC_URL, REDDIT_CLIENT_ID/SECRET, CRYPTOCOMPARE_API_KEY, KAIROS_API_KEY`. Missing: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `DISCORD_WEBHOOK_URL`, `SMTP_*`, `COINGECKO_API_KEY`, `CRYPTOPANIC_API_KEY`, `ALPHA_VANTAGE_API_KEY`. Alerting silently no-ops in containerized deployments.
**Fix:** Add the missing vars to `docker-compose.yml` environment block. Also add them to `.env.example` (currently undocumented there too).

### 9. Paper-trade persistence-failure divergence (needs confirmation)
**File:** `kairos/papertrade.py:249`
Broad except around the trade-persistence INSERT logs and continues without rolling back in-memory position/capital state already mutated before the write. On restart, `_load_trades` reconstructs from DB only — a failed write means in-memory state (which said the trade closed) diverges from what's persisted.
Countervailing evidence: `tests/test_papertrade.py` already has `test_failed_close_persist_does_not_corrupt_memory` and `test_persist_close_idempotent`. **Read those tests and the :249 code path together** before treating this as unresolved — it may already be handled; if so, downgrade/close this item.
**Fix (if confirmed unhandled):** on persistence failure, either retry with backoff (DuckDB lock contention is a known pattern here — see `db.py` connection retry logic) or revert the in-memory mutation so state matches what's actually durable.

### 10. Silent synthetic-model fallback with no downstream signal
**File:** `kairos/signals/regime.py:123, 301`
`fit()` on failure falls back to `fit_synthetic_fallback()` (a model trained on synthetic, not real, data) with no marker distinguishing it from a real fit. Separately, `predict_regime`'s except path returns `"neutral"` with `confidence=0.0` on transform failure — verify that live.py's signal-generation path actually gates on `confidence > 0` before consuming a regime read; if it doesn't, a silently-neutral zero-confidence regime could still feed a trade decision.
**Fix:** Add an `is_synthetic: bool` (or similar) flag on the model/cache so callers can log/alert when trading against a synthetic fallback. Confirm and, if needed, add an explicit `confidence == 0` gate wherever regime output feeds trading decisions.

### 11. Untested retry/backoff on price fetch
**File:** `kairos/live.py:51-67` (`_coingecko_get`)
Real exponential backoff on HTTP 429 (1→2→4→8→16s, 5 attempts, ~31s worst case) feeds directly into signal freshness. The only related test (`tests/test_live.py:367-386`) mocks the client to raise `HTTPStatusError` immediately rather than simulate an actual 429 response — the retry loop itself is never exercised.
**Fix:** Add a test that returns a mocked 429 response (not a raised exception) and asserts the backoff sequence/attempt count/eventual success-or-give-up behavior.

### 12. Broad exception swallowing (multiple sites)
**Files:** `engine.py:875-881` (backfill loop; inner `except Exception: pass` around `store.close()` is fully silent), `live.py:99,127,421`, `circuit_breaker.py:67,181`, `regime.py:123,301,333,444`
Code bugs (TypeError/KeyError from malformed data) and transient network errors are handled identically — logged (or not logged, in the `close()` case) and swallowed, with no differentiation and no alerting on repeated non-transient failures.
**Fix:** At minimum, make every bare-`pass` except site log something (even DEBUG level). Where feasible, narrow the exception type to what's actually expected (e.g. `httpx.RequestError` instead of `Exception`) so real bugs surface instead of being treated as expected transient failures.

### 13. DuckDB concurrent-write contention (needs confirmation)
**Files:** `kairos/db.py` (retry/backoff on lock contention), `kairos/ingest/whale.py`, `kairos/papertrade.py`
The existence of lock-contention retry logic in `db.py` implies known contention; unclear whether per-asset concurrent writes (e.g. via `asyncio.gather` in live.py) are serialized above the connection-retry level, or just rely on retry-and-hope.
**Fix:** Trace whether concurrent per-asset signal processing can hit the same DuckDB file simultaneously; if so, either serialize writes with an explicit lock or confirm the existing retry budget is sufficient and document why.

### 14. FRED macro ingestion aborts entire series on one malformed observation
**File:** `kairos/ingest/macro.py:76-84`
`obs["date"]` uses bracket access (not `.get`) inside a loop that only catches `(ValueError, TypeError)` around the `float(raw_val)` conversion. A malformed/truncated FRED observation missing the `date` key raises an uncaught `KeyError`, aborting `_fetch_and_store_macro_inner` for **all remaining series** in that call — no retry, no per-observation isolation.
**Fix:** Wrap per-observation processing so one bad record is skipped (logged) rather than aborting the whole batch; use `.get("date")` and check for `None` explicitly.

### 15. Price ingestion retry loop bypassed by malformed payloads; silent data truncation
**File:** `kairos/ingest/price.py:26-44, 56`
The retry loop only catches `(httpx.RequestError, httpx.HTTPStatusError)`. `resp.json()` raising `JSONDecodeError`, or downstream `KeyError`/`TypeError` on a malformed CoinGecko payload, breaks out of the retry loop immediately instead of retrying — inconsistent with the retry behavior for HTTP-level failures. Separately, `price.py:56` does `zip(prices, volumes)`: if CoinGecko returns mismatched-length arrays, extra points are dropped silently with no warning.
**Fix:** Broaden the retry-eligible exception set to cover payload-shape errors (or explicitly decide they should NOT retry and document why), and log a warning when `zip` truncates due to length mismatch.

### 16. Whale-transfer persistence: non-atomic check-then-insert race
**File:** `kairos/ingest/whale.py:296-301, 320-322`
Does `SELECT 1 ... WHERE signature = ?` then `INSERT`. Since `signature` is `PRIMARY KEY` (`db.py:155`), two coroutines processing the same signature concurrently can race: the loser hits a DuckDB constraint violation, caught only by the broad `except Exception` at :320-322 — logged as `"Failed to persist whale transfer"` (reads as a real failure when it's a benign duplicate) and unconditionally resets the thread-local connection (`_persist_local.conn = None`), forcing needless reconnect churn.
**Fix:** Catch the constraint-violation case specifically (e.g. `duckdb.ConstraintException`) and treat it as an expected benign duplicate (debug-log, don't reset the connection), rather than routing it through the generic failure path.

### 17. Whale websocket: one malformed frame kills the whole connection
**File:** `kairos/ingest/whale.py:684`
`json.loads(raw)` inside `async for raw in ws` has no try/except — one malformed frame from the RPC provider throws out of the entire `async with websockets.connect(...)` block, forcing a full reconnect + resubscribe cycle. Mitigated by exponential backoff (:697-704) so not a crash, but avoidable latency/churn.
**Fix:** Wrap the per-message `json.loads` in a try/except that logs and continues to the next message instead of tearing down the whole connection.

### 18. Whale module: multiple independent DuckDB connections to the same file
**File:** `kairos/ingest/whale.py:75, 284-288, 340`
Mixes a per-thread cached connection (`_get_persist_conn`) with separately opened/closed connections in `get_whale_metrics` (:340) and `refresh_exchange_wallets_from_db` (:75), all against the same `DB_PATH`. `db.py:get_connection`'s lock-contention retry (max ~11.5s total backoff) mitigates this, but no test verifies behavior when the retry budget is exhausted — the resulting bare `RuntimeError` propagates uncaught to callers like `sentiment.py:54`, `github.py:491`.
**Fix:** Either consolidate to a single shared connection pattern per process, or add a test that simulates retry-budget exhaustion and confirms callers handle the `RuntimeError` gracefully (or explicitly decide it should propagate and crash — document the decision).

### 19. Discord webhook URL not log-suppressed (Telegram bot token is)
**File:** `kairos/notifier.py:176-187` vs `208-211`
`_send_telegram` explicitly drops the `httpx`/`httpcore` logger to ERROR around the call because the URL embeds the bot token. `_send_discord` (`notifier.py:208-211`) makes the equivalent `httpx.AsyncClient().post(f"{webhook_url}?wait=true", ...)` call with no such suppression, even though the Discord webhook URL is itself bearer-style secret. Latent (default Python root logger is WARNING, so only bites if logging is reconfigured to INFO+) — hence SUSPECTED, not CONFIRMED.
**Fix:** Apply the same logger-suppression pattern used for Telegram around the Discord call, for consistency and defense-in-depth.

### 20. papertrade.py has zero circuit-breaker integration (info — confirm intentional)
**File:** `kairos/papertrade.py` vs `kairos/live.py:17,483`
`CIRCUIT_BREAKERS` is only imported/used in `live.py`. Likely fine by design (papertrade doesn't call external APIs directly), but worth a deliberate check that a stalled/failed upstream data source can't leave a paper position stuck open with no circuit-breaker-driven timeout/exit.
**Fix:** Confirm intentional; if there's any path where papertrade waits on live data without a breaker or timeout, add one. Otherwise close as by-design.

---

## LOW / BACKLOG

- **Circuit-breaker state is in-memory only** (`circuit_breaker.py`) — no DB persistence. On crash+restart, breaker resets to CLOSED regardless of pre-crash failure history — if the crash was caused by the same failing dependency, expect an immediate retry against a known-bad dependency before it re-trips. Consider persisting breaker state or at least logging "cold start after crash" for operator visibility.
- **`report.py:105-116` `_series_for_rows`** — realigns `equity_curve`/`benchmark_curve` against `timestamps` via a length heuristic (`if len(values) == row_count + 1: values = values[1:]`), compensating for `engine.py:996` seeding `equity_curve` with an extra unpaired value before the loop. Works correctly today but is fragile — a future change that happens to produce a `row_count + 1`-length list for unrelated reasons would silently (and wrongly) drop a real data point. Fix: either don't seed with an unpaired value (pair it with a synthetic day-0 timestamp), or pass an explicit `has_seed_offset: bool` from engine to report instead of inferring from length.
- **Possible NaN/div-zero**: high-watermark drawdown calc in `papertrade.py` (`(high_watermark - current_price) / high_watermark`), narrative-velocity calc in `narrative.py` (`(current - previous) / previous`-style) — neither showed an obvious epsilon/zero-guard the way `causal.py`'s feature-vector sanitizer does. Verify whether `high_watermark`/`previous` can legitimately be 0 in practice; if so, guard against div-by-zero explicitly.
- **Kelly-fraction formula** — `MAX_KELLY_FRACTION` clamps the output fraction but the underlying Kelly formula's odds term wasn't confirmed to be guarded against zero before the clamp is applied. Verify.
- **`causal.py:33-41`** — `hv_up` (high-vol up) gets no directional boost (grouped with `transition` for a confidence-only discount) while `hv_down` gets the full bearish `p_up *= 0.70` multiplier alongside `lv_down`/`distribution`. Matches the module's stated Soros-reflexivity design intent (high vol ⇒ uncertainty, not direction) but is a real asymmetry between the two high-vol regimes — get explicit sign-off from whoever owns the regime model that this is intentional, not a missing `hv_up` bullish branch.
- **Timing-dependent circuit-breaker tests** — `tests/test_circuit_breaker.py:50-70` uses `recovery_timeout=0.01` + `asyncio.sleep(0.03)` (3x margin) to force HALF_OPEN transitions. Fine locally, latent flake risk under CI runner contention. Consider a fake clock instead of real sleeps if it ever flakes.
- **Model-cache corruption silently discarded and retrained** (`regime.py:444`) — self-healing behavior, but repeated silent corruption (e.g. a disk issue) is never surfaced to an operator. Consider logging at WARNING when a cache load fails, even though retraining continues.
- **`ensemble.py`/`causal.py` tests assert ranges/ordering, not exact magnitudes** — a regression that shifts probability magnitudes while preserving ordering/range would pass undetected. Consider pinning a few golden-value regression tests once behavior is stable.
- **`engine.py:254-261` `_multi_asset_kelly` unbounded overlap-exposure sum** — `overlapping_exposure` is an unbounded additive sum across all other assets rather than a normalized portfolio-variance term; only tamed by the final `gross_exposure > 1.0` renormalization, not a principled risk budget. Fine for the current 2–3 asset universe; revisit if the asset universe grows.

---

## Verified clean — do not "fix" these

- **Walk-forward look-ahead leakage**: checked `engine.py` train/feature window slicing end to end — all windows correctly bounded at the current step. The one full-array precompute (Kalman slopes for Strategy C, `engine.py:1016-1022`) is safe because `kalman_smooth` (`kalman.py:22-27`) is a strictly causal forward-only recursive filter with no backward/RTS pass.
- **Kelly-fraction correlation-matrix construction** (`engine.py:195-267`, `_multi_asset_kelly`): index alignment, time alignment, and directional sign logic all verified correct by truth table (same-direction + positive-corr → penalized; opposite-direction + positive-corr → correctly treated as hedge; opposite-direction + negative-corr → correctly penalized).
- **`db.py:128` f-string SQL**: interpolates only `_feature_store_columns()`, built from the `FeatureVector` dataclass's own field names — never user/external input. Not injectable today. (Still worth an explicit identifier-safety assertion if `FeatureVector` ever gains an externally-derived field — noted, not urgent.)
- **API CORS/auth surface** (`server.py`): CORS restrictive (localhost default, `allow_credentials=False`, GET-only), all non-health routes fail-closed without a configured key, `/signals` params bounded (`limit` 1–1000), docs/redoc/openapi disabled. Good.
- **Dockerfile privilege posture**: non-root `appuser`, multi-stage build strips tests/pycache, `no-new-privileges:true` + `cap_drop: [ALL]` in compose. Good.
- **`model_cache.py`**: atomic save (temp file + rename), XGBoost native binary format only (no pickle), path-traversal-safe filename sanitization. Good.
- **No skipped/xfail tests anywhere in `tests/`** — no hidden failures.

---

## Suggested order for Codex

1. Item 1 (schema fix — 5 min, unblocks real data).
2. Item 3 (one-liner, `hmac.compare_digest`).
3. Item 2 (regime fallback labeling + test).
4. Item 4 (CI lint job).
5. Item 5 (circuit-breaker race trace — read before touching; may turn out to be a non-issue).
6. Items 6–20 (mediums) in any order; each is independent. Items 9, 13, 20 require a quick read-and-confirm before deciding whether a code change is even needed.
7. Low/backlog items opportunistically.

Run `make test` after each item; run `make lint` after touching anything in `kairos/`.
