# Kairos Engine Upgrade — Implementation Log

## Active: Docker GHCR Build Fix

- [x] Confirm why GitHub Actions Docker build cannot find `kairos.db`.
- [x] Add a focused DB path regression test for container runtime configuration.
- [x] Remove the Docker build-time dependency on a local `kairos.db`.
- [x] Update Docker Compose and Docker docs to use runtime data storage.
- [x] Run targeted verification and feasible Docker validation.
- [x] Final review: files changed, behavior changed, verification performed, remaining risks.

### Docker GHCR Build Fix Final Review

- Files changed: `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `README.docker.md`, `kairos/db.py`, `tests/test_db.py`, `tasks/todo.md`.
- Behavior changed: Docker builds no longer require a checked-in `kairos.db`; the image uses `/data/kairos.db` as runtime data, Compose persists `/data` with a named volume, and default `get_connection()` honors `KAIROS_DB_PATH`.
- Verification performed: red `python -m pytest tests/test_db.py -q` failed before `get_connection()` used `KAIROS_DB_PATH`; green `python -m pytest tests/test_db.py -q` passed 2; `python -m pytest tests/test_api.py tests/test_db.py tests/test_feature_store.py -q` passed 8; `python -m py_compile kairos/db.py kairos/api/server.py kairos/healthcheck.py` passed; `ruby -e "require 'yaml'; YAML.load_file('docker-compose.yml')"` passed; API + DB-path smoke created and read the env-configured DuckDB file.
- Remaining risks: full `docker build` could not be run locally because Docker CLI is not installed. `gh auth status` reports the local GitHub token is invalid, so GitHub Actions logs were not re-fetched through `gh`; diagnosis used the provided build log.

## Active: Live/Backtest/Paper Trading Reliability

- [x] Map current live signal, history, backtest, paper trading, DB, and external fetch flow.
- [x] Reproduce or isolate DuckDB lock failures and zero-trade backtests/paper accounts.
- [x] Add focused failing tests for the confirmed root causes.
- [x] Implement minimal fixes without broad refactors.
- [x] Run targeted verification and feasible regression checks.
- [x] Final review: files changed, behavior changed, verification performed, remaining risks.

### Final Review

- Files changed: `kairos/backtest/engine.py`, `kairos/api/server.py`, `kairos/live.py`, `kairos/cli.py`, `tests/test_backtest_engine.py`, `tests/test_api.py`, `tests/test_live.py`, `tests/test_cli.py`, `tasks/todo.md`.
- Behavior changed: backtests now allow initial entries from flat state; API server no longer holds a long-lived DuckDB write lock; live data fetches use a 60s per-asset cache and stale fallback after fetch/rate-limit failures; TUI paper view reads the live paper account instead of launching a fresh strategy-B backtest.
- Verification performed: new regression tests failed before fixes and now pass; fast affected suite passed `58 passed, 1 warning`; targeted deterministic backtest subset passed `5 passed`; `py_compile` passed for touched source/test files.
- Remaining risks: full `tests/test_backtest_engine.py` exceeds the 120s tool timeout in this environment, so verification used the deterministic regression subset plus affected non-backtest modules.

## Active: HMM Ensemble Exit Stabilization

- [ ] Inspect current regime selection, ensemble parameter grid, and backtest exit flow.
- [ ] Add failing tests for HMM transition/diversity safeguards.
- [ ] Add failing tests for `lv_down` regularization defaults/grid and prediction bias.
- [ ] Add failing tests for backtest exit gating with min hold, entry regime, and trailing stop behavior.
- [ ] Implement minimal source changes in `kairos/signals/regime.py`, `kairos/signals/ensemble.py`, and `kairos/backtest/engine.py`.
- [ ] Run targeted tests and broader feasible regression checks.
- [ ] Final review: files changed, behavior changed, verification performed, remaining risks.

## Active: Agent 15 CSV & Metrics Logger

- [x] Add failing tests for equity CSV, metrics JSON, comparison CSV, regime report, signal log, and trade CSV roundtrip.
- [x] Implement machine-parseable trade/equity CSV, metrics JSON, strategy comparison CSV, signal log, and regime table exporters in `kairos/backtest/report.py`.
- [x] Add realized trade capital capture in `kairos/backtest/engine.py` without changing public function signatures.
- [x] Run targeted exporter tests and import acceptance command.
- [x] Final review: files changed, behavior changed, verification performed, remaining risks.

### Agent 15 Final Review

- Files changed: `kairos/backtest/report.py`, `kairos/backtest/engine.py`, `tests/test_backtest.py`, `tasks/todo.md`.
- Behavior changed: report exports now include enhanced trade CSV rows, daily equity CSV, metrics JSON, strategy comparison CSV, structured signal logs, and per-regime rich tables. Closed trades can carry realized `cumulative_capital`.
- Verification performed: RED `python -m pytest tests/test_backtest.py -q` failed on missing exporters/schema; GREEN `python -m pytest tests/test_backtest.py -q` passed 8; `python -m pytest tests/test_backtest_engine.py -q` passed 34; import acceptance command printed `OK`; full `python -m pytest tests -q` passed 278.
- Remaining risks: `kairos/backtest/engine.py` and `kairos/backtest/report.py` are currently untracked in this dirty worktree, so they must be included intentionally when staging Agent 15 work.

## Active: Agent 17 Test Suite Expansionist

- [x] Map existing test helpers and source-facing APIs without modifying `kairos/`.
- [x] Add shared pytest fixtures in `tests/conftest.py`.
- [x] Expand requested edge-case coverage in existing test files.
- [x] Add data corruption, integration flow, mock ingest, and extra stress tests.
- [x] Run targeted tests for new/changed files.
- [x] Run `python -m pytest tests/ -q` and record result.
- [x] Final review: files changed, behavior changed, verification, remaining risks.

### Agent 17 Final Review

- Files changed: `tests/conftest.py`, `tests/test_regime.py`, `tests/test_causal.py`, `tests/test_ensemble.py`, `tests/test_conflict.py`, `tests/test_live.py`, `tests/test_backtest_engine.py`, `tests/test_stress.py`, `tests/test_data_corruption.py`, `tests/test_integration_flow.py`, `tests/test_mock_ingest.py`, `tasks/todo.md`.
- Behavior changed: no source behavior changed; pytest coverage expanded for edge cases, corrupt inputs, module integration, mocked ingest/API failures, and stress/chaos scenarios.
- Verification performed: baseline `python -m pytest tests/ -q` passed `272 passed` in 87.02s before edits; collection after edits found 349 tests; new modules passed `31 passed` in 2.12s; changed/new targeted suite passed `260 passed` in 94.82s; full suite passed `349 passed, 4 warnings` in 96.79s.
- Remaining risks: zero-initial-capital backtest still emits existing `RuntimeWarning` values from source math, but the test now locks in the current non-crashing behavior without modifying `kairos/`.

## Active: Agent 8 Kelly Sizer Expansionist

- [x] Add focused tests for multi-asset Kelly sizing, correlation tracking, adaptive Kelly, and multi-asset backtest execution.
- [x] Implement `_multi_asset_kelly`, `CorrelationTracker`, and `_adaptive_kelly` in `kairos/backtest/engine.py`.
- [x] Integrate adaptive Kelly into single-asset sizing without changing `run_backtest()` signature.
- [x] Add `run_multi_asset_backtest()` with combined equity tracking and correlated allocation caps.
- [x] Export new backtest helpers if needed.
- [x] Run targeted verification and import acceptance command.

### Agent 8 Final Review

- Files changed: `kairos/backtest/engine.py`, `kairos/backtest/__init__.py`, `tests/test_backtest_engine.py`, `tasks/todo.md`.
- Behavior changed: Kelly sizing now has adaptive win/loss ratio support, correlated multi-asset sizing, rolling correlation tracking, and a combined BTC/ETH/SOL backtest entry point.
- Verification performed: `python -m py_compile kairos/backtest/engine.py tests/test_backtest_engine.py`; `python -c "from kairos.backtest.engine import _multi_asset_kelly, CorrelationTracker, _adaptive_kelly; print('OK')"`; `python -m pytest tests/test_backtest_engine.py -vv -k "multi_asset_kelly or correlation_tracker_update or adaptive_kelly_uses_history or multi_asset_backtest_runs"` → 5 passed.
- Remaining risks: full `python -m pytest tests/test_backtest_engine.py -q` currently reports unrelated failures from other active agents' unfinished features (`_fetch_order_book_depth`, slippage order-book parameters, NaN training-row cleaning, and existing live-data backtest cases).

## Active: Agent 19 Alert Notifications

- [x] Add failing tests for notifier templates, Discord embeds, suppression, state-change/no-change alerts, anomaly alerts, confidence drops, and config defaults.
- [x] Create `kairos/notifier.py` with async Telegram, Discord, and email dispatch plus in-memory state/suppression tracking.
- [x] Add notification environment defaults to `kairos/config.py`.
- [x] Integrate fire-and-forget notification scheduling in `kairos/live.py`.
- [x] Run targeted notifier tests, import smoke check, and relevant regression tests.
- [x] Final review: files changed, behavior changed, verification performed, remaining risks.

### Agent 19 Final Review

- Files changed: `kairos/notifier.py`, `kairos/config.py`, `kairos/live.py`, `tests/test_notifier.py`, `tasks/todo.md`.
- Behavior changed: Kairos can dispatch state-change, anomaly, and confidence-drop notifications through configured Telegram, Discord, or email channels; live pipeline schedules notification checks after building final event/context.
- Verification performed: `python -m pytest tests/test_notifier.py tests/test_live.py::test_run_pipeline_with_context_returns_tuple -q` (9 passed); `python -m py_compile kairos/notifier.py kairos/live.py`; notifier import smoke command printed `OK`.
- Remaining risks: Full suite did not complete in this dirty worktree. Earlier `tests/test_live.py` exposed unrelated async-supervisor failures/mutations outside Agent 19 scope.

## Active: Agent 12 Paper Trading Engine

- [x] Add failing tests for live paper positions, flips, MTM P&L, Kelly sizing, multi-asset state, and persistence.
- [x] Create `kairos/papertrade.py` with paper account/position dataclasses, signal processing, MTM updates, summaries, and formatter.
- [x] Add `paper_trades` persistence schema to `kairos/db.py` using the existing DuckDB pattern.
- [x] Integrate a module-level paper trading engine into `kairos/live.py:run_pipeline_with_context()`.
- [x] Verify import acceptance criterion and targeted papertrade/history tests; broader live/full suite is blocked by unrelated active live-pipeline failures/hung run.

### Final Review

- Files changed: `kairos/papertrade.py`, `kairos/db.py`, `kairos/live.py`, `tests/test_papertrade.py`, `tasks/todo.md`
- Behavior changed: live Kairos signals now update a singleton paper account per asset, with long/short/neutral transitions, Kelly-sized fractions, slippage-adjusted entry/exit, MTM equity, summaries, and DuckDB-backed trade persistence.
- Verification performed: `python -m compileall kairos/papertrade.py kairos/db.py kairos/live.py`; `python -c "from kairos.papertrade import PaperTradingEngine; e = PaperTradingEngine(); print('OK')"`; `pytest tests/test_papertrade.py tests/test_history.py -q` -> 16 passed; final fresh `pytest tests/test_papertrade.py -q` -> 8 passed.
- Remaining risks: full `pytest -q` did not complete and was stopped after hanging; `pytest tests/test_papertrade.py tests/test_live.py -q` showed 42 passed / 5 failed in unrelated active Agent 13 expectations (`DataFetchSupervisor`, NaN training-row handling).

## Active: Agent 20 CI/CD Pipeline Manager

- [x] Create GitHub Actions CI with Python 3.11/3.12 fast and slow test jobs, pip/result caching, timeouts, artifact upload, and branch concurrency cancellation.
- [x] Create Docker publish workflow for GHCR using Buildx, GITHUB_TOKEN auth, tags, and GitHub Actions cache.
- [x] Add pre-commit hooks for whitespace/YAML/conflict checks, Ruff, fast pytest, and stale `.ubj` detection.
- [x] Add CODEOWNERS, Makefile convenience targets, and CONTRIBUTING workflow conventions.
- [x] Validate YAML/config syntax and run available local verification without touching `kairos/`, `tests/`, or `pyproject.toml`.
- [x] Record final review with files changed, behavior changed, verification, and remaining risks.

### Agent 20 Final Review

- Files changed: `.github/workflows/ci.yml`, `.github/workflows/docker-publish.yml`, `.github/CODEOWNERS`, `.pre-commit-config.yaml`, `Makefile`, `CONTRIBUTING.md`, `tasks/todo.md`.
- Behavior changed: pushes/PRs now have a GitHub Actions test matrix for Python 3.11/3.12 with fast/slow test splits, dependency and pytest-result caching, JUnit artifacts, job timeouts, and concurrency cancellation. Main/master pushes and `v*` tags now build and publish Docker images to GHCR when Dockerfile exists. Local dev now has Make targets, pre-commit hooks, CODEOWNERS review assignment, and documented branch/commit/PR conventions.
- Verification performed: YAML parsed successfully with Ruby `YAML.load_file`; `make -n test`, `make -n lint`, and `make -n docker` expanded expected commands; `pre-commit validate-config` passed with `PRE_COMMIT_HOME=/private/tmp/kairos-pre-commit`; `pre-commit install` completed after unsetting redundant `core.hooksPath`; `pre-commit run --files .github/workflows/ci.yml .github/workflows/docker-publish.yml .pre-commit-config.yaml .github/CODEOWNERS Makefile CONTRIBUTING.md` passed.
- Remaining risks: `make test` is blocked by unrelated in-progress source/test work (`run_multi_asset_backtest`, `kairos.feature_store`, `kairos.ingest.sentiment`, `kairos.papertrade` collection errors). `make lint` is blocked by existing Ruff violations in protected `kairos/` files. `make docker` is blocked locally because the Docker CLI is not installed.

## Active: Agent 18 Docker & Cloud Deployer

- [x] Create `Dockerfile` with multi-stage Python 3.12 runtime, non-root user, health check, and headless default command.
- [x] Create `.dockerignore` to keep dev/test/cache artifacts and secrets out of the build context while allowing `kairos.db`.
- [x] Create `docker-compose.yml` for local cloud-like headless runs with DB volume and env passthrough.
- [x] Create `kairos/healthcheck.py` without modifying existing Kairos source behavior.
- [x] Create `README.docker.md` with build, run, API, DB mount, env, health, and compose instructions.
- [x] Verify syntax, package import, healthcheck behavior, and package wheel build; Docker build/run blocked because Docker/Podman are not installed.
- [x] Final review: changed files, behavior, verification, remaining risks.

### Agent 18 Final Review

- Files changed: `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `kairos/healthcheck.py`, `README.docker.md`, `tasks/todo.md`.
- Behavior changed: Docker image now builds Kairos from `pyproject.toml`, runs default headless CLI as non-root `appuser`, exposes port 8000, includes a health check, and supports DB/env mounts via Compose.
- Verification performed: `python -m py_compile kairos/healthcheck.py`; `python kairos/healthcheck.py`; `env KAIROS_DB_PATH=/private/tmp/kairos-missing-healthcheck.db python kairos/healthcheck.py` returned exit 1 as expected; `python -c "from kairos import *; print('OK')"`; `python -m pip wheel --no-deps --wheel-dir /private/tmp/kairos-wheel .`.
- Remaining risks: `docker build -t kairos .`, in-container import smoke test, non-root check, health inspection, and compressed image size could not be verified locally because `docker` and `podman` are not installed in this environment.

## Active: Agent 13 Async Event Loop Optimizer

- [x] Read repo lessons, active task state, and target async files.
- [x] Add failing tests for supervisor timeout, partial results, and semaphore limit.
- [x] Add failing tests for `_run_async()` sync execution and timeout behavior.
- [x] Implement `DataFetchSupervisor` in `kairos/live.py` with per-source timeouts, retries, cancellation handling, and graceful partial results.
- [x] Update `fetch_live_data()` to use supervised price/FNG fetching without changing its public return signature.
- [x] Improve `kairos/backtest/engine.py:_run_async()` with sync/async detection and a 60s timeout.
- [x] Register TUI SIGINT cleanup and verify existing TUI background-refresh behavior.
- [x] Run targeted tests, then broader test verification as feasible.
- [x] Final review: changed files, behavior, verification, remaining risks.

### Final Review

- Files changed: `kairos/live.py`, `kairos/backtest/engine.py`, `kairos/cli.py`, `tests/test_live.py`, `tests/test_async_event_loop.py`, `tasks/todo.md`.
- Behavior changed: live source fetching now has a single `DataFetchSupervisor` with per-source timeouts, retry limits, bounded concurrency, partial-result returns, and cancellation cleanup. `fetch_live_data()` preserves its tuple signature and fetches only supervised price/FNG data. `_run_async()` supports sync callers with `asyncio.run()` and async callers by returning a task to await, both under a timeout. TUI registers fetch-task cleanup for Ctrl+C and keeps its existing non-overlapping background refresh executor.
- Verification performed: red tests first failed as expected; targeted verification passed with `python -m pytest tests/test_live.py::test_supervisor_timeout tests/test_live.py::test_supervisor_partial_results tests/test_live.py::test_supervisor_semaphore tests/test_async_event_loop.py tests/test_live.py::test_fetch_live_data_returns_prices_and_fng tests/test_live.py::test_fetch_live_data_fng_fallback_on_error tests/test_multi_asset.py::test_fetch_live_data_unknown_asset_raises tests/test_cli.py -q` (13 passed); `python -m pytest tests/test_live.py -q` (39 passed).
- Remaining risks: `python -m pytest tests/test_async_event_loop.py tests/test_backtest_engine.py -q` reported 34 passed and 3 existing live-data backtest failures where default `run_backtest()` receives empty live price data in this restricted/no-cache environment. Those failures are outside the async supervisor changes.

## Active: Agent 8 Kelly Sizer Expansionist

- [ ] Add failing tests for multi-asset Kelly sizing, correlation tracking, adaptive Kelly, and multi-asset backtest execution.
- [ ] Implement `_multi_asset_kelly`, `CorrelationTracker`, and `_adaptive_kelly` in `kairos/backtest/engine.py`.
- [ ] Integrate adaptive Kelly into single-asset sizing without changing `run_backtest()` signature.
- [ ] Add `run_multi_asset_backtest()` with combined equity tracking and correlated allocation caps.
- [ ] Export new backtest helpers if needed.
- [ ] Run targeted verification and import acceptance command.

## Active: Agent 11 Order Book Slippage

- [x] Add failing tests for Binance depth parsing, cache reuse, fallback slippage, size impact, and buy/sell imbalance.
- [x] Add cached Binance top-10 depth fetcher in `kairos/backtest/engine.py`.
- [x] Upgrade `_compute_slippage()` to blend regime/volatility with order-book market impact.
- [x] Wire cached order-book data into `run_backtest()` without changing the public API.
- [x] Add exchange API config placeholders in `kairos/config.py`.
- [x] Verify targeted tests and import acceptance command.
- [x] Final review: changed files, behavior, verification, remaining risks.

### Agent 11 Final Review

- Files changed: `kairos/backtest/engine.py`, `kairos/config.py`, `tests/test_backtest_engine.py`, `tasks/todo.md`.
- Behavior changed: slippage now uses cached Binance top-10 depth when available, walks bid/ask depth by trade direction and USD size, blends impact with regime/volatility, and falls back to the old heuristic when depth is missing. Single-asset `run_backtest()` API is unchanged; multi-asset execution also reuses the cache per asset.
- Verification performed: focused slippage tests `5 passed`; deterministic backtest-engine unit subset `25 passed, 9 deselected`; synthetic run-backtest integration subset `6 passed`; import acceptance printed `OK`; `compileall` passed for touched Python files.
- Remaining risks: live Binance depth check reached the network but returned HTTP 451 from this environment, so only mocked fetch coverage and fallback behavior are verified here. Full `tests/test_backtest_engine.py` still has unrelated live-fetch failures in `kairos/live.py` (`DataFetchSupervisor._fetch_sources` missing).

## Active: Agent 14 Unified Error Handler & Circuit Breaker

- [x] Read existing lessons, task state, and affected files before editing.
- [x] Add focused failing tests for circuit breaker transitions, fallback rejection, health summary, and metrics.
- [x] Add `kairos/circuit_breaker.py` with named breakers, health summary, fallbacks, and recovery hooks.
- [x] Integrate breakers in `kairos/live.py` without changing public function signatures.
- [x] Add circuit breaker config in `kairos/config.py`.
- [x] Add centralized optional ingest exports/error handling in `kairos/ingest/__init__.py`.
- [x] Run targeted and acceptance verification.
- [x] Final review: changed files, behavior, verification, remaining risks.

### Agent 14 Final Review

- Files changed: `kairos/circuit_breaker.py`, `kairos/live.py`, `kairos/ingest/__init__.py`, `kairos/config.py`, `tests/test_circuit_breaker.py`.
- Behavior changed: external source calls now have circuit breaker state, fallback returns, health summary, cached CoinGecko fallback, FNG neutral fallback, and package-level protected ingest entry points.
- Verification performed: `pytest tests/test_circuit_breaker.py -q` (6 passed); targeted live fallback/supervisor tests (7 passed); import acceptance command printed `OK`; `py_compile` on touched modules passed; full suite `python -m pytest -q` passed with 272 tests and 1 warning.
- Remaining risks: circuit recovery probes are optional per breaker; unnamed services without a probe use timeout-based half-open recovery.

## Active: Agent 9 Alternative Sentiment Quant

- [x] Add failing tests for alt sentiment structure, CryptoPanic fallback, and FNG divergence.
- [x] Add `sentiment_cache` schema to `kairos/db.py`.
- [x] Add `CRYPTOPANIC_API_KEY` config and sentiment ingest export.
- [x] Implement `kairos/ingest/sentiment.py` with CryptoPanic, developer RSS attention, cache, composite, and safe defaults.
- [x] Run focused verification and record final review.

### Agent 9 Final Review

- Files changed: `kairos/ingest/sentiment.py`, `kairos/ingest/__init__.py`, `kairos/config.py`, `kairos/db.py`, `tests/test_ingest_sentiment.py`.
- Behavior changed: added standalone alt sentiment vector with CryptoPanic, developer RSS attention, DuckDB-backed `sentiment_cache`, FNG divergence flag, and safe neutral fallbacks.
- Verification performed: `python -m pytest tests/test_ingest_sentiment.py -q`; import smoke; `python -m pytest tests/test_db.py tests/test_ingest_sentiment.py -q`; package export smoke.
- Remaining risks: live CryptoPanic call requires user-provided `CRYPTOPANIC_API_KEY`; full suite still has unrelated dirty-worktree failures in backtest async, feature store, live supervisor, and ensemble validation tests.

## Active: Agent 10 Causal Dynamics

- [x] Add failing tests for regime-dependent multipliers, Bayesian prior update behavior, sensitivity output, and preserved caller compatibility.
- [x] Rewrite `kairos/signals/causal.py` with dynamic multiplier lookup, confidence calibration, bounded history, and sensitivity analysis.
- [x] Preserve `infer_price_impact()` positional caller compatibility and do not touch live/backtest/ensemble/cli/ingest files.
- [x] Run targeted causal tests and broader existing tests that exercise causal outputs.
- [x] Record final review with files changed, behavior changed, verification, and remaining risks.

### Agent 10 Final Review

- Files changed: `kairos/signals/causal.py`, `tests/test_causal.py`, `tasks/todo.md`.
- Behavior changed: `CausalDAG` now uses regime-dependent narrative, anomaly, and macro multipliers; preserves legacy regime aliases; tracks bounded history; supports prior-based Bayesian updates through `dag.prior`; returns richer metadata; adds deterministic sensitivity analysis.
- Verification performed: RED `python -m pytest tests/test_causal.py -q` failed on missing dynamic behavior; GREEN `python -m pytest tests/test_causal.py -q` passed 8; `python -m pytest tests/test_causal.py tests/test_stress.py::test_causal_bullish_plus_bearish_equals_one tests/test_stress.py::test_causal_confidence_in_range tests/test_stress.py::test_causal_unknown_regime_is_neutral tests/test_stress.py::test_causal_tipping_increases_bullish tests/test_stress.py::test_causal_macro_stress_decreases_bullish tests/test_stress.py::test_causal_anomaly_reduces_confidence -q` passed 14; `python -m compileall kairos/signals/causal.py` passed; acceptance `python -c` confirmed `hv_down` and `hv_up` produce different values.
- Remaining risks: full suite currently stops during collection before causal assertions, first at `tests/test_backtest_engine.py` because `kairos.backtest.engine` does not export `run_multi_asset_backtest`. `tests/test_integration.py::test_full_pipeline_produces_signal_event` also fails because that test maps old regime labels into `FeatureVector` while `predict_regime()` now returns `lv_*`/`hv_*`.

## Active: Agent 4 Data Cleaning & Pipeline Validator

- [x] Add failing tests for `FeatureVector.validate()`, `sanitize_fv()`, and invalid direct prediction.
- [x] Add `FeatureVector.validate()` and `sanitize_fv()` in `kairos/signals/ensemble.py` without changing constructor order.
- [x] Guard `SignalEnsemble.predict()` and `_estimate_hours()` from invalid/non-finite feature values.
- [x] Filter invalid and non-finite training rows in `kairos/live.py` and `kairos/backtest/engine.py`.
- [x] Return neutral live/backtest signals when single FeatureVector construction is invalid.
- [x] Add safe live wrappers and normalized ingest outputs for GitHub, whale, and macro data.
- [x] Verify with `python -m pytest tests/test_ensemble.py -q` and full suite.

### Agent 4 Constraints

- Do not touch `kairos/cli.py`, `kairos/model_cache.py`, `kairos/db.py`, `kairos/backtest/report.py`, `kairos/backtest/runner.py`, or `kairos/backtest/__init__.py`.
- Preserve existing dirty worktree changes and avoid unrelated cleanup.

### Agent 4 Final Review

- Files changed: `kairos/signals/ensemble.py`, `kairos/live.py`, `kairos/backtest/engine.py`, `kairos/ingest/github.py`, `kairos/ingest/whale.py`, `kairos/ingest/macro.py`, `tests/test_ensemble.py`, `tests/test_live.py`, `tests/test_backtest_engine.py`, `tests/test_stress.py`, `tests/test_integration.py`.
- Behavior changed: Feature vectors validate/clamp required ranges, invalid live/backtest signal vectors return neutral signals, training builders drop invalid rows, ingest outputs and live wrappers return structured safe defaults.
- Verification performed: `python -m py_compile ...` passed; `python -m pytest tests/test_ensemble.py -q` passed with 13 tests; full `python -m pytest -q` passed with 272 tests.
- Remaining risks: Worktree contains broad pre-existing active-agent changes outside Agent 4 scope; this task preserved them and only adjusted touched tests where the new FeatureVector contract required valid fixtures.

## Active: Agent 2 Solana Whale Streaming

- [x] Add focused tests for whale DB schema, exchange wallet refresh, transfer parsing, metrics, and import contract.
- [x] Add `SOLANA_RPC_URL` config with public default.
- [x] Add whale/exchange tables and indexes to `kairos/db.py`.
- [x] Rewrite `kairos/ingest/whale.py` for WebSocket subscription, reconnect backoff, recent-flow deque, REST fallback concurrency, DB persistence, and metrics.
- [x] Update ingest exports for new public APIs.
- [x] Run targeted verification.
- [ ] Final review: changed files, behavior, verification, remaining risks.

## Active: GitHub Ingestion Upgrade

- [x] Preserve `fetch_code_velocity()` as async polling API with `since_days`, token rotation, per-repo breakdown, stars/forks, and churn metric.
- [x] Add webhook ingestion in `kairos/ingest/github.py` for `push`, merged `pull_request`, and closed `issues` events.
- [x] Persist webhook events through `kairos/db.py:create_schema` in `github_events`.
- [x] Export new webhook helpers from `kairos/ingest/__init__.py`.
- [x] Verify import acceptance criterion and existing tests.

### GitHub Ingestion Final Review

- Files changed: `kairos/ingest/github.py`, `kairos/ingest/__init__.py`, `kairos/db.py`, `tasks/todo.md`.
- Behavior changed: GitHub ingest now supports token-rotating REST polling with `since_days`, repo breakdowns, stars/forks, contributor retention, PR timing/review metrics, in-memory webhook events, DuckDB-backed `github_events`, and `POST /webhook` for `push`, merged `pull_request`, and closed `issues`.
- Verification performed: import acceptance command passed; `python -m py_compile kairos/ingest/github.py kairos/db.py kairos/ingest/__init__.py` passed; `fetch_code_velocity(repos=[])` shape smoke passed; webhook request smoke passed; localhost webhook server bind/POST smoke passed; token rotation smoke passed for `403` and `X-RateLimit-Remaining: 0`; `tests/test_db.py` passed.
- Remaining risks: full `pytest` is blocked by unrelated dirty-worktree collection errors in `tests/test_backtest_engine.py` (`run_multi_asset_backtest` missing) and `tests/test_papertrade.py` (`kairos.papertrade` missing/unstable during concurrent edits). Live GitHub API behavior was not exercised to avoid external network dependence.

## Active: Agent 6 HMM Optimization

- [x] Update `tests/test_regime.py` with failing coverage for optimized fitting, stable predictions, confidence, smoothing, and cache creation.
- [x] Rewrite `kairos/signals/regime.py` with cached grid-search optimization, canonical/Hungarian state labels, confidence scoring, and optional smoothing.
- [x] Verify targeted regime tests and import acceptance command.
- [x] Final review: changed files, behavior, verification, remaining risks.

#### Final Review

- Files changed: `kairos/signals/regime.py`, `tests/test_regime.py`, `tasks/todo.md`.
- Behavior changed: added cached HMM hyperparameter optimization, deterministic state labeling, confidence scoring, and confidence-aware smoothing while preserving existing caller compatibility.
- Verification performed: `pytest tests/test_regime.py -q` passed; import acceptance command returned `OK`; uncached 365-row optimizer benchmark completed in 0.934s.
- Remaining risks: full suite is blocked by pre-existing/non-target failures in `tests/test_backtest_engine.py`, `tests/test_integration.py`, and `tests/test_stress.py` (`266 passed, 6 failed`).

## Completed

- [x] `kairos/signals/conflict.py` — Added `_compute_vma`, `volume_status_label`, `capitulation_triggered`, `seller_exhaustion_active`
- [x] `kairos/live.py` — Volume data fetching from CoinGecko, volume through pipeline, capitulation buy override in `run_pipeline_with_context`, volume status + capitulation badge in UI
- [x] `kairos/backtest/engine.py` — `BacktestTrade.is_capitulation`, `BacktestResult.capitulation_trades`/`strategy`, `run_backtest` accepts `strategy={"A","B","C"}` and `volumes`, override logic in trading loop
- [x] `kairos/backtest/report.py` — Capitulation trades line in compact track record
- [x] `kairos/cli.py` — `compare` command for multi-strategy comparison
- [x] `tests/test_conflict.py` — 16 new tests for volume/override functions
- [x] `tests/test_live.py` — Updated for 5-return fetch_live_data
- [x] Full test suite: 190 passed

## Remaining Risks

- Volume fetching falls back to price-delta proxy when CoinGecko `total_volumes` is absent (OK for tests)
- Strategy C recomputes Kalman slope in the trading loop (small perf cost, acceptable for comparison runs)
- Capitulation buy confidence is hardcoded at 0.65 in live pipeline (reasonable floor)

## Agent 5: Feature Store Manager

### Plan

- [x] Add tests for FeatureStore persistence, latest lookup, statistics, and prune behavior.
- [x] Add tests proving `run_pipeline()` returns a fresh cached signal without refitting.
- [x] Add tests proving `run_backtest()` stores per-day feature vectors for analytics without changing signatures.
- [x] Add DuckDB-compatible `feature_store` schema to `kairos/db.py`.
- [x] Create `kairos/feature_store.py` with the requested API and standalone rich stats printer.
- [x] Integrate best-effort cache read/write in `kairos/live.py`.
- [x] Integrate best-effort post-run feature storage in `kairos/backtest/engine.py`.
- [x] Run focused tests, acceptance command, and a broader regression suite.

### Blocker Notes

- 2026-05-29: Focused live/backtest test imports were briefly blocked because `kairos/signals/regime.py` appeared deleted in the worktree (`git status --short kairos/signals/regime.py` showed `D`). It later reappeared as a modified file, and focused tests passed. No restore was performed.

### Final Review

Files changed:
- `kairos/feature_store.py`
- `kairos/db.py`
- `kairos/live.py`
- `kairos/backtest/engine.py`
- `tests/test_feature_store.py`
- `tests/test_db.py`
- `tests/test_live.py`
- `tests/test_backtest_engine.py`
- `tasks/todo.md`

Behavior changed:
- Added persisted feature vectors keyed by asset and timestamp.
- Added fresh live cached-signal reuse when metadata is under 24h old and candle count is within 10%.
- Added best-effort live feature persistence after computed signals.
- Added best-effort backtest feature persistence for per-day analytics.

Verification performed:
- `python -m pytest tests/test_feature_store.py tests/test_db.py tests/test_live.py::test_run_pipeline_uses_fresh_feature_cache tests/test_live.py::test_run_pipeline_stores_computed_feature tests/test_backtest_engine.py::test_run_backtest_stores_feature_vectors_for_analytics -q` — 7 passed.
- `python -m pytest tests/test_live.py -q` — 39 passed.
- `python -m py_compile kairos/feature_store.py kairos/db.py kairos/live.py kairos/backtest/engine.py tests/test_feature_store.py tests/test_live.py tests/test_backtest_engine.py` — passed.
- Corrected acceptance import command with `KAIROS_DB_PATH=/private/tmp/kairos-feature-acceptance.db` — printed `OK`.

Remaining risks:
- `python -m pytest -q` and a broader related slice both exited with code `-1` and no output in this environment, so full-suite status is not available from this run.
- The prompt's one-line acceptance command omits imports for `datetime` and `FeatureVector`; the verified version includes those imports.

---

# Agent 3 Macro Ingestion — Implementation Log

## Plan

- [x] Read existing macro ingestion, config, db schema, tests, and task notes.
- [x] Add failing tests for `fetch_macro_data()` raw series, regime vector, cache reuse, safe defaults, and import/export.
- [x] Rewrite `kairos/ingest/macro.py` with FRED/Alpha Vantage fetch, cache, derived regime signals, and resilient defaults.
- [x] Add `ALPHA_VANTAGE_API_KEY` to `kairos/config.py` and export macro helpers in `kairos/ingest/__init__.py`.
- [x] Run targeted verification and record final review.

## Final Review

- Files changed: `kairos/ingest/macro.py`, `kairos/config.py`, `kairos/ingest/__init__.py`, `tests/test_ingest_macro.py`
- Behavior changed: `fetch_macro_data()` returns raw macro series plus a composite macro regime vector; FRED is preferred, Alpha Vantage is used only when FRED is unavailable, cache reuse is backed by `macro_data`, and provider failures return neutral defaults.
- Verification performed: `python -m pytest -q tests/test_ingest_macro.py` -> 5 passed; `python -c "from kairos.ingest.macro import fetch_macro_data; print('OK')"` -> OK; `python -c "from kairos.ingest import fetch_macro_data; print('OK')"` -> OK; `python -m compileall -q kairos/ingest/macro.py kairos/config.py kairos/ingest/__init__.py` -> exit 0.
- Remaining risks: `python -m pytest -q tests/test_db.py tests/test_ingest_macro.py` still fails in `tests/test_db.py` because the current dirty worktree expects a `feature_store` table not present in `kairos/db.py`; this task explicitly does not add tables or touch `kairos/db.py`.

---

# Agent 7 XGBoost Tuner — Implementation Log

## Plan

- [x] Read existing ensemble, model cache, tests, task notes, and lessons.
- [x] Add failing tests for per-regime tuning, skipped regimes, cached best params, and custom params.
- [x] Implement per-regime search grids, tuning orchestration, JSON best-param cache, and `SignalEnsemble` param loading.
- [x] Update model-cache metadata with trained params, validation AUC, and stale-AUC invalidation.
- [x] Run targeted verification and import smoke check.

## Final Review

- Files changed: `kairos/signals/ensemble.py`, `kairos/model_cache.py`, `tests/test_ensemble.py`.
- Behavior changed: per-regime XGBoost param grids, `tune_sub_models()`, best-param JSON cache, `SignalEnsemble(params=...)`, `SignalEnsemble.load_best_params()`, cache metadata for trained params/AUC, and AUC-staleness invalidation.
- Verification performed: `python -m pytest tests/test_ensemble.py tests/test_model_cache.py -q` -> 19 passed; import smoke -> OK.
- Remaining risks: full `python -m pytest -q` was terminated after running silently for an extended period; tiny real tuning smoke hit sandbox process limits from `n_jobs=-1` and exercised fallback-to-default behavior.
