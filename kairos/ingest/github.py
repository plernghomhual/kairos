"""GitHub developer activity ingestion for code velocity signals."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from kairos.config import DB_PATH, GITHUB_TOKEN
from kairos.db import create_schema, get_connection

_REPOS = [
    "solana-labs/solana",
    "ethereum/go-ethereum",
    "paritytech/polkadot-sdk",
    "MystenLabs/sui",
    "aptos-labs/aptos-core",
]

BASE = "https://api.github.com"
_USER_AGENT = "kairos/0.1"
_LOW_RATE_LIMIT_THRESHOLD = 100
_MAX_RETRIES = 4
_EVENT_TYPES = {"push", "pull_request", "issues"}
_RECENT_EVENTS: deque[dict[str, Any]] = deque(maxlen=10_000)
_EVENTS_LOCK = asyncio.Lock()
_LOG = logging.getLogger(__name__)


class _TokenRotator:
    """Rotate GitHub tokens under a lock when rate limits are encountered."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = [token.strip() for token in tokens if token and token.strip()]
        self._index = 0
        self._lock = asyncio.Lock()

    @property
    def has_tokens(self) -> bool:
        return bool(self._tokens)

    def current(self) -> str | None:
        if not self._tokens:
            return None
        return self._tokens[self._index]

    async def rotate(self) -> str | None:
        async with self._lock:
            if not self._tokens:
                return None
            self._index = (self._index + 1) % len(self._tokens)
            return self._tokens[self._index]


def _default_code_velocity() -> dict[str, Any]:
    return {
        "available": False,
        "commit_velocity": 0,
        "contributor_count": 0,
        "merged_prs": 0,
        "stargazer_count": 0,
        "fork_count": 0,
        "code_churn_velocity": 0.0,
        "pr_merge_time": 0.0,
        "review_depth": 0.0,
        "contributor_retention": 0.0,
        "issue_close_rate": 0.0,
        "repo_breakdown": {},
        "repos_scraped": [],
        "since_days": 7,
        "fetch_ts": datetime.now(timezone.utc).isoformat(),
    }


def _safe_nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not result == result or result in (float("inf"), float("-inf")):
        return default
    return max(result, 0.0)


def _sanitize_code_velocity_result(result: dict[str, Any]) -> dict[str, Any]:
    clean = _default_code_velocity()
    clean["commit_velocity"] = int(_safe_nonnegative_float(result.get("commit_velocity")))
    clean["contributor_count"] = int(_safe_nonnegative_float(result.get("contributor_count")))
    clean["merged_prs"] = int(_safe_nonnegative_float(result.get("merged_prs")))
    clean["stargazer_count"] = int(_safe_nonnegative_float(result.get("stargazer_count")))
    clean["fork_count"] = int(_safe_nonnegative_float(result.get("fork_count")))
    clean["code_churn_velocity"] = _safe_nonnegative_float(result.get("code_churn_velocity"))
    clean["pr_merge_time"] = _safe_nonnegative_float(result.get("pr_merge_time"))
    clean["review_depth"] = _safe_nonnegative_float(result.get("review_depth"))
    clean["contributor_retention"] = min(_safe_nonnegative_float(result.get("contributor_retention")), 1.0)
    clean["issue_close_rate"] = min(_safe_nonnegative_float(result.get("issue_close_rate")), 1.0)
    clean["repo_breakdown"] = result.get("repo_breakdown") if isinstance(result.get("repo_breakdown"), dict) else {}
    repos_scraped = result.get("repos_scraped")
    clean["repos_scraped"] = list(repos_scraped) if isinstance(repos_scraped, list) else []
    clean["since_days"] = max(int(_safe_nonnegative_float(result.get("since_days"), 7.0)), 1)
    clean["fetch_ts"] = str(result.get("fetch_ts") or clean["fetch_ts"])
    clean["available"] = bool(clean["repos_scraped"])
    return clean


async def fetch_code_velocity(
    repos: list[str] | None = None,
    tokens: list[str] | None = None,
    since_days: int = 7,
) -> dict[str, Any]:
    """Fetch repository code velocity and recent webhook-derived metrics."""
    targets = _REPOS if repos is None else repos
    now = datetime.now(timezone.utc)
    since_days = max(since_days, 1)
    since = now - timedelta(days=since_days)
    retention_since = now - timedelta(days=28)
    commit_since = min(since, retention_since)
    token_rotator = _TokenRotator(_configured_tokens(tokens))

    repo_breakdown: dict[str, dict[str, Any]] = {}
    total_commits = 0
    total_merged_prs = 0
    total_stars = 0
    total_forks = 0
    total_contributors: set[str] = set()
    retained_this_week: set[str] = set()
    retained_four_weeks: set[str] = set()
    repos_scraped: list[str] = []

    event_metrics = _calculate_event_metrics(
        await get_recent_events(minutes=since_days * 24 * 60),
        period_days=since_days,
    )

    async with httpx.AsyncClient(timeout=15.0) as client:
        for repo in targets:
            try:
                metrics = await _fetch_repo_metrics(
                    client=client,
                    repo=repo,
                    token_rotator=token_rotator,
                    since=since,
                    commit_since=commit_since,
                    retention_since=retention_since,
                    now=now,
                )
            except Exception as exc:
                _LOG.warning("GitHub scrape failed for %s: %s", repo, exc)
                continue

            repo_event_metrics = event_metrics["by_repo"].get(repo, {})
            for key, value in repo_event_metrics.items():
                if value is not None:
                    metrics[key] = value
            repo_breakdown[repo] = metrics
            repos_scraped.append(repo)

            total_commits += metrics["commit_velocity"]
            total_merged_prs += metrics["merged_prs"]
            total_stars += metrics["stargazer_count"]
            total_forks += metrics["fork_count"]
            total_contributors.update(metrics["_contributors"])
            retained_this_week.update(metrics["_contributors_this_week"])
            retained_four_weeks.update(metrics["_contributors_four_weeks"])

    for metrics in repo_breakdown.values():
        metrics.pop("_contributors", None)
        metrics.pop("_contributors_this_week", None)
        metrics.pop("_contributors_four_weeks", None)

    return _sanitize_code_velocity_result(
        {
            "commit_velocity": total_commits,
            "contributor_count": len(total_contributors),
            "merged_prs": total_merged_prs,
            "stargazer_count": total_stars,
            "fork_count": total_forks,
            "code_churn_velocity": event_metrics["code_churn_velocity"],
            "pr_merge_time": event_metrics["pr_merge_time"]
            if event_metrics["pr_merge_time"] is not None
            else _average(
                [
                    metrics["pr_merge_time"]
                    for metrics in repo_breakdown.values()
                    if metrics["pr_merge_time"] is not None
                ]
            ),
            "review_depth": event_metrics["review_depth"]
            if event_metrics["review_depth"] is not None
            else _average(
                [metrics["review_depth"] for metrics in repo_breakdown.values() if metrics["review_depth"] is not None]
            ),
            "contributor_retention": _ratio(
                len(retained_this_week),
                len(retained_four_weeks),
            ),
            "issue_close_rate": event_metrics["issue_close_rate"],
            "repo_breakdown": repo_breakdown,
            "repos_scraped": repos_scraped,
            "since_days": since_days,
            "fetch_ts": datetime.now(timezone.utc).isoformat(),
        }
    )


async def start_webhook_server(host: str = "0.0.0.0", port: int = 8080) -> dict[str, Any] | None:
    """Start an async HTTP listener for GitHub webhook POST /webhook events."""
    try:
        server = await asyncio.start_server(_handle_webhook_client, host, port)
    except OSError as exc:
        _LOG.warning(
            "GitHub webhook server failed to bind %s:%s; falling back to polling: %s",
            host,
            port,
            exc,
        )
        return await fetch_code_velocity()

    sockets = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
    _LOG.info("GitHub webhook server listening on %s", sockets)
    async with server:
        await server.serve_forever()
    return None


async def get_recent_events(minutes: int = 60) -> list[dict[str, Any]]:
    """Return recent in-memory GitHub webhook events."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(minutes, 0))
    async with _EVENTS_LOCK:
        events = list(_RECENT_EVENTS)
    return [
        event
        for event in events
        if (_parse_datetime(event.get("received_at")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff
    ]


async def _fetch_repo_metrics(
    client: httpx.AsyncClient,
    repo: str,
    token_rotator: _TokenRotator,
    since: datetime,
    commit_since: datetime,
    retention_since: datetime,
    now: datetime,
) -> dict[str, Any]:
    repo_info = await _request_json(client, f"/repos/{repo}", token_rotator)
    commits = await _paginated_get(
        client,
        f"/repos/{repo}/commits",
        token_rotator,
        params={"since": _github_ts(commit_since), "per_page": 100},
        max_pages=5,
    )
    prs = await _paginated_get(
        client,
        f"/repos/{repo}/pulls",
        token_rotator,
        params={
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": 100,
        },
        max_pages=3,
    )

    period_commits = []
    contributors: set[str] = set()
    contributors_this_week: set[str] = set()
    contributors_four_weeks: set[str] = set()

    for commit in commits:
        committed_at = _commit_timestamp(commit)
        login = _commit_login(commit)
        if committed_at is None:
            continue
        if login and committed_at >= retention_since:
            contributors_four_weeks.add(login)
        if login and committed_at >= now - timedelta(days=7):
            contributors_this_week.add(login)
        if committed_at >= since:
            period_commits.append(commit)
            if login:
                contributors.add(login)

    merged_prs = [_pr for _pr in prs if _is_recent_merge(_pr, since)]
    merge_hours = [_merge_hours(pr) for pr in merged_prs]
    merge_hours = [hours for hours in merge_hours if hours is not None]
    review_counts = [int(pr.get("review_comments") or 0) for pr in merged_prs if pr.get("review_comments") is not None]

    return {
        "commit_velocity": len(period_commits),
        "contributor_count": len(contributors),
        "merged_prs": len(merged_prs),
        "stargazer_count": int(repo_info.get("stargazers_count") or 0),
        "fork_count": int(repo_info.get("forks_count") or repo_info.get("fork_count") or 0),
        "pr_merge_time": _average(merge_hours),
        "review_depth": _average(review_counts),
        "contributor_retention": _ratio(
            len(contributors_this_week),
            len(contributors_four_weeks),
        ),
        "code_churn_velocity": 0.0,
        "issue_close_rate": None,
        "_contributors": contributors,
        "_contributors_this_week": contributors_this_week,
        "_contributors_four_weeks": contributors_four_weeks,
    }


async def _request_json(
    client: httpx.AsyncClient,
    path: str,
    token_rotator: _TokenRotator,
    params: dict[str, Any] | None = None,
) -> Any:
    url = f"{BASE}{path}"
    last_response: httpx.Response | None = None

    for attempt in range(_MAX_RETRIES):
        response = await client.get(
            url,
            params=params,
            headers=_headers(token_rotator.current()),
        )
        last_response = response
        rotated_for_rate_limit = await _handle_rate_headers(response, token_rotator)

        if response.status_code == 403:
            if token_rotator.has_tokens:
                if not rotated_for_rate_limit:
                    await token_rotator.rotate()
                continue
            response.raise_for_status()

        if response.status_code in {429, 503}:
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
                continue
            response.raise_for_status()

        response.raise_for_status()
        return response.json()

    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError(f"GitHub request failed after {_MAX_RETRIES} retries: {path}")


async def _paginated_get(
    client: httpx.AsyncClient,
    path: str,
    token_rotator: _TokenRotator,
    params: dict[str, Any],
    max_pages: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    per_page = int(params.get("per_page") or 100)
    for page in range(1, max_pages + 1):
        page_params = {**params, "page": page}
        payload = await _request_json(client, path, token_rotator, page_params)
        if not isinstance(payload, list):
            return results
        results.extend(payload)
        if len(payload) < per_page:
            break
    return results


async def _handle_rate_headers(
    response: httpx.Response,
    token_rotator: _TokenRotator,
) -> bool:
    remaining = _safe_int(response.headers.get("X-RateLimit-Remaining"))
    if remaining is None:
        return False

    if remaining == 0 and token_rotator.has_tokens:
        await token_rotator.rotate()
        return True

    if remaining < _LOW_RATE_LIMIT_THRESHOLD:
        await asyncio.sleep(_rate_limit_sleep_seconds(response))
    return False


def _rate_limit_sleep_seconds(response: httpx.Response) -> float:
    reset_at = _safe_int(response.headers.get("X-RateLimit-Reset"))
    if reset_at is None:
        return 1.0
    delay = reset_at - int(datetime.now(timezone.utc).timestamp())
    return float(min(max(delay, 1), 60))


_MAX_WEBHOOK_HEADER_BYTES = 8_192  # 8 KB — sufficient for any legitimate HTTP header block
_MAX_WEBHOOK_BODY_BYTES = 1_048_576  # 1 MB — GitHub payloads are well under this


async def _handle_webhook_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        header_bytes = await reader.readuntil(b"\r\n\r\n", limit=_MAX_WEBHOOK_HEADER_BYTES)
        request_line, headers = _parse_http_headers(header_bytes)
        method, path, _version = request_line.split(" ", 2)
        raw_cl = _safe_int(headers.get("content-length")) or 0
        content_length = min(raw_cl, _MAX_WEBHOOK_BODY_BYTES)
        body = await reader.readexactly(content_length) if content_length else b""
        status, response = await _process_webhook_request(method, path, headers, body)
    except Exception as exc:
        _LOG.warning("GitHub webhook request failed: %s", exc)
        status, response = 500, {"ok": False, "error": "webhook request failed"}

    _write_http_response(writer, status, response)
    writer.close()
    await writer.wait_closed()


def _verify_webhook_signature(body: bytes, signature_header: str | None) -> bool:
    secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        return True  # no secret configured — allow (warn at startup if desired)
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature_header)


async def _process_webhook_request(
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
) -> tuple[int, dict[str, Any]]:
    if method.upper() != "POST":
        return 405, {"ok": False, "error": "method not allowed"}
    if path.split("?", 1)[0] != "/webhook":
        return 404, {"ok": False, "error": "not found"}

    sig = headers.get("x-hub-signature-256")
    if not _verify_webhook_signature(body, sig):
        return 401, {"ok": False, "error": "invalid webhook signature"}

    event_type = headers.get("x-github-event", "")
    if event_type not in _EVENT_TYPES:
        return 202, {"ok": True, "ignored": True}

    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except json.JSONDecodeError:
        return 400, {"ok": False, "error": "invalid json"}

    if not _is_supported_payload(event_type, payload):
        return 202, {"ok": True, "ignored": True}

    await _record_event(event_type, payload)
    return 202, {"ok": True}


async def _record_event(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    event = {
        "event_type": event_type,
        "repo": _payload_repo(payload),
        "payload": payload,
        "received_at": datetime.now(timezone.utc).isoformat(),
    }
    async with _EVENTS_LOCK:
        _RECENT_EVENTS.append(event)

    try:
        await asyncio.to_thread(_persist_event, event)
    except Exception as exc:
        _LOG.warning("Failed to persist GitHub webhook event: %s", exc)
    return event


def _persist_event(event: dict[str, Any]) -> None:
    conn = get_connection(DB_PATH)
    try:
        create_schema(conn)
        conn.execute(
            """
            INSERT INTO github_events (event_type, repo, payload, received_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                event["event_type"],
                event["repo"],
                json.dumps(event["payload"]),
                event["received_at"],
            ],
        )
    finally:
        conn.close()


def _calculate_event_metrics(
    events: list[dict[str, Any]],
    period_days: int | float,
) -> dict[str, Any]:
    per_repo: dict[str, dict[str, Any]] = {}
    churn_total = 0
    merge_hours: list[float] = []
    review_counts: list[int] = []
    opened_issues = 0
    closed_issues = 0
    days = max(float(period_days), 1.0)

    for event in events:
        repo = str(event.get("repo") or "unknown")
        payload = _payload_dict(event.get("payload"))
        repo_metrics = per_repo.setdefault(
            repo,
            {
                "code_churn_velocity": 0.0,
                "pr_merge_time": None,
                "review_depth": None,
                "issue_close_rate": None,
                "_churn_total": 0,
                "_merge_hours": [],
                "_review_counts": [],
                "_opened_issues": 0,
                "_closed_issues": 0,
            },
        )

        if event.get("event_type") == "push":
            churn = _extract_push_churn(payload)
            churn_total += churn
            repo_metrics["_churn_total"] += churn
        elif event.get("event_type") == "pull_request":
            hours = _merge_hours(payload.get("pull_request", {}))
            if hours is not None:
                merge_hours.append(hours)
                repo_metrics["_merge_hours"].append(hours)
            review_count = _safe_int(payload.get("pull_request", {}).get("review_comments"))
            if review_count is not None:
                review_counts.append(review_count)
                repo_metrics["_review_counts"].append(review_count)
        elif event.get("event_type") == "issues":
            action = payload.get("action")
            if action == "opened":
                opened_issues += 1
                repo_metrics["_opened_issues"] += 1
            elif action == "closed":
                closed_issues += 1
                repo_metrics["_closed_issues"] += 1

    for repo_metrics in per_repo.values():
        repo_metrics["code_churn_velocity"] = repo_metrics["_churn_total"] / days
        repo_metrics["pr_merge_time"] = _average(repo_metrics["_merge_hours"])
        repo_metrics["review_depth"] = _average(repo_metrics["_review_counts"])
        repo_metrics["issue_close_rate"] = _ratio(
            repo_metrics["_closed_issues"],
            repo_metrics["_opened_issues"],
        )
        for key in (
            "_churn_total",
            "_merge_hours",
            "_review_counts",
            "_opened_issues",
            "_closed_issues",
        ):
            repo_metrics.pop(key, None)

    return {
        "code_churn_velocity": churn_total / days,
        "pr_merge_time": _average(merge_hours),
        "review_depth": _average(review_counts),
        "issue_close_rate": _ratio(closed_issues, opened_issues),
        "by_repo": per_repo,
    }


def _is_supported_payload(event_type: str, payload: dict[str, Any]) -> bool:
    if event_type == "push":
        return True
    if event_type == "pull_request":
        pr = payload.get("pull_request", {})
        return payload.get("action") == "closed" and bool(pr.get("merged") or pr.get("merged_at"))
    if event_type == "issues":
        return payload.get("action") == "closed"
    return False


def _extract_push_churn(payload: dict[str, Any]) -> int:
    total = 0
    for commit in payload.get("commits", []) or []:
        stats = commit.get("stats") or {}
        total += int(stats.get("additions") or commit.get("additions") or commit.get("added_lines") or 0)
        total += int(stats.get("deletions") or commit.get("deletions") or commit.get("deleted_lines") or 0)
    head_stats = payload.get("head_commit", {}).get("stats") or {}
    total += int(head_stats.get("additions") or 0)
    total += int(head_stats.get("deletions") or 0)
    return total


def _parse_http_headers(header_bytes: bytes) -> tuple[str, dict[str, str]]:
    header_text = header_bytes.decode("iso-8859-1")
    lines = header_text.split("\r\n")
    request_line = lines[0]
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    return request_line, headers


def _write_http_response(
    writer: asyncio.StreamWriter,
    status: int,
    payload: dict[str, Any],
) -> None:
    reason = {
        202: "Accepted",
        400: "Bad Request",
        404: "Not Found",
        405: "Method Not Allowed",
        500: "Internal Server Error",
    }.get(status, "OK")
    body = json.dumps(payload).encode("utf-8")
    writer.write(
        b"\r\n".join(
            [
                f"HTTP/1.1 {status} {reason}".encode("ascii"),
                b"Content-Type: application/json",
                f"Content-Length: {len(body)}".encode("ascii"),
                b"Connection: close",
                b"",
                body,
            ]
        )
    )


def _headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _configured_tokens(tokens: list[str] | None) -> list[str]:
    if tokens is not None:
        return tokens
    if not GITHUB_TOKEN:
        return []
    return [token.strip() for token in GITHUB_TOKEN.split(",") if token.strip()]


def _commit_timestamp(commit: dict[str, Any]) -> datetime | None:
    raw = commit.get("commit", {}).get("author", {}).get("date") or commit.get("commit", {}).get("committer", {}).get(
        "date"
    )
    return _parse_datetime(raw)


def _commit_login(commit: dict[str, Any]) -> str | None:
    author = commit.get("author") or {}
    if author.get("login"):
        return str(author["login"])
    committer = commit.get("committer") or {}
    if committer.get("login"):
        return str(committer["login"])
    return None


def _is_recent_merge(pr: dict[str, Any], since: datetime) -> bool:
    merged_at = _parse_datetime(pr.get("merged_at"))
    return merged_at is not None and merged_at >= since


def _merge_hours(pr: dict[str, Any]) -> float | None:
    created_at = _parse_datetime(pr.get("created_at"))
    merged_at = _parse_datetime(pr.get("merged_at"))
    if created_at is None or merged_at is None:
        return None
    return max((merged_at - created_at).total_seconds() / 3600.0, 0.0)


def _payload_repo(payload: dict[str, Any]) -> str:
    repo = payload.get("repository", {})
    return str(repo.get("full_name") or "unknown")


def _payload_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _github_ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _average(values: list[int] | list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
