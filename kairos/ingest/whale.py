"""Solana stablecoin whale flow tracker.

The WebSocket stream reacts to Token Program transfer logs, then fetches the
full parsed transaction over REST because Solana logs do not include mint,
owner, or token amount details.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from kairos.config import DB_PATH, SOLANA_RPC_URL
from kairos.db import create_schema, get_connection

logger = logging.getLogger(__name__)

SOLANA_RPC = SOLANA_RPC_URL
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
TRACKED_MINTS = {USDC_MINT, USDT_MINT}
MIN_WHALE_USD = 100_000.0
RECENT_FLOW_LIMIT = 2_000

EXCHANGE_WALLETS: dict[str, str] = {
    "38DtNzkSeg2kHzo2wM3Fw2iH4AtJ6r4K5K8a6hQYJQZu": "Binance",
    "2UE3k4SHxjyKCxx2JwA1q2BKj9HqYGTN2Niz7RFPkUCP": "Coinbase",
    "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM": "Kraken",
    "3tNQ6GtP1PQ5KZXQm3xGzHKQjQ2qKF8YKKQgVJj8aMkC": "OKX",
}

_RECENT_TRANSFERS: deque[dict[str, Any]] = deque(maxlen=RECENT_FLOW_LIMIT)
_RECENT_TRANSFER_KEYS: set[tuple[Any, ...]] = set()
_SEEN_SIGNATURES: set[str] = set()
_STREAM_TASK: asyncio.Task[None] | None = None

# Lazy async locks — created on first use to avoid "no event loop" at import time.
_TRANSFERS_LOCK: asyncio.Lock | None = None
_SIGNATURES_LOCK: asyncio.Lock | None = None


def _get_transfers_lock() -> asyncio.Lock:
    global _TRANSFERS_LOCK
    if _TRANSFERS_LOCK is None:
        _TRANSFERS_LOCK = asyncio.Lock()
    return _TRANSFERS_LOCK


def _get_signatures_lock() -> asyncio.Lock:
    global _SIGNATURES_LOCK
    if _SIGNATURES_LOCK is None:
        _SIGNATURES_LOCK = asyncio.Lock()
    return _SIGNATURES_LOCK


def register_exchange_wallet(address: str, exchange: str) -> None:
    """Register or update an exchange wallet mapping at runtime."""
    clean_address = address.strip()
    clean_exchange = exchange.strip()
    if not clean_address or not clean_exchange:
        raise ValueError("address and exchange are required")
    EXCHANGE_WALLETS[clean_address] = clean_exchange


def refresh_exchange_wallets_from_db(db_path: str | None = None) -> None:
    """Load exchange wallet labels from the `exchange_wallets` table."""
    conn = get_connection(db_path or DB_PATH)
    try:
        create_schema(conn)
        rows = conn.execute("SELECT address, exchange FROM exchange_wallets WHERE address IS NOT NULL").fetchall()
        for address, exchange in rows:
            register_exchange_wallet(str(address), str(exchange))
    finally:
        conn.close()


def _rpc_to_ws_url(rpc_url: str) -> str:
    if rpc_url.startswith("wss://") or rpc_url.startswith("ws://"):
        return rpc_url
    if rpc_url.startswith("https://"):
        return "wss://" + rpc_url[len("https://") :]
    if rpc_url.startswith("http://"):
        return "ws://" + rpc_url[len("http://") :]
    return rpc_url


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_from_unix(ts: int | float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return _utc_now()


def _account_pubkey(account_key: Any) -> str:
    if isinstance(account_key, dict):
        return str(account_key.get("pubkey", ""))
    return str(account_key)


def _account_keys(result: dict[str, Any]) -> list[str]:
    keys = result.get("transaction", {}).get("message", {}).get("accountKeys", [])
    return [_account_pubkey(key) for key in keys]


def _token_amount_to_units(token_amount: dict[str, Any] | None) -> tuple[int, int]:
    if not token_amount:
        return 0, 0
    decimals = int(token_amount.get("decimals") or 0)
    amount = token_amount.get("amount")
    if amount is not None:
        return int(amount), decimals
    ui_amount = token_amount.get("uiAmountString", token_amount.get("uiAmount", 0))
    return int(float(ui_amount or 0) * (10**decimals)), decimals


def _token_balance_map(result: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    account_keys = _account_keys(result)
    balances: dict[str, dict[str, Any]] = {}
    for balance in result.get("meta", {}).get(key, []) or []:
        idx = int(balance.get("accountIndex", -1))
        if idx < 0 or idx >= len(account_keys):
            continue
        amount, decimals = _token_amount_to_units(balance.get("uiTokenAmount"))
        account = account_keys[idx]
        balances[account] = {
            "mint": balance.get("mint"),
            "owner": balance.get("owner") or account,
            "amount": amount,
            "decimals": decimals,
        }
    return balances


def _token_meta(
    account: str | None,
    pre: dict[str, dict[str, Any]],
    post: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not account:
        return {}
    return post.get(account) or pre.get(account) or {}


def _iter_token_instructions(result: dict[str, Any]) -> list[dict[str, Any]]:
    message = result.get("transaction", {}).get("message", {})
    instructions = list(message.get("instructions", []) or [])
    for inner in result.get("meta", {}).get("innerInstructions", []) or []:
        instructions.extend(inner.get("instructions", []) or [])
    token_instructions = []
    for instruction in instructions:
        parsed = instruction.get("parsed") if isinstance(instruction, dict) else None
        if not isinstance(parsed, dict):
            continue
        program = str(instruction.get("program", ""))
        parsed_type = str(parsed.get("type", ""))
        if program.startswith("spl-token") and parsed_type.startswith("transfer"):
            token_instructions.append(instruction)
    return token_instructions


def _instruction_amount(info: dict[str, Any], decimals: int) -> float:
    token_amount = info.get("tokenAmount")
    if isinstance(token_amount, dict):
        amount, token_decimals = _token_amount_to_units(token_amount)
        decimals = token_decimals
    else:
        amount = int(info.get("amount") or 0)
    return amount / (10**decimals) if decimals >= 0 else 0.0


def _exchange_for_wallet(wallet: str | None, token_account: str | None = None) -> str | None:
    if wallet and wallet in EXCHANGE_WALLETS:
        return EXCHANGE_WALLETS[wallet]
    if token_account and token_account in EXCHANGE_WALLETS:
        return EXCHANGE_WALLETS[token_account]
    return None


def _direction(from_exchange: str | None, to_exchange: str | None) -> str:
    if to_exchange:
        return "inflow"
    if from_exchange:
        return "outflow"
    return "outflow"


def _parse_transaction_result(signature: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract large USDC/USDT transfers from a parsed Solana transaction."""
    pre = _token_balance_map(result, "preTokenBalances")
    post = _token_balance_map(result, "postTokenBalances")
    transfers: list[dict[str, Any]] = []

    for instruction in _iter_token_instructions(result):
        info = instruction.get("parsed", {}).get("info", {})
        source = info.get("source")
        destination = info.get("destination")
        source_meta = _token_meta(source, pre, post)
        dest_meta = _token_meta(destination, pre, post)
        mint = info.get("mint") or source_meta.get("mint") or dest_meta.get("mint")
        if mint not in TRACKED_MINTS:
            continue

        decimals = int(source_meta.get("decimals", dest_meta.get("decimals", 6)))
        usd_value = _instruction_amount(info, decimals)
        if usd_value < MIN_WHALE_USD:
            continue

        from_wallet = source_meta.get("owner") or source
        to_wallet = dest_meta.get("owner") or destination
        from_exchange = _exchange_for_wallet(from_wallet, source)
        to_exchange = _exchange_for_wallet(to_wallet, destination)

        transfers.append(
            {
                "signature": signature,
                "mint": mint,
                "from_wallet": from_wallet,
                "to_wallet": to_wallet,
                "from_exchange": from_exchange,
                "to_exchange": to_exchange,
                "usd_value": float(round(usd_value, 2)),
                "direction": _direction(from_exchange, to_exchange),
                "slot": result.get("slot"),
                "block_time": _iso_from_unix(result.get("blockTime")),
            }
        )

    return transfers


def _transfer_key(transfer: dict[str, Any]) -> tuple[Any, ...]:
    return (
        transfer.get("signature"),
        transfer.get("mint"),
        transfer.get("from_wallet"),
        transfer.get("to_wallet"),
        transfer.get("usd_value"),
    )


async def _remember_transfer(transfer: dict[str, Any]) -> None:
    async with _get_transfers_lock():
        key = _transfer_key(transfer)
        if key in _RECENT_TRANSFER_KEYS:
            return
        item = dict(transfer)
        item.setdefault("detected_at", _utc_now().isoformat())
        _RECENT_TRANSFERS.append(item)
        _RECENT_TRANSFER_KEYS.add(key)
        if len(_RECENT_TRANSFER_KEYS) > RECENT_FLOW_LIMIT * 2:
            _RECENT_TRANSFER_KEYS.clear()
            _RECENT_TRANSFER_KEYS.update(_transfer_key(t) for t in _RECENT_TRANSFERS)


def _is_exchange_related(transfer: dict[str, Any]) -> bool:
    return bool(transfer.get("from_exchange") or transfer.get("to_exchange"))


_persist_local = threading.local()


def _get_persist_conn(db_path: str | None = None):
    if not hasattr(_persist_local, "conn") or _persist_local.conn is None:
        _persist_local.conn = get_connection(db_path or DB_PATH)
        create_schema(_persist_local.conn)
    return _persist_local.conn


def _persist_transfer(transfer: dict[str, Any], db_path: str | None = None) -> None:
    if not _is_exchange_related(transfer):
        return

    try:
        conn = _get_persist_conn(db_path)
        if conn.execute(
            "SELECT 1 FROM whale_transfers WHERE signature = ?",
            [transfer.get("signature")],
        ).fetchone():
            return
        conn.execute(
            """
            INSERT INTO whale_transfers(
                signature, mint, from_wallet, to_wallet, usd_value,
                direction, slot, block_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                transfer.get("signature"),
                transfer.get("mint"),
                transfer.get("from_wallet"),
                transfer.get("to_wallet"),
                float(transfer.get("usd_value", 0.0)),
                transfer.get("direction"),
                transfer.get("slot"),
                transfer.get("block_time"),
            ],
        )
    except Exception as exc:
        logger.warning("Failed to persist whale transfer: %s", exc)
        _persist_local.conn = None  # reset so next call gets a fresh connection


def _net_exchange_flow(transfers: list[dict[str, Any]]) -> float:
    total = 0.0
    for transfer in transfers:
        if not _is_exchange_related(transfer):
            continue
        usd_value = float(transfer.get("usd_value", 0.0))
        total += usd_value if transfer.get("direction") == "inflow" else -usd_value
    return round(total, 2)


def get_whale_metrics(
    db_path: str | None = None,
    now: datetime | None = None,
) -> dict[str, float | int]:
    """Return exchange-relative whale flow metrics from persisted transfers."""
    conn = get_connection(db_path or DB_PATH)
    try:
        create_schema(conn)
        current = now or _utc_now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        cutoff_1h = current - timedelta(hours=1)
        cutoff_24h = current - timedelta(hours=24)

        def net_since(cutoff: datetime) -> float:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(
                    CASE
                        WHEN direction = 'inflow' THEN usd_value
                        WHEN direction = 'outflow' THEN -usd_value
                        ELSE 0
                    END
                ), 0)
                FROM whale_transfers
                WHERE detected_at >= ?
                """,
                [cutoff],
            ).fetchone()
            return round(float(row[0] or 0.0), 2)

        row = conn.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE detected_at >= ? AND usd_value >= ?
                ) AS whale_count_1h,
                COALESCE(AVG(usd_value) FILTER (
                    WHERE detected_at >= ? AND usd_value >= ?
                ), 0) AS avg_whale_size_24h,
                COALESCE(MAX(usd_value) FILTER (
                    WHERE detected_at >= ? AND usd_value >= ?
                ), 0) AS largest_flow_24h
            FROM whale_transfers
            """,
            [
                cutoff_1h,
                MIN_WHALE_USD,
                cutoff_24h,
                MIN_WHALE_USD,
                cutoff_24h,
                MIN_WHALE_USD,
            ],
        ).fetchone()

        return {
            "net_exchange_flow_1h": net_since(cutoff_1h),
            "net_exchange_flow_24h": net_since(cutoff_24h),
            "whale_count_1h": int(row[0] or 0),
            "avg_whale_size_24h": round(float(row[1] or 0.0), 2),
            "largest_flow_24h": round(float(row[2] or 0.0), 2),
        }
    finally:
        conn.close()


async def get_recent_flows(minutes: int = 5) -> list[dict[str, Any]]:
    """Return in-memory transfers detected within the last `minutes` minutes."""
    cutoff = _utc_now() - timedelta(minutes=minutes)
    async with _get_transfers_lock():
        return [dict(transfer) for transfer in _RECENT_TRANSFERS if _parse_dt(transfer.get("detected_at")) >= cutoff]


async def _rpc_post(
    client: httpx.AsyncClient,
    rpc_url: str,
    method: str,
    params: list[Any],
    request_id: int = 1,
) -> Any:
    resp = await client.post(
        rpc_url,
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    return data.get("result")


async def _get_signatures(
    client: httpx.AsyncClient,
    mint: str,
    limit: int = 50,
    rpc_url: str = SOLANA_RPC_URL,
) -> list[str]:
    result = await _rpc_post(
        client,
        rpc_url,
        "getSignaturesForAddress",
        [mint, {"limit": limit}],
    )
    return [row["signature"] for row in result or [] if row.get("signature")]


async def _get_transaction(
    client: httpx.AsyncClient,
    signature: str,
    rpc_url: str = SOLANA_RPC_URL,
) -> dict[str, Any] | None:
    return await _rpc_post(
        client,
        rpc_url,
        "getTransaction",
        [
            signature,
            {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
        ],
    )


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def _fetch_transaction_batch(
    client: httpx.AsyncClient,
    signatures: list[str],
    rpc_url: str,
    db_path: str | None,
) -> list[dict[str, Any]]:
    transfers: list[dict[str, Any]] = []
    for batch in _chunks(signatures, 10):
        tasks = [_get_transaction(client, sig, rpc_url) for sig in batch]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=10.0,
            )
        except TimeoutError:
            logger.warning("Timed out fetching Solana transaction batch")
            continue

        for signature, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.debug("Failed to fetch Solana transaction %s: %s", signature, result)
                continue
            if not result:
                continue
            parsed = _parse_transaction_result(signature, result)
            for transfer in parsed:
                _remember_transfer(transfer)
                _persist_transfer(transfer, db_path)
            transfers.extend(parsed)
    return transfers


def _default_whale_flows() -> dict[str, Any]:
    return {
        "available": False,
        "transfers_count": 0,
        "net_flow_usd": 0.0,
        "largest_flow_usd": 0.0,
        "transfers": [],
        "net_exchange_flow_1h": 0.0,
        "net_exchange_flow_24h": 0.0,
        "whale_count_1h": 0,
        "avg_whale_size_24h": 0.0,
        "largest_flow_24h": 0.0,
        "fetch_ts": _utc_now().isoformat(),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not np_isfinite(result):
        return default
    return result


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _sanitize_whale_flows_result(result: dict[str, Any]) -> dict[str, Any]:
    clean = _default_whale_flows()
    clean["available"] = bool(result.get("available", True))
    clean["transfers_count"] = max(int(_safe_float(result.get("transfers_count"))), 0)
    clean["net_flow_usd"] = _safe_float(result.get("net_flow_usd"))
    clean["largest_flow_usd"] = max(_safe_float(result.get("largest_flow_usd")), 0.0)
    transfers = result.get("transfers")
    clean["transfers"] = transfers[:20] if isinstance(transfers, list) else []
    clean["net_exchange_flow_1h"] = _safe_float(result.get("net_exchange_flow_1h"))
    clean["net_exchange_flow_24h"] = _safe_float(result.get("net_exchange_flow_24h"))
    clean["whale_count_1h"] = max(int(_safe_float(result.get("whale_count_1h"))), 0)
    clean["avg_whale_size_24h"] = max(_safe_float(result.get("avg_whale_size_24h")), 0.0)
    clean["largest_flow_24h"] = max(_safe_float(result.get("largest_flow_24h")), 0.0)
    clean["fetch_ts"] = str(result.get("fetch_ts") or clean["fetch_ts"])
    return clean


async def fetch_whale_flows(limit: int = 30) -> dict[str, Any]:
    """Fetch recent large stablecoin transfers through the REST fallback path."""
    refresh_exchange_wallets_from_db()
    async with httpx.AsyncClient(timeout=10.0) as client:
        signature_lists = await asyncio.gather(
            *[_get_signatures(client, mint, limit=limit, rpc_url=SOLANA_RPC_URL) for mint in TRACKED_MINTS],
            return_exceptions=True,
        )
        signatures: list[str] = []
        seen: set[str] = set()
        for result in signature_lists:
            if isinstance(result, Exception):
                logger.debug("Failed to fetch Solana signatures: %s", result)
                continue
            for signature in result:
                if signature not in seen:
                    seen.add(signature)
                    signatures.append(signature)

        transfers = await _fetch_transaction_batch(
            client,
            signatures,
            SOLANA_RPC_URL,
            DB_PATH,
        )

    transfers.sort(key=lambda t: t.get("usd_value", 0.0), reverse=True)
    metrics = get_whale_metrics()
    return _sanitize_whale_flows_result(
        {
            "transfers_count": len(transfers),
            "net_flow_usd": _net_exchange_flow(transfers),
            "largest_flow_usd": round(
                max((float(t["usd_value"]) for t in transfers), default=0.0),
                2,
            ),
            "transfers": transfers[:20],
            "fetch_ts": _utc_now().isoformat(),
            **metrics,
        }
    )


def _logs_subscribe_request(request_id: int) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "logsSubscribe",
        "params": [
            {"mentions": [TOKEN_PROGRAM_ID]},
            {"commitment": "confirmed"},
        ],
    }


def _program_subscribe_request(request_id: int, mint: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "programSubscribe",
        "params": [
            TOKEN_PROGRAM_ID,
            {
                "encoding": "base64",
                "commitment": "confirmed",
                "filters": [
                    {"dataSize": 165},
                    {"memcmp": {"offset": 0, "bytes": mint}},
                ],
            },
        ],
    }


async def _send_subscriptions(ws: Any) -> None:
    await ws.send(json.dumps(_logs_subscribe_request(1)))
    for offset, mint in enumerate(sorted(TRACKED_MINTS), start=2):
        await ws.send(json.dumps(_program_subscribe_request(offset, mint)))


def _has_transfer_log(logs: list[str]) -> bool:
    return any("Instruction: Transfer" in log for log in logs)


async def _handle_logs_notification(
    client: httpx.AsyncClient,
    value: dict[str, Any],
    rpc_url: str,
    db_path: str | None,
) -> None:
    if value.get("err"):
        return
    logs = value.get("logs", []) or []
    if not _has_transfer_log(logs):
        return
    signature = value.get("signature")
    if not signature:
        return
    async with _get_signatures_lock():
        if signature in _SEEN_SIGNATURES:
            return
        _SEEN_SIGNATURES.add(signature)
        if len(_SEEN_SIGNATURES) > RECENT_FLOW_LIMIT * 4:
            _SEEN_SIGNATURES.clear()

    result = await _get_transaction(client, signature, rpc_url)
    if not result:
        return
    for transfer in _parse_transaction_result(signature, result):
        await _remember_transfer(transfer)
        _persist_transfer(transfer, db_path)


async def stream_whale_flows(
    rpc_url: str | None = None,
    db_path: str | None = None,
) -> None:
    """Run the Solana WebSocket stream forever with exponential reconnect."""
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("websockets is required for Solana streaming") from exc

    rest_url = rpc_url or SOLANA_RPC_URL
    ws_url = _rpc_to_ws_url(rest_url)
    backoff = 1

    while True:
        try:
            refresh_exchange_wallets_from_db(db_path)
            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                await _send_subscriptions(ws)
                backoff = 1
                async with httpx.AsyncClient(timeout=10.0) as client:
                    async for raw in ws:
                        message = json.loads(raw)
                        if message.get("method") != "logsNotification":
                            continue
                        value = message.get("params", {}).get("result", {}).get("value", {})
                        await _handle_logs_notification(
                            client,
                            value,
                            rest_url,
                            db_path or DB_PATH,
                        )
                raise ConnectionError("Solana WebSocket closed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Solana whale stream disconnected; reconnecting in %ss: %s",
                backoff,
                exc,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def start_whale_stream(
    rpc_url: str | None = None,
    db_path: str | None = None,
) -> asyncio.Task[None]:
    """Start the whale stream as a background task in the current event loop."""
    global _STREAM_TASK
    if _STREAM_TASK and not _STREAM_TASK.done():
        return _STREAM_TASK
    _STREAM_TASK = asyncio.create_task(stream_whale_flows(rpc_url, db_path))
    return _STREAM_TASK
