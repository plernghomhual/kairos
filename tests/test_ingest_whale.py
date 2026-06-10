import json
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from kairos.db import create_schema


def test_schema_creates_exchange_and_whale_tables(tmp_path):
    db_path = str(tmp_path / "whales.db")
    conn = duckdb.connect(db_path)

    create_schema(conn)

    table_names = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert "exchange_wallets" in table_names
    assert "whale_transfers" in table_names
    conn.close()


def test_exchange_wallets_register_and_refresh_from_db(tmp_path):
    from kairos.ingest.whale import (
        EXCHANGE_WALLETS,
        refresh_exchange_wallets_from_db,
        register_exchange_wallet,
    )

    original = dict(EXCHANGE_WALLETS)
    try:
        EXCHANGE_WALLETS.clear()
        register_exchange_wallet("RuntimeWallet", "Runtime Exchange")
        assert EXCHANGE_WALLETS["RuntimeWallet"] == "Runtime Exchange"

        db_path = str(tmp_path / "whales.db")
        conn = duckdb.connect(db_path)
        create_schema(conn)
        conn.execute(
            "INSERT INTO exchange_wallets(address, exchange, label) VALUES (?, ?, ?)",
            ["DbWallet", "DB Exchange", "hot wallet"],
        )
        conn.close()

        refresh_exchange_wallets_from_db(db_path=db_path)

        assert EXCHANGE_WALLETS["DbWallet"] == "DB Exchange"
        assert EXCHANGE_WALLETS["RuntimeWallet"] == "Runtime Exchange"
    finally:
        EXCHANGE_WALLETS.clear()
        EXCHANGE_WALLETS.update(original)


def test_parse_transaction_result_extracts_usdc_exchange_inflow():
    from kairos.ingest.whale import (
        USDC_MINT,
        _parse_transaction_result,
        register_exchange_wallet,
    )

    register_exchange_wallet("CoinbaseOwner", "Coinbase")
    tx = {
        "slot": 123,
        "blockTime": 1_700_000_000,
        "transaction": {
            "message": {
                "accountKeys": [
                    {"pubkey": "SourceToken"},
                    {"pubkey": "DestToken"},
                ],
                "instructions": [
                    {
                        "program": "spl-token",
                        "parsed": {
                            "type": "transferChecked",
                            "info": {
                                "source": "SourceToken",
                                "destination": "DestToken",
                                "mint": USDC_MINT,
                                "tokenAmount": {
                                    "amount": "250000000000",
                                    "decimals": 6,
                                },
                            },
                        },
                    }
                ],
            }
        },
        "meta": {
            "preTokenBalances": [
                {
                    "accountIndex": 0,
                    "mint": USDC_MINT,
                    "owner": "TraderOwner",
                    "uiTokenAmount": {"amount": "250000000000", "decimals": 6},
                },
                {
                    "accountIndex": 1,
                    "mint": USDC_MINT,
                    "owner": "CoinbaseOwner",
                    "uiTokenAmount": {"amount": "0", "decimals": 6},
                },
            ],
            "postTokenBalances": [
                {
                    "accountIndex": 0,
                    "mint": USDC_MINT,
                    "owner": "TraderOwner",
                    "uiTokenAmount": {"amount": "0", "decimals": 6},
                },
                {
                    "accountIndex": 1,
                    "mint": USDC_MINT,
                    "owner": "CoinbaseOwner",
                    "uiTokenAmount": {"amount": "250000000000", "decimals": 6},
                },
            ],
        },
    }

    transfers = _parse_transaction_result("sig123", tx)

    assert transfers == [
        {
            "signature": "sig123",
            "mint": USDC_MINT,
            "from_wallet": "TraderOwner",
            "to_wallet": "CoinbaseOwner",
            "from_exchange": None,
            "to_exchange": "Coinbase",
            "usd_value": 250000.0,
            "direction": "inflow",
            "slot": 123,
            "block_time": "2023-11-14T22:13:20+00:00",
        }
    ]


def test_get_whale_metrics_reads_persisted_transfers(tmp_path):
    from kairos.ingest.whale import get_whale_metrics

    db_path = str(tmp_path / "whales.db")
    conn = duckdb.connect(db_path)
    create_schema(conn)
    now = datetime.now(timezone.utc)
    rows = [
        ("sig-in", "USDC", "a", "b", 150000.0, "inflow", now),
        ("sig-out", "USDC", "b", "a", 120000.0, "outflow", now - timedelta(minutes=20)),
        ("sig-old", "USDT", "a", "b", 300000.0, "inflow", now - timedelta(hours=2)),
        (
            "sig-expired",
            "USDT",
            "a",
            "b",
            900000.0,
            "inflow",
            now - timedelta(hours=25),
        ),
    ]
    for row in rows:
        conn.execute(
            """
            INSERT INTO whale_transfers(
                signature, mint, from_wallet, to_wallet, usd_value,
                direction, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
    conn.close()

    metrics = get_whale_metrics(db_path=db_path, now=now)

    assert metrics["net_exchange_flow_1h"] == 30000.0
    assert metrics["net_exchange_flow_24h"] == 330000.0
    assert metrics["whale_count_1h"] == 2
    assert metrics["avg_whale_size_24h"] == pytest.approx(190000.0, abs=0.01)
    assert metrics["largest_flow_24h"] == 300000.0


def test_import_acceptance_contract():
    from kairos.config import SOLANA_RPC_URL
    from kairos.ingest import get_recent_flows, start_whale_stream
    from kairos.ingest.whale import fetch_whale_flows, register_exchange_wallet

    assert SOLANA_RPC_URL == "https://api.mainnet-beta.solana.com"
    assert callable(fetch_whale_flows)
    assert callable(register_exchange_wallet)
    assert callable(get_recent_flows)
    assert callable(start_whale_stream)


@pytest.mark.asyncio
async def test_send_subscriptions_requests_logs_and_program_streams():
    from kairos.ingest.whale import (
        TOKEN_PROGRAM_ID,
        TRACKED_MINTS,
        _rpc_to_ws_url,
        _send_subscriptions,
    )

    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(json.loads(payload))

    ws = FakeWebSocket()

    await _send_subscriptions(ws)

    assert _rpc_to_ws_url("https://api.mainnet-beta.solana.com") == ("wss://api.mainnet-beta.solana.com")
    assert ws.sent[0]["method"] == "logsSubscribe"
    assert ws.sent[0]["params"][0] == {"mentions": [TOKEN_PROGRAM_ID]}
    program_subs = [msg for msg in ws.sent if msg["method"] == "programSubscribe"]
    assert len(program_subs) == len(TRACKED_MINTS)
    assert {msg["params"][0] for msg in program_subs} == {TOKEN_PROGRAM_ID}
    assert {msg["params"][1]["filters"][1]["memcmp"]["bytes"] for msg in program_subs} == TRACKED_MINTS


@pytest.mark.asyncio
async def test_remember_transfer_concurrent_no_duplicate():
    """Concurrent _remember_transfer calls with the same signature must not insert duplicates."""
    import asyncio

    from kairos.ingest.whale import (
        _RECENT_TRANSFER_KEYS,
        _RECENT_TRANSFERS,
        _remember_transfer,
    )

    _RECENT_TRANSFERS.clear()
    _RECENT_TRANSFER_KEYS.clear()

    transfer = {
        "signature": "sig-race-concurrent-001",
        "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "from_wallet": "wallet_a",
        "to_wallet": "wallet_b",
        "usd_value": 5_000_000.0,
        "direction": "inflow",
        "slot": 123,
        "block_time": 1700000000,
    }

    await asyncio.gather(*[_remember_transfer(dict(transfer)) for _ in range(10)])

    matching = [t for t in _RECENT_TRANSFERS if t.get("signature") == "sig-race-concurrent-001"]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_fetch_transaction_batch_remembers_rest_transfers(monkeypatch):
    from kairos.ingest.whale import (
        _RECENT_TRANSFER_KEYS,
        _RECENT_TRANSFERS,
        _fetch_transaction_batch,
    )

    _RECENT_TRANSFERS.clear()
    _RECENT_TRANSFER_KEYS.clear()
    transfer = {
        "signature": "sig-rest-remembered",
        "mint": "USDC",
        "from_wallet": "wallet_a",
        "to_wallet": "wallet_b",
        "usd_value": 250_000.0,
        "direction": "inflow",
        "slot": 456,
        "block_time": "2026-06-09T00:00:00+00:00",
    }

    async def fake_get_transaction(_client, signature, _rpc_url):
        return {"signature": signature}

    monkeypatch.setattr("kairos.ingest.whale._get_transaction", fake_get_transaction)
    monkeypatch.setattr("kairos.ingest.whale._parse_transaction_result", lambda signature, result: [dict(transfer)])
    monkeypatch.setattr("kairos.ingest.whale._persist_transfer", lambda transfer, db_path: None)

    transfers = await _fetch_transaction_batch(object(), ["sig-rest-remembered"], "https://rpc.example.test", None)

    assert transfers == [transfer]
    assert [item["signature"] for item in _RECENT_TRANSFERS] == ["sig-rest-remembered"]


@pytest.mark.asyncio
async def test_recent_transfer_keys_track_evicted_buffer_entries():
    from kairos.ingest.whale import (
        _RECENT_TRANSFER_KEYS,
        _RECENT_TRANSFERS,
        RECENT_FLOW_LIMIT,
        _remember_transfer,
        _transfer_key,
    )

    _RECENT_TRANSFERS.clear()
    _RECENT_TRANSFER_KEYS.clear()

    for idx in range(RECENT_FLOW_LIMIT + 1):
        await _remember_transfer(
            {
                "signature": f"sig-{idx}",
                "mint": "USDC",
                "from_wallet": "wallet_a",
                "to_wallet": "wallet_b",
                "usd_value": 250_000.0,
                "direction": "inflow",
                "slot": idx,
                "block_time": "2026-06-09T00:00:00+00:00",
            }
        )

    assert len(_RECENT_TRANSFERS) == RECENT_FLOW_LIMIT
    assert _RECENT_TRANSFER_KEYS == {_transfer_key(transfer) for transfer in _RECENT_TRANSFERS}


def test_non_exchange_transfer_direction_is_external():
    from kairos.ingest.whale import _direction

    assert _direction(None, None) == "external"
