import importlib
import ssl
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from kairos.models.signal_event import SignalEvent


def _event(direction: str = "bullish", confidence: float = 0.72, asset: str = "BTC") -> SignalEvent:
    return SignalEvent(
        asset=asset,
        direction=direction,
        confidence=confidence,
        regime="lv_up",
        narrative_velocity=0.03,
        narrative_tipping_point=True,
        mechanism="regime_routed(lv_up) -> narrative(0.03) -> price",
        estimated_hours=24.0,
        citations=["test citation"],
        triggered_at=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
    )


def test_telegram_format():
    from kairos.notifier import _format_direction_change

    message = _format_direction_change(_event(direction="bearish"), "bullish", "bearish")

    assert "BTC" in message
    assert "bullish -> *BEARISH*" in message
    assert "72%" in message
    assert "```" in message


def test_discord_embed():
    from kairos.notifier import _build_discord_embed

    embed = _build_discord_embed(_event(), {"current_price": 67500.25})
    field_names = {field["name"] for field in embed["fields"]}

    assert embed["title"] == "Kairos BTC alert"
    assert embed["color"] == 0x2ECC71
    assert {
        "Asset",
        "Direction",
        "Confidence",
        "Regime",
        "Price",
        "Mechanism",
    } <= field_names
    assert embed["thumbnail"]["url"].startswith("https://")


@pytest.mark.asyncio
async def test_notifier_suppress():
    from kairos.notifier import Notifier

    notifier = Notifier()
    sent: list[str] = []

    async def record_send(message: str, event: SignalEvent, ctx: dict) -> dict[str, bool]:
        sent.append(message)
        return {"telegram": True}

    notifier._dispatch_signal_alert = record_send
    notifier._last_state["BTC"] = "bearish"
    notifier.suppress_for("BTC", minutes=30.0)

    await notifier.check_and_notify(_event(direction="bullish"), {})

    assert sent == []
    assert notifier._last_state["BTC"] == "bullish"


@pytest.mark.asyncio
async def test_notifier_state_change():
    from kairos.notifier import Notifier

    notifier = Notifier()
    sent: list[str] = []

    async def record_send(message: str, event: SignalEvent, ctx: dict) -> dict[str, bool]:
        sent.append(message)
        return {"telegram": True}

    notifier._dispatch_signal_alert = record_send
    notifier._last_state["BTC"] = "bearish"

    await notifier.check_and_notify(_event(direction="bullish"), {})

    assert len(sent) == 1
    assert "bearish -> *BULLISH*" in sent[0]


@pytest.mark.asyncio
async def test_notifier_no_change():
    from kairos.notifier import Notifier

    notifier = Notifier()
    sent: list[str] = []

    async def record_send(message: str, event: SignalEvent, ctx: dict) -> dict[str, bool]:
        sent.append(message)
        return {"telegram": True}

    notifier._dispatch_signal_alert = record_send
    notifier._last_state["BTC"] = "bullish"

    await notifier.check_and_notify(_event(direction="bullish"), {})

    assert sent == []


@pytest.mark.asyncio
async def test_notifier_anomaly_alert():
    from kairos.notifier import Notifier

    notifier = Notifier()
    sent: list[str] = []

    async def record_send(message: str, event: SignalEvent, ctx: dict) -> dict[str, bool]:
        sent.append(message)
        return {"telegram": True}

    notifier._dispatch_signal_alert = record_send
    notifier._last_state["BTC"] = "bullish"

    await notifier.check_and_notify(_event(direction="bullish"), {"anomaly_score": 1})

    assert len(sent) == 1
    assert "Anomaly" in sent[0]


@pytest.mark.asyncio
async def test_notifier_confidence_drop():
    from kairos.notifier import Notifier

    notifier = Notifier()
    sent: list[str] = []

    async def record_send(message: str, event: SignalEvent, ctx: dict) -> dict[str, bool]:
        sent.append(message)
        return {"telegram": True}

    notifier._dispatch_signal_alert = record_send
    notifier._last_state["BTC"] = "bullish"
    notifier._last_confidence["BTC"] = 0.90

    await notifier.check_and_notify(_event(direction="bullish", confidence=0.62), {})

    assert len(sent) == 1
    assert "Confidence drop" in sent[0]
    assert "90%" in sent[0]
    assert "62%" in sent[0]


def test_config_defaults(monkeypatch):
    for name in (
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "DISCORD_WEBHOOK_URL",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USERNAME",
        "SMTP_PASSWORD",
        "EMAIL_FROM",
        "EMAIL_TO",
    ):
        monkeypatch.delenv(name, raising=False)

    import kairos.config as config

    config = importlib.reload(config)

    assert config.TELEGRAM_BOT_TOKEN == ""
    assert config.TELEGRAM_CHAT_ID == ""
    assert config.DISCORD_WEBHOOK_URL == ""
    assert config.SMTP_HOST == ""
    assert config.SMTP_PORT == 587
    assert config.SMTP_USERNAME == ""
    assert config.SMTP_PASSWORD == ""
    assert config.EMAIL_FROM == ""
    assert config.EMAIL_TO == ""


def test_smtp_starttls_passes_ssl_context():
    from kairos.notifier import Notifier, NotifierConfig

    mock_smtp = MagicMock()
    mock_smtp.starttls = MagicMock()
    mock_smtp.login = MagicMock()
    mock_smtp.send_message = MagicMock()

    with patch("smtplib.SMTP") as smtp_cls:
        smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        config = NotifierConfig(
            email_smtp_host="smtp.example.com",
            email_smtp_port=587,
            email_username="user@example.com",
            email_password="secret",
            email_from="from@example.com",
            email_to="to@example.com",
        )
        notifier = Notifier(config)
        notifier._send_email_sync("Test Subject", "Test body")

    mock_smtp.starttls.assert_called_once()
    call_kwargs = mock_smtp.starttls.call_args.kwargs
    assert "context" in call_kwargs
    assert isinstance(call_kwargs["context"], ssl.SSLContext)


@pytest.mark.asyncio
async def test_transport_failures_log_exception_type_without_traceback(monkeypatch):
    from kairos.notifier import Notifier, NotifierConfig

    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            raise RuntimeError("transport includes secret-bearing URL")

    warnings = []

    def record_warning(message, *args, **kwargs):
        warnings.append((message, args, kwargs))

    monkeypatch.setattr("kairos.notifier.httpx.AsyncClient", lambda *args, **kwargs: FailingClient())
    monkeypatch.setattr("kairos.notifier.logger.warning", record_warning)

    notifier = Notifier(
        NotifierConfig(
            telegram_token="secret-token",
            telegram_chat_id="chat-id",
            discord_webhook_url="https://discord.example.test/webhook/secret",
            email_smtp_host="smtp.example.test",
            email_from="from@example.test",
            email_to="to@example.test",
        )
    )
    monkeypatch.setattr(
        notifier,
        "_send_email_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("smtp secret")),
    )

    assert await notifier._send_telegram("hello") is False
    assert await notifier._send_discord("hello") is False
    assert await notifier._send_email("subject", "body") is False

    assert len(warnings) == 3
    assert all(call[2].get("exc_info") is not True for call in warnings)
    assert all("RuntimeError" in call[1] for call in warnings)
