"""Async notification dispatch for Kairos signal alerts."""

from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
import time
from collections import deque
from dataclasses import dataclass
from email.message import EmailMessage
from html import escape
from typing import Any

import httpx

from kairos.models.signal_event import SignalEvent

logger = logging.getLogger(__name__)

_LEVELS = {"info": 10, "warning": 20, "critical": 30}
_CHANNEL_MIN_LEVEL = {"telegram": "info", "discord": "info", "email": "warning"}
_DISCORD_COLORS = {
    "bullish": 0x2ECC71,
    "bearish": 0xE74C3C,
    "neutral": 0xF1C40F,
}
_COINGECKO_THUMBNAILS = {
    "BTC": "https://assets.coingecko.com/coins/images/1/large/bitcoin.png",
    "ETH": "https://assets.coingecko.com/coins/images/279/large/ethereum.png",
    "SOL": "https://assets.coingecko.com/coins/images/4128/large/solana.png",
}


@dataclass
class NotifierConfig:
    telegram_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: str = ""


class Notifier:
    def __init__(self, config: NotifierConfig | None = None):
        self._config = config or NotifierConfig()
        self._last_state: dict[str, str] = {}
        self._last_confidence: dict[str, float] = {}
        self._silent_until: dict[str, float] = {}
        self._telegram_sent_at: deque[float] = deque()
        self._discord_sent_at: deque[float] = deque()
        self._email_alerts: deque[tuple[float, str]] = deque()

    async def send(self, message: str, channels: list[str] | None = None) -> dict[str, bool]:
        """Send a message to specified channels.

        Supported channels are telegram, discord, and email. Failures are logged
        per channel and do not stop other channels.
        """
        targets = channels or self._enabled_channels()
        results: dict[str, bool] = {}
        for channel in targets:
            if channel == "telegram":
                results[channel] = await self._send_telegram(message)
            elif channel == "discord":
                results[channel] = await self._send_discord(message)
            elif channel == "email":
                results[channel] = await self._send_email("[Kairos] Alert", message)
            else:
                logger.warning("Unknown notification channel: %s", channel)
                results[channel] = False
        return results

    async def check_and_notify(self, event: SignalEvent, ctx: dict) -> None:
        """Notify when signal state changes, anomalies fire, or confidence drops."""
        asset = event.asset
        old_direction = self._last_state.get(asset)
        old_confidence = self._last_confidence.get(asset)

        if self._is_suppressed(asset):
            self._remember(event)
            return

        messages: list[str] = []
        if old_direction is not None and old_direction != event.direction:
            messages.append(_format_direction_change(event, old_direction, event.direction))

        if float(ctx.get("anomaly_score", 0.0) or 0.0) > 0.1:
            messages.append(_format_anomaly_alert(event, ctx))

        if old_confidence is not None and old_confidence - event.confidence > 0.20:
            messages.append(_format_confidence_drop(event, old_confidence))

        for message in messages:
            await self._dispatch_signal_alert(message, event, ctx)

        self._remember(event)

    def suppress_for(self, asset: str, minutes: float = 30.0) -> None:
        """Suppress notifications for an asset for N minutes."""
        self._silent_until[asset] = time.time() + minutes * 60.0

    async def send_alert(self, level: str, title: str, body: str) -> dict[str, bool]:
        """Send a formatted alert, respecting per-channel minimum levels."""
        level_key = level.lower()
        if level_key not in _LEVELS:
            raise ValueError("level must be one of: info, warning, critical")

        message = f"*{title}*\n\n{body}"
        channels = [
            channel
            for channel in self._enabled_channels()
            if _LEVELS[level_key] >= _LEVELS[_CHANNEL_MIN_LEVEL[channel]]
        ]
        return await self.send(message, channels=channels)

    def _enabled_channels(self) -> list[str]:
        channels: list[str] = []
        if self._config.telegram_token and self._config.telegram_chat_id:
            channels.append("telegram")
        if self._config.discord_webhook_url:
            channels.append("discord")
        if self._email_configured():
            channels.append("email")
        return channels

    def _email_configured(self) -> bool:
        return all(
            (
                self._config.email_smtp_host,
                self._config.email_from,
                self._config.email_to,
            )
        )

    def _is_suppressed(self, asset: str) -> bool:
        return time.time() < self._silent_until.get(asset, 0.0)

    def _remember(self, event: SignalEvent) -> None:
        self._last_state[event.asset] = event.direction
        self._last_confidence[event.asset] = event.confidence

    async def _dispatch_signal_alert(self, message: str, event: SignalEvent, ctx: dict) -> dict[str, bool]:
        targets = self._enabled_channels()
        results: dict[str, bool] = {}
        for channel in targets:
            if channel == "telegram":
                results[channel] = await self._send_telegram(message)
            elif channel == "discord":
                results[channel] = await self._send_discord(message, event=event, ctx=ctx)
            elif channel == "email":
                subject = _email_subject(event)
                results[channel] = await self._send_email(subject, message)
        return results

    async def _send_telegram(self, message: str) -> bool:
        if not self._config.telegram_token or not self._config.telegram_chat_id:
            return False
        if not _allow_rate(self._telegram_sent_at, limit=20):
            logger.warning("Telegram notification rate limit reached")
            return False

        url = "https://api.telegram.org/bot{}/sendMessage".format(self._config.telegram_token)
        payload = {
            "chat_id": self._config.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "reply_markup": {"inline_keyboard": [[{"text": "Acknowledge", "callback_data": "kairos_ack"}]]},
        }
        try:
            _httpx_log = logging.getLogger("httpx")
            _httpcore_log = logging.getLogger("httpcore")
            _prev = (_httpx_log.level, _httpcore_log.level)
            _httpx_log.setLevel(logging.ERROR)
            _httpcore_log.setLevel(logging.ERROR)
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
            finally:
                _httpx_log.setLevel(_prev[0])
                _httpcore_log.setLevel(_prev[1])
            return True
        except Exception:
            logger.warning("Telegram notification failed", exc_info=True)
            return False

    async def _send_discord(self, message: str, event: SignalEvent | None = None, ctx: dict | None = None) -> bool:
        if not self._config.discord_webhook_url:
            return False
        if not _allow_rate(self._discord_sent_at, limit=30):
            logger.warning("Discord notification rate limit reached")
            return False

        payload: dict[str, Any]
        if event is not None:
            payload = {
                "content": message,
                "embeds": [_build_discord_embed(event, ctx or {})],
            }
        else:
            payload = {"content": message}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{self._config.discord_webhook_url}?wait=true", json=payload)
                resp.raise_for_status()
            return True
        except Exception:
            logger.warning("Discord notification failed", exc_info=True)
            return False

    async def _send_email(self, subject: str, body: str) -> bool:
        if not self._email_configured():
            return False

        now = time.time()
        self._email_alerts.append((now, body))
        _prune_window(self._email_alerts, seconds=86400.0)
        if len(self._email_alerts) > 5:
            subject = "[Kairos] Daily Digest"
            body = _format_daily_digest_messages([msg for _, msg in self._email_alerts])
            self._email_alerts.clear()

        try:
            await asyncio.to_thread(self._send_email_sync, subject, body)
            return True
        except Exception:
            logger.warning("Email notification failed", exc_info=True)
            return False

    def _send_email_sync(self, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self._config.email_from
        msg["To"] = self._config.email_to
        msg.set_content(body)
        html = "<html><body><pre>" + escape(body) + "</pre></body></html>"
        msg.add_alternative(html, subtype="html")

        with smtplib.SMTP(self._config.email_smtp_host, self._config.email_smtp_port, timeout=10) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            if self._config.email_username:
                smtp.login(self._config.email_username, self._config.email_password)
            smtp.send_message(msg)


def _allow_rate(sent_at: deque[float], limit: int) -> bool:
    _prune_window(sent_at, seconds=60.0)
    if len(sent_at) >= limit:
        return False
    sent_at.append(time.time())
    return True


def _prune_window(items: deque, seconds: float) -> None:
    cutoff = time.time() - seconds
    while items:
        stamp = items[0][0] if isinstance(items[0], tuple) else items[0]
        if stamp >= cutoff:
            break
        items.popleft()


def _format_direction_change(event: SignalEvent, old_direction: str, new_direction: str) -> str:
    """Format a direction change alert."""
    return (
        f"*Kairos state change* | {event.asset}\n"
        f"{old_direction} -> *{new_direction.upper()}* ({event.confidence:.0%})\n"
        "```text\n"
        f"regime: {event.regime}\n"
        f"eta: {event.estimated_hours:.1f}h\n"
        f"mechanism: {event.mechanism}\n"
        "```"
    )


def _format_anomaly_alert(event: SignalEvent, ctx: dict) -> str:
    """Format an anomaly detection alert."""
    score = float(ctx.get("anomaly_score", 0.0) or 0.0)
    price = ctx.get("current_price")
    price_line = f"\nprice: ${float(price):,.2f}" if price is not None else ""
    return (
        f"*Kairos Anomaly* | {event.asset}\n"
        f"*{event.direction.upper()}* ({event.confidence:.0%})\n"
        "```text\n"
        f"anomaly_score: {score:g}{price_line}\n"
        f"regime: {event.regime}\n"
        f"mechanism: {event.mechanism}\n"
        "```"
    )


def _format_confidence_drop(event: SignalEvent, old_confidence: float) -> str:
    """Format a sharp confidence drop alert."""
    return (
        f"*Kairos Confidence drop* | {event.asset}\n"
        f"previous: {old_confidence:.0%} -> current: {event.confidence:.0%}\n"
        "```text\n"
        f"direction: {event.direction}\n"
        f"regime: {event.regime}\n"
        f"mechanism: {event.mechanism}\n"
        "```"
    )


def _format_daily_digest(events: list[SignalEvent], trades: list) -> str:
    """Format a daily summary of all signals and paper trades."""
    lines = ["Kairos daily digest", "", "Signals:"]
    for event in events:
        lines.append(f"- {event.asset}: {event.direction} {event.confidence:.0%} ({event.regime})")
    if not events:
        lines.append("- none")
    lines.extend(["", "Paper trades:"])
    if trades:
        for trade in trades:
            lines.append(f"- {trade}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def _format_daily_digest_messages(messages: list[str]) -> str:
    return "Kairos daily digest\n\n" + "\n\n---\n\n".join(messages)


def _email_subject(event: SignalEvent) -> str:
    return f"[Kairos] {event.asset} \u2192 {event.direction.upper()} ({event.confidence:.0%})"


def _build_discord_embed(event: SignalEvent, ctx: dict) -> dict:
    price = ctx.get("current_price", ctx.get("price", "n/a"))
    if isinstance(price, (int, float)):
        price_value = f"${price:,.2f}"
    else:
        price_value = str(price)
    return {
        "title": f"Kairos {event.asset} alert",
        "color": _DISCORD_COLORS.get(event.direction, _DISCORD_COLORS["neutral"]),
        "fields": [
            {"name": "Asset", "value": event.asset, "inline": True},
            {"name": "Direction", "value": event.direction.upper(), "inline": True},
            {"name": "Confidence", "value": f"{event.confidence:.0%}", "inline": True},
            {"name": "Regime", "value": event.regime, "inline": True},
            {"name": "Price", "value": price_value, "inline": True},
            {"name": "Mechanism", "value": event.mechanism[:1024], "inline": False},
        ],
        "thumbnail": {"url": _COINGECKO_THUMBNAILS.get(event.asset.upper(), "https://www.coingecko.com/favicon.ico")},
        "timestamp": event.triggered_at.isoformat(),
    }
