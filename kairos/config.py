import os
import pathlib

from dotenv import load_dotenv

load_dotenv()

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "kairos/0.1.0")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
DB_PATH = os.getenv("KAIROS_DB_PATH", str(pathlib.Path.home() / ".kairos" / "kairos.db"))
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
SMTP_HOST = os.getenv("SMTP_HOST", "")
try:
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or "587")
except ValueError:
    SMTP_PORT = 587
# Canonical SMTP credentials — SMTP_USERNAME and EMAIL_FROM/EMAIL_TO are the
# authoritative env vars. Legacy SMTP_USER / ALERT_EMAIL_* aliases are still
# accepted so existing deployments do not break.
SMTP_USERNAME: str = os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER", "")
SMTP_USER = SMTP_USERNAME  # backward-compat alias
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM: str = os.getenv("EMAIL_FROM") or os.getenv("ALERT_EMAIL_FROM", "")
EMAIL_TO: str = os.getenv("EMAIL_TO") or os.getenv("ALERT_EMAIL_TO", "")
ALERT_EMAIL_FROM = EMAIL_FROM  # backward-compat alias
ALERT_EMAIL_TO = EMAIL_TO  # backward-compat alias

CIRCUIT_BREAKER_CONFIG: dict[str, dict] = {
    "coingecko": {"failure_threshold": 5, "recovery_timeout": 60.0},
    "fng": {"failure_threshold": 5, "recovery_timeout": 60.0},
    "github": {"failure_threshold": 3, "recovery_timeout": 120.0},
    "solana_rpc": {"failure_threshold": 5, "recovery_timeout": 30.0},
    "macro": {"failure_threshold": 3, "recovery_timeout": 300.0},
    "binance": {"failure_threshold": 5, "recovery_timeout": 30.0},
}
