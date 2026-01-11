from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel


# =========================================================
# FORCE LOAD .env (WINDOWS SAFE)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


# =========================================================
# ENV HELPERS (robust)
# =========================================================
def _get(key: str, default: str = "") -> str:
    return (os.getenv(key, default) or "").strip()


def _get_int(key: str, default: int) -> int:
    v = _get(key, "")
    if v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    v = _get(key, "")
    if v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _get_bool(key: str, default: bool = False) -> bool:
    v = _get(key, "")
    if v == "":
        return default
    v = v.lower()
    return v in ("1", "true", "yes", "y", "on")


class Settings(BaseModel):
    # =====================================================
    # ENGINE CORE
    # =====================================================
    mode: str = _get("MODE", _get("TRADING_MODE", "SAFE")).upper()

    # Support both LOOP_SECONDS and legacy LOOP_SLEEP_SECONDS
    loop_seconds: int = _get_int("LOOP_SECONDS", _get_int("LOOP_SLEEP_SECONDS", 15))

    # =====================================================
    # TARGET
    # =====================================================
    symbol: str = _get("SYMBOL", "XAUUSDm")

    # =====================================================
    # MT5 CONNECTION
    # =====================================================
    mt5_login: int = _get_int("MT5_LOGIN", 0)
    mt5_password: str = _get("MT5_PASSWORD", "")
    mt5_server: str = _get("MT5_SERVER", "")
    mt5_path: str = _get("MT5_PATH", "")

    # =====================================================
    # DISCORD (PRIMARY OUTPUT)
    # =====================================================
    discord_enabled: bool = _get_bool("DISCORD_ENABLED", True)
    discord_bot_token: str = _get("DISCORD_BOT_TOKEN", "")
    discord_channel_id: str = _get("DISCORD_CHANNEL_ID", "")

    # Anti-spam publish
    signal_cooldown_seconds: int = _get_int("SIGNAL_COOLDOWN_SECONDS", 180)
    watch_status_seconds: int = _get_int("WATCH_STATUS_SECONDS", 600)

    # =====================================================
    # TELEGRAM (optional / legacy)
    # =====================================================
    telegram_bot_token: str = _get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = _get("TELEGRAM_CHAT_ID", "")

    # =====================================================
    # RISK MGMT (currently informational; order engine can use later)
    # =====================================================
    risk_per_trade_pct: float = _get_float("RISK_PER_TRADE_PCT", 1.0)
    max_daily_drawdown_pct: float = _get_float("MAX_DAILY_DRAWDOWN_PCT", 3.0)
    max_open_trades: int = _get_int("MAX_OPEN_TRADES", 3)

    # =====================================================
    # SWING MODE TUNING (Env-configurable, no hardcode)
    # =====================================================
    # Swing requires HTF not ranging & aligned by default
    swing_require_htf_alignment: bool = _get_bool("SWING_REQUIRE_HTF_ALIGNMENT", True)
    swing_disallow_h1_ranging: bool = _get_bool("SWING_DISALLOW_H1_RANGING", True)
    swing_disallow_h4_ranging: bool = _get_bool("SWING_DISALLOW_H4_RANGING", True)

    # ATR floors (can tune if too strict)
    min_atr_m15: float = _get_float("MIN_ATR_M15", 40.0)
    min_atr_h1: float = _get_float("MIN_ATR_H1", 150.0)
    min_atr_h4: float = _get_float("MIN_ATR_H4", 350.0)

    # =====================================================
    # VISUALS
    # =====================================================
    chart_last_n: int = _get_int("CHART_LAST_N", 220)
    artifacts_dir: str = _get("ARTIFACTS_DIR", str((BASE_DIR / "artifacts").resolve()))
    
    # =====================================================
    # AI / INTELLIGENCE (MEGALLM)
    # =====================================================
    # Ganti "USE_GEMINI..." jadi generic "USE_AI..." biar rapi, 
    # tapi kita tetep baca value lama kalau user belum update .env
    use_ai_narrative: bool = _get_bool("USE_AI_NARRATIVE", _get_bool("USE_GEMINI_FOR_SENTIMENT", False))
    
    # MegaLLM (OpenAI Compatible)
    megallm_api_key: str = _get("MEGALLM_API_KEY", _get("OPENAI_API_KEY", "")) 
    megallm_base_url: str = _get("MEGALLM_BASE_URL", "https://ai.megallm.io/v1")
    megallm_model: str = _get("MEGALLM_MODEL", "deepseek-ai/deepseek-v3.1") 

    # =====================================================
    # VALIDATION
    # =====================================================
    def validate(self) -> None:
        errors = []

        # MT5 required for live market feed in this build
        if self.mt5_login <= 0:
            errors.append("MT5_LOGIN belum diisi / invalid")
        if not self.mt5_server:
            errors.append("MT5_SERVER belum diisi")
        
        # Discord optional but if enabled must be complete
        if self.discord_enabled:
            if not self.discord_bot_token:
                errors.append("DISCORD_BOT_TOKEN kosong")
            if not self.discord_channel_id:
                errors.append("DISCORD_CHANNEL_ID kosong")

        # Sanitization
        if self.loop_seconds < 3:
            errors.append("LOOP_SECONDS terlalu kecil (min 3 detik biar gak spam)")

        if self.signal_cooldown_seconds < 30:
            errors.append("SIGNAL_COOLDOWN_SECONDS terlalu kecil (min 30 detik)")

        if self.watch_status_seconds < 30:
            errors.append("WATCH_STATUS_SECONDS terlalu kecil (min 30 detik)")

        if self.chart_last_n < 80:
            errors.append("CHART_LAST_N terlalu kecil (min 80)")
            
        # AI Validation
        if self.use_ai_narrative:
            if not self.megallm_api_key:
                errors.append("USE_AI_NARRATIVE=true tapi MEGALLM_API_KEY kosong")

        if errors:
            raise RuntimeError("❌ CONFIG ERROR:\n- " + "\n- ".join(errors))

    # Convenience computed dicts
    @property
    def min_atr(self) -> dict:
        return {
            "M15": float(self.min_atr_m15),
            "H1": float(self.min_atr_h1),
            "H4": float(self.min_atr_h4),
        }


settings = Settings()
settings.validate()