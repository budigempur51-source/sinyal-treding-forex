from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple
import pandas as pd


Side = Literal["BUY", "SELL"]
Bias = Literal["BULLISH", "BEARISH", "RANGING"]


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    side: Side
    market_bias: Bias
    entry_zone: Tuple[float, float]
    sl: float
    tp1: float
    tp2: float
    tp3: float
    rr: float
    confidence: float
    reason: str


def _clamp_conf(x: float) -> float:
    return max(0.0, min(100.0, x))


def _rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return 0.0
    return reward / risk


def build_trade_plan(
    symbol: str,
    htf_bias: Bias,
    ltf_bias: Bias,
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    zones_m15: Dict,
    liquidity_m15: Dict,
    fakeout_m15: Dict,
    mode: str = "SAFE",
) -> Optional[TradePlan]:
    """
    Rules (SAFE):
      - Requires HTF bias != RANGING and aligns with LTF bias
      - Prefer entries at Demand zone for BUY, Supply zone for SELL
      - Avoid when sweep/fakeout contradicts direction
      - SL beyond zone with ATR buffer
      - TP based on RR ladder using ATR & recent range

    Returns TradePlan or None if no valid setup.
    """

    if htf_bias == "RANGING":
        return None

    # Align
    if ltf_bias != htf_bias:
        return None

    last_close = float(m15_df["close"].iloc[-1])
    atr_m15 = float(m15_df["atr14"].iloc[-1]) if "atr14" in m15_df.columns else 0.0

    # Liquidity filters
    sweep = liquidity_m15.get("sweep")
    fake = fakeout_m15.get("fake")

    # Determine side
    if htf_bias == "BULLISH":
        side: Side = "BUY"
        # If just swept HIGH (stop hunt above) it's bearish pressure -> avoid BUY
        if sweep == "SWEEP_HIGH" or fake == "FAKE_UP":
            return None
        zone = zones_m15.get("DEMAND")
        if not zone:
            return None

        entry_low = float(zone["low"])
        entry_high = float(zone["high"])
        entry_mid = (entry_low + entry_high) / 2.0

        sl = entry_low - max(atr_m15 * 0.35, 1.0)
        # TP ladder
        tp1 = entry_mid + max(atr_m15 * 1.0, 1.0)
        tp2 = entry_mid + max(atr_m15 * 2.0, 1.0)
        tp3 = entry_mid + max(atr_m15 * 3.0, 1.0)

    else:  # BEARISH
        side = "SELL"
        # If just swept LOW it's bullish pressure -> avoid SELL
        if sweep == "SWEEP_LOW" or fake == "FAKE_DOWN":
            return None
        zone = zones_m15.get("SUPPLY")
        if not zone:
            return None

        entry_low = float(zone["low"])
        entry_high = float(zone["high"])
        entry_mid = (entry_low + entry_high) / 2.0

        sl = entry_high + max(atr_m15 * 0.35, 1.0)
        tp1 = entry_mid - max(atr_m15 * 1.0, 1.0)
        tp2 = entry_mid - max(atr_m15 * 2.0, 1.0)
        tp3 = entry_mid - max(atr_m15 * 3.0, 1.0)

    # RR based on TP2 as main target
    rr_val = _rr(entry_mid, sl, tp2)

    # Confidence scoring (simple but disciplined)
    conf = 50.0

    # Stronger if M15 is BOS
    # caller can pass event via ltf_bias only; we’ll infer a bit from EMA alignment
    ema50 = float(m15_df["ema50"].iloc[-1]) if "ema50" in m15_df.columns else last_close
    ema200 = float(m15_df["ema200"].iloc[-1]) if "ema200" in m15_df.columns else last_close

    if side == "BUY" and ema50 > ema200:
        conf += 10
    if side == "SELL" and ema50 < ema200:
        conf += 10

    # RSI sanity
    rsi = float(m15_df["rsi14"].iloc[-1]) if "rsi14" in m15_df.columns else 50.0
    if side == "BUY" and 45 <= rsi <= 70:
        conf += 8
    if side == "SELL" and 30 <= rsi <= 55:
        conf += 8

    # Penalize if close is far from zone (missed entry)
    dist = abs(last_close - entry_mid)
    if atr_m15 > 0 and dist > atr_m15 * 2.0:
        conf -= 15

    # Reward RR
    if rr_val >= 2.0:
        conf += 10
    elif rr_val >= 1.5:
        conf += 6
    elif rr_val < 1.2:
        conf -= 10

    # SAFE mode cap
    if mode.upper() == "SAFE":
        conf = min(conf, 80.0)

    conf = _clamp_conf(conf)

    reason = (
        f"HTF aligned ({htf_bias}); zone-based entry; "
        f"liquidity filters OK; ATR={atr_m15:.2f}; RR(TP2)={rr_val:.2f}"
    )

    return TradePlan(
        symbol=symbol,
        side=side,
        market_bias=htf_bias,
        entry_zone=(entry_low, entry_high),
        sl=float(sl),
        tp1=float(tp1),
        tp2=float(tp2),
        tp3=float(tp3),
        rr=float(rr_val),
        confidence=float(conf),
        reason=reason,
    )
