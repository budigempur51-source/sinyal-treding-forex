import pandas as pd
from typing import Literal, Dict, Optional

StructureBias = Literal["BULLISH", "BEARISH", "RANGING"]
StructureEvent = Literal["BOS", "CHoCH", "NONE"]

def detect_swings(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Mark swing highs/lows using fractal-style logic.
    Adds columns:
      - swing_high (bool)
      - swing_low (bool)
    """
    out = df.copy()
    out["swing_high"] = False
    out["swing_low"] = False

    for i in range(lookback, len(out) - lookback):
        current_high = out["high"].iloc[i]
        current_low = out["low"].iloc[i]

        # Fractal High: High ditengah lebih tinggi dari n candle kiri kanan
        if current_high == max(out["high"].iloc[i - lookback : i + lookback + 1]):
            out.at[out.index[i], "swing_high"] = True

        # Fractal Low: Low ditengah lebih rendah dari n candle kiri kanan
        if current_low == min(out["low"].iloc[i - lookback : i + lookback + 1]):
            out.at[out.index[i], "swing_low"] = True

    return out


def extract_last_swings(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get last confirmed swing high & low prices.
    """
    swings_high = df[df["swing_high"]]
    swings_low = df[df["swing_low"]]

    last_high = float(swings_high["high"].iloc[-1]) if len(swings_high) > 0 else 0.0
    last_low = float(swings_low["low"].iloc[-1]) if len(swings_low) > 0 else 0.0

    return {
        "last_swing_high": last_high,
        "last_swing_low": last_low,
    }


def analyze_structure(df: pd.DataFrame) -> Dict[str, str]:
    """
    Determine structure bias and event (BOS / CHoCH).
    IMPROVED: Uses EMA context to resolve 'RANGING' conditions during trends.
    """
    # 1. Detect Fractals
    df = detect_swings(df, lookback=3) 

    swings = extract_last_swings(df)
    last_high = swings["last_swing_high"]
    last_low = swings["last_swing_low"]

    close = float(df["close"].iloc[-1])
    
    # Ambil data EMA (pastikan add_indicators sudah dijalankan di main.py)
    ema50 = float(df["ema50"].iloc[-1]) if "ema50" in df.columns else 0.0
    ema200 = float(df["ema200"].iloc[-1]) if "ema200" in df.columns else 0.0

    bias: StructureBias = "RANGING"
    event: StructureEvent = "NONE"

    # 2. Pure Structure Logic (Fractal Breakout)
    if last_high > 0 and close > last_high:
        bias = "BULLISH"
        event = "BOS"

    elif last_low > 0 and close < last_low:
        bias = "BEARISH"
        event = "BOS"

    # CHoCH logic (Change of Character) - Reversal Detection
    # Bullish tapi jebol Low -> Bearish
    if bias == "BULLISH" and last_low > 0 and close < last_low:
        bias = "BEARISH"
        event = "CHoCH"
    
    # Bearish tapi jebol High -> Bullish
    if bias == "BEARISH" and last_high > 0 and close > last_high:
        bias = "BULLISH"
        event = "CHoCH"

    # 3. EMA Context / Trend Continuation Logic (The "Smart" Fix)
    # Jika struktur bilang RANGING (karena belum break high/low baru),
    # tapi harga trending kuat di atas/bawah EMA, kita override jadi trending.
    if bias == "RANGING":
        if ema50 > 0 and ema200 > 0:
            # Bullish Context: Price > EMA50 > EMA200
            if close > ema50 and ema50 > ema200:
                bias = "BULLISH"
                # Event tetap NONE karena ini bukan BOS struktural, tapi validasi tren
            
            # Bearish Context: Price < EMA50 < EMA200
            elif close < ema50 and ema50 < ema200:
                bias = "BEARISH"
                # Event tetap NONE

    return {
        "bias": bias,
        "event": event,
        "last_swing_high": f"{last_high:.2f}" if last_high else "-",
        "last_swing_low": f"{last_low:.2f}" if last_low else "-",
    }