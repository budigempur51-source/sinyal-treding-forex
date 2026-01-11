import pandas as pd
from typing import Literal, Dict, List

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
        high = out["high"].iloc[i]
        low = out["low"].iloc[i]

        if high == max(out["high"].iloc[i - lookback : i + lookback + 1]):
            out.at[out.index[i], "swing_high"] = True

        if low == min(out["low"].iloc[i - lookback : i + lookback + 1]):
            out.at[out.index[i], "swing_low"] = True

    return out


def extract_last_swings(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get last confirmed swing high & low prices.
    """
    swings_high = df[df["swing_high"]]
    swings_low = df[df["swing_low"]]

    last_high = float(swings_high["high"].iloc[-1]) if len(swings_high) else 0.0
    last_low = float(swings_low["low"].iloc[-1]) if len(swings_low) else 0.0

    return {
        "last_swing_high": last_high,
        "last_swing_low": last_low,
    }


def analyze_structure(df: pd.DataFrame) -> Dict[str, str]:
    """
    Determine structure bias and event (BOS / CHoCH).
    Uses close vs last swing levels.
    """
    df = detect_swings(df)

    swings = extract_last_swings(df)
    last_high = swings["last_swing_high"]
    last_low = swings["last_swing_low"]

    close = float(df["close"].iloc[-1])

    bias: StructureBias = "RANGING"
    event: StructureEvent = "NONE"

    if last_high > 0 and close > last_high:
        bias = "BULLISH"
        event = "BOS"

    elif last_low > 0 and close < last_low:
        bias = "BEARISH"
        event = "BOS"

    # CHoCH logic (simple & safe)
    if bias == "BULLISH" and last_low > 0 and close < last_low:
        bias = "BEARISH"
        event = "CHoCH"

    if bias == "BEARISH" and last_high > 0 and close > last_high:
        bias = "BULLISH"
        event = "CHoCH"

    return {
        "bias": bias,
        "event": event,
        "last_swing_high": f"{last_high:.2f}" if last_high else "-",
        "last_swing_low": f"{last_low:.2f}" if last_low else "-",
    }
