import pandas as pd
from typing import Dict, Optional


def _prev_swing_levels(df: pd.DataFrame, n: int = 50) -> Dict[str, float]:
    """
    Simple liquidity reference levels: recent high/low excluding last bar.
    """
    window = df.iloc[-(n+1):-1] if len(df) > n+1 else df.iloc[:-1]
    if len(window) < 5:
        return {"prev_high": 0.0, "prev_low": 0.0}
    return {
        "prev_high": float(window["high"].max()),
        "prev_low": float(window["low"].min()),
    }


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 50, wick_ratio: float = 0.55) -> Dict[str, Optional[str]]:
    """
    Detect liquidity sweep / stop hunt on last candle:
      - Sweeps above prev_high then closes back below it => 'SWEEP_HIGH'
      - Sweeps below prev_low then closes back above it => 'SWEEP_LOW'

    wick_ratio ensures wick dominance to reduce false signals.
    Returns dict:
      {
        "sweep": "SWEEP_HIGH" | "SWEEP_LOW" | None,
        "level": float,
        "notes": str
      }
    """
    if len(df) < lookback + 5:
        return {"sweep": None, "level": None, "notes": "insufficient_bars"}

    levels = _prev_swing_levels(df, n=lookback)
    prev_high = levels["prev_high"]
    prev_low = levels["prev_low"]

    last = df.iloc[-1]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])

    rng = max(h - l, 1e-9)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # Sweep high: break above, close back below
    if prev_high > 0 and h > prev_high and c < prev_high:
        if upper_wick / rng >= wick_ratio:
            return {"sweep": "SWEEP_HIGH", "level": prev_high, "notes": "wick_dominant"}
        return {"sweep": "SWEEP_HIGH", "level": prev_high, "notes": "weak_wick"}

    # Sweep low: break below, close back above
    if prev_low > 0 and l < prev_low and c > prev_low:
        if lower_wick / rng >= wick_ratio:
            return {"sweep": "SWEEP_LOW", "level": prev_low, "notes": "wick_dominant"}
        return {"sweep": "SWEEP_LOW", "level": prev_low, "notes": "weak_wick"}

    return {"sweep": None, "level": None, "notes": "none"}


def detect_fake_breakout(df: pd.DataFrame, lookback: int = 50) -> Dict[str, Optional[str]]:
    """
    Fake breakout heuristic:
      - Close breaks level then next close returns inside range.
    We detect only for last 2 candles.

    Returns:
      { "fake": "FAKE_UP" | "FAKE_DOWN" | None, "level": float | None }
    """
    if len(df) < lookback + 10:
        return {"fake": None, "level": None}

    levels = _prev_swing_levels(df.iloc[:-2], n=lookback)  # exclude last 2 candles
    prev_high = levels["prev_high"]
    prev_low = levels["prev_low"]

    c1 = float(df["close"].iloc[-2])
    c2 = float(df["close"].iloc[-1])

    # Fake up: candle-1 closes above prev_high, candle-2 closes back below
    if prev_high > 0 and c1 > prev_high and c2 < prev_high:
        return {"fake": "FAKE_UP", "level": prev_high}

    # Fake down: candle-1 closes below prev_low, candle-2 closes back above
    if prev_low > 0 and c1 < prev_low and c2 > prev_low:
        return {"fake": "FAKE_DOWN", "level": prev_low}

    return {"fake": None, "level": None}
