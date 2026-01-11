import pandas as pd
from typing import Dict, Literal, Optional

ZoneType = Literal["SUPPLY", "DEMAND"]

def detect_zones(
    df: pd.DataFrame,
    impulse_pct: float = 0.6,
    base_candles: int = 3
) -> Dict[str, Optional[Dict]]:
    """
    Detect last Supply / Demand zone using:
    - Base (small body candles)
    - Impulse move after base

    Returns:
      {
        "SUPPLY": {high, low, time} | None
        "DEMAND": {high, low, time} | None
      }
    """

    out = {
        "SUPPLY": None,
        "DEMAND": None
    }

    df = df.copy()
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = (df["high"] - df["low"]).replace(0, 1e-9)
    df["body_ratio"] = df["body"] / df["range"]

    # scan from older → recent
    for i in range(base_candles + 2, len(df) - 1):
        base = df.iloc[i - base_candles:i]
        impulse = df.iloc[i]

        # Base candles = small bodies
        if (base["body_ratio"] < 0.35).all():

            # Bullish impulse → DEMAND
            if impulse["close"] > impulse["open"]:
                move_pct = (impulse["close"] - impulse["open"]) / impulse["range"]
                if move_pct >= impulse_pct:
                    zone_low = base["low"].min()
                    zone_high = base["high"].max()
                    out["DEMAND"] = {
                        "low": float(zone_low),
                        "high": float(zone_high),
                        "time": base.index[-1]
                    }

            # Bearish impulse → SUPPLY
            if impulse["close"] < impulse["open"]:
                move_pct = (impulse["open"] - impulse["close"]) / impulse["range"]
                if move_pct >= impulse_pct:
                    zone_low = base["low"].min()
                    zone_high = base["high"].max()
                    out["SUPPLY"] = {
                        "low": float(zone_low),
                        "high": float(zone_high),
                        "time": base.index[-1]
                    }

    return out
