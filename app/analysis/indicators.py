import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns: time, open, high, low, close, volume
    Output adds:
      - ema50, ema200
      - rsi14
      - atr14
      - vol_sma20 (tick volume SMA)
      - vol_z20 (z-score volume vs last 20)
    """
    out = df.copy()

    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Indicators
    out["ema50"] = EMAIndicator(close=out["close"], window=50).ema_indicator()
    out["ema200"] = EMAIndicator(close=out["close"], window=200).ema_indicator()
    out["rsi14"] = RSIIndicator(close=out["close"], window=14).rsi()
    out["atr14"] = AverageTrueRange(high=out["high"], low=out["low"], close=out["close"], window=14).average_true_range()

    # Volume features (tick volume)
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    vol_mean = out["volume"].rolling(20).mean()
    vol_std = out["volume"].rolling(20).std(ddof=0)
    out["vol_z20"] = (out["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    return out


def last_indicator_snapshot(df: pd.DataFrame) -> dict:
    """
    Returns last-bar snapshot (safe casting).
    """
    last = df.iloc[-1]
    def f(x, d=0.0):
        try:
            if pd.isna(x):
                return d
            return float(x)
        except Exception:
            return d

    return {
        "close": f(last.get("close")),
        "ema50": f(last.get("ema50")),
        "ema200": f(last.get("ema200")),
        "rsi14": f(last.get("rsi14")),
        "atr14": f(last.get("atr14")),
        "volume": f(last.get("volume")),
        "vol_sma20": f(last.get("vol_sma20")),
        "vol_z20": f(last.get("vol_z20")),
        "time_utc": last.get("time").strftime("%Y-%m-%d %H:%M:%S UTC") if "time" in last and hasattr(last.get("time"), "strftime") else "",
    }
