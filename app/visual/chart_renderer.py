from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class RenderPlan:
    symbol: str
    tf: str
    side: str  # "BUY" or "SELL"
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    entry_zone: Tuple[float, float]
    market_bias: str
    confidence: float
    reason: str


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _plot_candles(ax, ohlc: pd.DataFrame):
    """
    Minimal candlestick renderer (no external mplfinance).
    Requires columns: open, high, low, close
    """
    x = np.arange(len(ohlc))
    o = ohlc["open"].to_numpy()
    h = ohlc["high"].to_numpy()
    l = ohlc["low"].to_numpy()
    c = ohlc["close"].to_numpy()

    for i in range(len(ohlc)):
        up = c[i] >= o[i]
        # wick
        ax.vlines(x[i], l[i], h[i], linewidth=1)
        # body
        body_low = min(o[i], c[i])
        body_high = max(o[i], c[i])
        height = max(body_high - body_low, 0.0000001)
        ax.add_patch(
            plt.Rectangle(
                (x[i] - 0.35, body_low),
                0.7,
                height,
                fill=True,
                alpha=0.9,
            )
        )


def _zone_box(ax, x0: int, x1: int, low: float, high: float, label: str):
    ax.add_patch(
        plt.Rectangle(
            (x0, low),
            x1 - x0,
            high - low,
            fill=True,
            alpha=0.12,
        )
    )
    ax.text(x0 + 1, high, f" {label} [{low:.2f}-{high:.2f}]", va="bottom", fontsize=9)


def render_swing_chart(
    df: pd.DataFrame,
    zones: Dict[str, Optional[Dict[str, Any]]],
    plan: Optional[RenderPlan],
    out_path: str,
    last_n: int = 220,
):
    """
    Creates PNG chart:
    - Candles
    - EMA50, EMA200 (if available)
    - DEMAND/SUPPLY zones (box)
    - Entry arrow + SL/TP lines (if plan exists)
    """
    if df is None or len(df) < 50:
        raise RuntimeError("Not enough data to render chart")

    work = df.copy()

    # Normalize timestamp column name
    if "time" in work.columns:
        times = pd.to_datetime(work["time"])
    elif "datetime" in work.columns:
        times = pd.to_datetime(work["datetime"])
    else:
        times = pd.RangeIndex(start=0, stop=len(work), step=1)

    work = work.tail(last_n).reset_index(drop=True)
    times = times.tail(last_n).reset_index(drop=True) if hasattr(times, "tail") else times

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(work.columns)):
        raise RuntimeError(f"DF missing columns: {required - set(work.columns)}")

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111)

    _plot_candles(ax, work)

    x = np.arange(len(work))

    # EMA plots if present
    if "ema50" in work.columns:
        ax.plot(x, work["ema50"].to_numpy(), linewidth=1.4, label="EMA50")
    if "ema200" in work.columns:
        ax.plot(x, work["ema200"].to_numpy(), linewidth=1.4, label="EMA200")

    # Zones
    x0 = max(0, len(work) - 140)
    x1 = len(work) - 1

    if zones and zones.get("DEMAND"):
        z = zones["DEMAND"]
        _zone_box(ax, x0, x1, float(z["low"]), float(z["high"]), "DEMAND")

    if zones and zones.get("SUPPLY"):
        z = zones["SUPPLY"]
        _zone_box(ax, x0, x1, float(z["low"]), float(z["high"]), "SUPPLY")

    # Plan overlays
    if plan:
        entry = float(plan.entry)
        sl = float(plan.sl)
        tp1 = float(plan.tp1)
        tp2 = float(plan.tp2)
        tp3 = float(plan.tp3)

        ax.axhline(entry, linewidth=1.6, linestyle="--")
        ax.axhline(sl, linewidth=1.2, linestyle="-")
        ax.axhline(tp1, linewidth=1.2, linestyle="-")
        ax.axhline(tp2, linewidth=1.2, linestyle="-")
        ax.axhline(tp3, linewidth=1.2, linestyle="-")

        # Arrow at last candle
        last_x = len(work) - 1
        if plan.side.upper() == "BUY":
            ax.annotate(
                "BUY",
                xy=(last_x, entry),
                xytext=(last_x - 25, entry - (work["atr14"].iloc[-1] if "atr14" in work.columns else 50)),
                arrowprops=dict(arrowstyle="->", linewidth=2),
                fontsize=12,
                fontweight="bold",
            )
        else:
            ax.annotate(
                "SELL",
                xy=(last_x, entry),
                xytext=(last_x - 25, entry + (work["atr14"].iloc[-1] if "atr14" in work.columns else 50)),
                arrowprops=dict(arrowstyle="->", linewidth=2),
                fontsize=12,
                fontweight="bold",
            )

        ax.text(2, entry, f" ENTRY {entry:.2f}", fontsize=10, va="bottom")
        ax.text(2, sl, f" SL {sl:.2f}", fontsize=10, va="bottom")
        ax.text(2, tp1, f" TP1 {tp1:.2f}", fontsize=10, va="bottom")
        ax.text(2, tp2, f" TP2 {tp2:.2f}", fontsize=10, va="bottom")
        ax.text(2, tp3, f" TP3 {tp3:.2f}", fontsize=10, va="bottom")

    # Title & cosmetics
    title = f"{plan.symbol if plan else ''} SWING M15 — EMA + ZONES"
    ax.set_title(title)
    ax.set_xlim(0, len(work) + 2)

    # Set y-limits with padding
    y_min = float(work["low"].min())
    y_max = float(work["high"].max())
    pad = (y_max - y_min) * 0.08 if (y_max - y_min) > 0 else 50
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper left")

    outp = Path(out_path)
    _ensure_dir(outp)
    plt.tight_layout()
    plt.savefig(outp.as_posix(), dpi=160)
    plt.close(fig)
