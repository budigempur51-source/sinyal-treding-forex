from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    Professional candlestick renderer.
    """
    x = np.arange(len(ohlc))
    o = ohlc["open"].to_numpy()
    h = ohlc["high"].to_numpy()
    l = ohlc["low"].to_numpy()
    c = ohlc["close"].to_numpy()

    # Define colors
    col_up = '#26a69a'   # Greenish
    col_dn = '#ef5350'   # Reddish
    
    # Vectorized color array
    colors = np.where(c >= o, col_up, col_dn)

    # Wicks (Vlines)
    ax.vlines(x, l, h, colors=colors, linewidth=1, alpha=0.9)
    
    # Bodies (Patches)
    for i in range(len(ohlc)):
        body_low = min(o[i], c[i])
        height = abs(c[i] - o[i])
        # Minimum height for doji visibility
        height = max(height, (h[i]-l[i])*0.05 if (h[i]-l[i])>0 else 0.0001)
        
        rect = plt.Rectangle(
            (x[i] - 0.35, body_low),
            0.7,
            height,
            facecolor=colors[i],
            edgecolor=colors[i],
            alpha=1.0
        )
        ax.add_patch(rect)


def _zone_box(ax, x0: int, x1: int, low: float, high: float, label: str):
    # Supply = Red tint, Demand = Green tint
    color = 'green' if "DEMAND" in label else 'red'
    
    # Fill
    ax.add_patch(
        plt.Rectangle(
            (x0, low),
            x1 - x0,
            high - low,
            facecolor=color,
            alpha=0.15,
            edgecolor=None
        )
    )
    # Border Lines
    ax.hlines(y=[low, high], xmin=x0, xmax=x1, colors=color, linestyles='--', linewidth=0.8, alpha=0.6)
    
    # Label
    mid_y = (low + high) / 2
    ax.text(x0 + 2, high, f"{label}\n[{low:.2f}-{high:.2f}]", 
            color=color, fontsize=8, fontweight='bold', va='bottom', alpha=0.8)


def render_swing_chart(
    df: pd.DataFrame,
    zones: Dict[str, Optional[Dict[str, Any]]],
    plan: Optional[RenderPlan],
    out_path: str,
    last_n: int = 220,
):
    """
    Creates Institutional-Style PNG chart.
    """
    if df is None or len(df) < 50:
        raise RuntimeError("Not enough data to render chart")

    work = df.copy()
    work = work.tail(last_n).reset_index(drop=True)

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(work.columns)):
        raise RuntimeError(f"DF missing columns: {required - set(work.columns)}")

    # Dark Theme Background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    
    # Background tweaks
    ax.set_facecolor('#0b0e11') # Binance-like dark blue/black
    fig.patch.set_facecolor('#0b0e11')

    # Plot Candles
    _plot_candles(ax, work)

    x = np.arange(len(work))
    last_idx = len(work) - 1
    last_price = float(work["close"].iloc[-1])

    # EMA plots
    if "ema50" in work.columns:
        ax.plot(x, work["ema50"].to_numpy(), color='#2962ff', linewidth=1.5, label="EMA50", alpha=0.8)
    if "ema200" in work.columns:
        ax.plot(x, work["ema200"].to_numpy(), color='#ff6d00', linewidth=1.5, label="EMA200", alpha=0.8)

    # Current Price Line
    ax.axhline(last_price, color='white', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.text(last_idx + 1, last_price, f" {last_price:.2f}", color='white', va='center', fontsize=9)

    # Zones
    x0 = max(0, len(work) - 160)
    x1 = len(work) + 15 # Extend to future

    if zones and zones.get("DEMAND"):
        z = zones["DEMAND"]
        _zone_box(ax, x0, x1, float(z["low"]), float(z["high"]), "DEMAND")

    if zones and zones.get("SUPPLY"):
        z = zones["SUPPLY"]
        _zone_box(ax, x0, x1, float(z["low"]), float(z["high"]), "SUPPLY")

    # Plan Overlays (Prediction)
    if plan:
        entry = float(plan.entry)
        sl = float(plan.sl)
        tp1 = float(plan.tp1)
        tp2 = float(plan.tp2)
        tp3 = float(plan.tp3)
        
        # Entry Line
        ax.axhline(entry, color='gray', linestyle='--', linewidth=1)
        
        # Stop Loss Area
        if plan.side.upper() == "BUY":
            rect_sl = plt.Rectangle((last_idx, sl), 15, entry-sl, facecolor='#ef5350', alpha=0.2)
            rect_tp = plt.Rectangle((last_idx, entry), 15, tp3-entry, facecolor='#26a69a', alpha=0.2)
        else:
            rect_sl = plt.Rectangle((last_idx, entry), 15, sl-entry, facecolor='#ef5350', alpha=0.2)
            rect_tp = plt.Rectangle((last_idx, tp3), 15, entry-tp3, facecolor='#26a69a', alpha=0.2)
            
        ax.add_patch(rect_sl)
        ax.add_patch(rect_tp)

        # Labels
        ax.text(last_idx + 2, sl, f"SL: {sl:.2f}", color='#ef5350', fontsize=10, fontweight='bold')
        ax.text(last_idx + 2, tp1, f"TP1: {tp1:.2f}", color='#26a69a', fontsize=9)
        ax.text(last_idx + 2, tp2, f"TP2: {tp2:.2f}", color='#26a69a', fontsize=9)
        ax.text(last_idx + 2, tp3, f"TP3: {tp3:.2f}", color='#26a69a', fontsize=9)

        # Arrow Logic
        if plan.side.upper() == "BUY":
            ax.arrow(last_idx, entry, 5, (tp1-entry)*0.5, head_width=2, head_length=2, fc='white', ec='white', alpha=0.5)
        else:
            ax.arrow(last_idx, entry, 5, (tp1-entry)*0.5, head_width=2, head_length=2, fc='white', ec='white', alpha=0.5)

    # Title & Cosmetics
    title_text = f"{plan.symbol if plan else df.columns.name or 'CHART'} | M15 SWING"
    ax.set_title(title_text, color='white', fontsize=12, pad=10)
    
    # Limits
    ax.set_xlim(0, len(work) + 15)
    
    # Smart Y-Lim
    y_vals = work["high"].tail(80).to_list() + work["low"].tail(80).to_list()
    if plan:
        y_vals.extend([plan.sl, plan.tp2])
    
    y_min, y_max = min(y_vals), max(y_vals)
    pad = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.grid(True, color='#2c3e50', alpha=0.3, linestyle='--')
    ax.legend(loc="upper left", facecolor='#1e272e', edgecolor='none', labelcolor='white')
    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#2c3e50')
    ax.spines['left'].set_color('#2c3e50')
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')

    outp = Path(out_path)
    _ensure_dir(outp)
    plt.tight_layout()
    plt.savefig(outp.as_posix(), dpi=120, facecolor='#0b0e11')
    plt.close(fig)