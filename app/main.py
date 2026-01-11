from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from app.config import settings
from app.data.mt5_feed import MT5Feed

from app.analysis.indicators import add_indicators
from app.analysis.structure import analyze_structure
from app.analysis.no_trade_gate import no_trade_gate
from app.analysis.zones import detect_zones
from app.analysis.liquidity import detect_liquidity_sweep, detect_fake_breakout
from app.analysis.ai_megallm import analyze_market_with_megallm

from app.signal.entry_engine import build_trade_plan

from app.notify.discord_bot import send_discord_embed, send_discord_embed_with_image
from app.visual.chart_renderer import render_swing_chart, RenderPlan


# =========================================================
# SWING CONFIG
# =========================================================
TIMEFRAMES = ["M1", "M5", "M15", "H1", "H4"]
ZONE_TFS = ["M15", "H1", "H4"]


# =========================================================
# UTIL: formatting
# =========================================================
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_zones(z: Dict[str, Any]) -> str:
    if not z:
        return "NOZ"
    out = []
    if z.get("DEMAND"):
        d = z["DEMAND"]
        out.append(f"D[{d['low']:.2f}-{d['high']:.2f}]")
    if z.get("SUPPLY"):
        s = z["SUPPLY"]
        out.append(f"S[{s['low']:.2f}-{s['high']:.2f}]")
    return " ".join(out) if out else "NOZ"


def _swing_gate(tf_results: dict) -> Tuple[bool, str]:
    """
    Swing rules (RELAXED & TREND FOLLOWING):
    1. Jika H1 & H4 searah (Aligned) -> GAS.
    2. Jika H4 RANGING, tapi H1 Trending -> GAS (Follow H1 Momentum).
    3. Cuma blokir kalau H1 & H4 tabrakan arah (Bull vs Bear).
    """
    h4_bias = tf_results["H4"]["bias"]
    h1_bias = tf_results["H1"]["bias"]

    # 1. Konflik Keras (Bull vs Bear) -> Bahaya, Minggir.
    if h1_bias == "BULLISH" and h4_bias == "BEARISH":
        return False, f"HTF Conflict (H1 Bull vs H4 Bear)"
    
    if h1_bias == "BEARISH" and h4_bias == "BULLISH":
        return False, f"HTF Conflict (H1 Bear vs H4 Bull)"

    # 2. H1 Ranging = No Trade (Kita butuh momentum di H1)
    if h1_bias == "RANGING":
        return False, "HTF ranging (H1 Neutral)"

    # 3. H4 Ranging = OK (Kita izinkan H1 yang nyetir)
    # Kita override setting 'swing_disallow_h4_ranging' disini biar trading jalan.

    return True, "OK"


def _format_trade_text(plan, ai_analysis: str = "") -> str:
    side = plan.side.upper()
    entry_lo, entry_hi = plan.entry_zone

    lines = []
    lines.append(f"**PAIR**: `{plan.symbol}`")
    lines.append(f"**MODE**: `SWING` | **ENTRY TF**: `M15`")
    lines.append("")
    lines.append(f"**Market Bias**: `{plan.market_bias}` | **Side**: `{side}`")
    lines.append("")
    lines.append(f"**Entry Zone**: `{entry_lo:.2f} → {entry_hi:.2f}`")
    lines.append(f"**Stop Loss**: `{plan.sl:.2f}`")
    lines.append("")
    lines.append("**Take Profit**")
    lines.append(f"- TP1: `{plan.tp1:.2f}`")
    lines.append(f"- TP2: `{plan.tp2:.2f}`")
    lines.append(f"- TP3: `{plan.tp3:.2f}`")
    lines.append("")
    lines.append(f"**RR (to TP2)**: `{plan.rr:.2f}`")
    lines.append(f"**Confidence**: `{plan.confidence:.1f}%`")
    lines.append("")
    
    if ai_analysis and "Disabled" not in ai_analysis:
        lines.append("🧠 **AI Analysis (DeepSeek V3)**")
        lines.append(ai_analysis)
        lines.append("")
        
    lines.append(f"**Technical Reason**: {plan.reason}")
    return "\n".join(lines)


def _format_watch_report(symbol: str, tf_results: dict, zones_m15: dict, liq: dict, fake: dict, reason: str, ai_analysis: str = "") -> str:
    lines = []
    lines.append(f"**PAIR**: `{symbol}`")
    lines.append(f"**MODE**: `SWING WATCH` | **STATUS**: `NO TRADE`")
    lines.append("")
    
    # AI Analysis di paling atas biar kebaca duluan
    if ai_analysis and "Disabled" not in ai_analysis:
        lines.append("🧠 **Market Insight (DeepSeek V3)**")
        lines.append(ai_analysis)
        lines.append("")

    lines.append("**HTF STATUS**")
    lines.append(f"- H4: `{tf_results['H4']['bias']}` | ATR `{tf_results['H4']['atr']:.2f}`")
    lines.append(f"- H1: `{tf_results['H1']['bias']}` | ATR `{tf_results['H1']['atr']:.2f}`")
    lines.append("")
    lines.append("**ENTRY TF (M15)**")
    lines.append(f"- M15: `{tf_results['M15']['bias']}` | Zones: `{_fmt_zones(zones_m15)}`")
    lines.append("")
    lines.append(f"🚫 **REASON**: {reason}")
    return "\n".join(lines)


# =========================================================
# PUBLISH HELPERS (cooldowns)
# =========================================================
class Cooldowns:
    def __init__(self):
        self.last_signal_ts = 0.0
        self.last_watch_ts = 0.0

    def can_signal(self) -> bool:
        return (time.time() - self.last_signal_ts) >= settings.signal_cooldown_seconds

    def can_watch(self) -> bool:
        return (time.time() - self.last_watch_ts) >= settings.watch_status_seconds

    def mark_signal(self):
        self.last_signal_ts = time.time()

    def mark_watch(self):
        self.last_watch_ts = time.time()


# =========================================================
# ENGINE LOOP
# =========================================================
async def engine_loop(feed: MT5Feed):
    cd = Cooldowns()
    art_dir = Path(settings.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    # Boot message
    if settings.discord_enabled:
        try:
            await send_discord_embed(
                title="🤖 Engine ONLINE — Swing Visual + DeepSeek AI",
                description=(
                    f"PAIR `{settings.symbol}` • Entry `M15` • AI `DeepSeek V3`\n"
                    f"Smart Logic: Follow H1 Momentum enabled."
                ),
                color=0x95A5A6,
                footer="Signal Engine — Swing Desk",
            )
            print("✅ Discord message sent (boot)")
        except Exception as e:
            print(f"⚠️ Discord boot message failed: {e}")

    while True:
        print(f"\n[ENGINE] Tick | Mode={settings.mode} | {utc_now()}")

        tf_results: Dict[str, Dict[str, Any]] = {}
        frames: Dict[str, Any] = {}
        zones: Dict[str, Dict[str, Any]] = {}

        # 1) fetch & analyze
        for tf in TIMEFRAMES:
            df = feed.fetch_ohlcv(tf, n=900)
            df = add_indicators(df)
            frames[tf] = df
            struct = analyze_structure(df)
            atr = float(df["atr14"].iloc[-1]) if "atr14" in df.columns else 0.0
            vol_z = float(df["vol_z20"].iloc[-1]) if "vol_z20" in df.columns else 0.0

            tf_results[tf] = {
                "bias": struct.get("bias", "RANGING"),
                "event": struct.get("event", "NONE"),
                "atr": atr,
                "vol_z": vol_z,
            }

            if tf in ZONE_TFS:
                zones[tf] = detect_zones(df)
            else:
                zones[tf] = {"SUPPLY": None, "DEMAND": None}
            
            # Simple print
            print(f"  - {tf} | {tf_results[tf]['bias']} | ATR={atr:.1f}")

        # 2) Sanity & Swing Gate
        atr_allowed, atr_reason = no_trade_gate(
            {k: v for k, v in tf_results.items() if k in ["M15", "H1", "H4"]}, settings.min_atr
        )
        gate_ok, gate_reason = _swing_gate(tf_results)

        m15_df = frames["M15"]
        h1_df = frames["H1"]
        liq = detect_liquidity_sweep(m15_df, lookback=60, wick_ratio=0.55)
        fake = detect_fake_breakout(m15_df, lookback=60)
        
        # --- Logic: Jika Gate Fail, Kirim Watch Report (Include AI) ---
        if not atr_allowed or not gate_ok:
            reason = atr_reason if not atr_allowed else gate_reason
            print(f"🚫 NO TRADE — {reason}")
            
            # Cek cooldown watch dulu biar gak boros AI
            if cd.can_watch() and settings.discord_enabled:
                print("🧠 Calling DeepSeek for WATCH Analysis...")
                ai_narrative = analyze_market_with_megallm(
                    symbol=settings.symbol,
                    timeframe="M15",
                    bias=tf_results["M15"]["bias"], 
                    close=float(m15_df["close"].iloc[-1]),
                    ema50=float(m15_df["ema50"].iloc[-1]),
                    ema200=float(m15_df["ema200"].iloc[-1]),
                    rsi=float(m15_df["rsi14"].iloc[-1]),
                    zones=zones["M15"],
                    liquidity=liq,
                    context="WATCH"
                )
                await publish_watch_report(frames, tf_results, zones, reason, cd, art_dir, ai_narrative)
            
            await asyncio.sleep(settings.loop_seconds)
            continue

        # 5) Build Plan (Kalau Gate Lolos)
        # CRITICAL FIX: Jika H4 Ranging, kita paksa pakai bias H1 supaya entry_engine tidak reject.
        htf_bias_to_use = tf_results["H4"]["bias"]
        if htf_bias_to_use == "RANGING":
             htf_bias_to_use = tf_results["H1"]["bias"]

        ltf_bias = tf_results["M15"]["bias"]

        plan = build_trade_plan(
            symbol=settings.symbol,
            htf_bias=htf_bias_to_use, # Use H1 bias if H4 is Ranging
            ltf_bias=ltf_bias,
            m15_df=m15_df,
            h1_df=h1_df,
            zones_m15=zones["M15"],
            liquidity_m15=liq,
            fakeout_m15=fake,
            mode=settings.mode,
        )

        if plan is None:
            reason = "Gate OK but No Setup (Zone/Liq)"
            print(f"🚫 NO TRADE — {reason}")
            await asyncio.sleep(settings.loop_seconds)
            continue

        # 6) Publish Signal
        if not cd.can_signal():
            print(f"⏳ Signal cooldown active. Skip.")
            await asyncio.sleep(settings.loop_seconds)
            continue
            
        print("🧠 Calling DeepSeek for TRADE Analysis...")
        ai_narrative = analyze_market_with_megallm(
            symbol=settings.symbol,
            timeframe="M15",
            bias=plan.market_bias,
            close=float(m15_df["close"].iloc[-1]),
            ema50=float(m15_df["ema50"].iloc[-1]),
            ema200=float(m15_df["ema200"].iloc[-1]),
            rsi=float(m15_df["rsi14"].iloc[-1]),
            zones=zones["M15"],
            liquidity=liq,
            context="TRADE"
        )

        image_path = (art_dir / f"{settings.symbol}_M15_signal.png").as_posix()
        rp = RenderPlan(
            symbol=plan.symbol, tf="M15", side=plan.side,
            entry=float(sum(plan.entry_zone)/2), sl=float(plan.sl),
            tp1=float(plan.tp1), tp2=float(plan.tp2), tp3=float(plan.tp3),
            entry_zone=plan.entry_zone, market_bias=plan.market_bias,
            confidence=float(plan.confidence), reason=plan.reason
        )

        try:
            render_swing_chart(m15_df, zones["M15"], rp, image_path, settings.chart_last_n)
            if settings.discord_enabled:
                title = f"📌 {settings.symbol} — {plan.side.upper()} SETUP"
                desc = _format_trade_text(plan, ai_analysis=ai_narrative)
                footer = f"DeepSeek V3 | {utc_now()}"
                color = 0x2ECC71 if plan.side.upper() == "BUY" else 0xE74C3C
                
                await send_discord_embed_with_image(title, desc, image_path, color, footer)
                cd.mark_signal()
                print("✅ SIGNAL SENT!")
        except Exception as e:
            print(f"❌ Signal Failed: {e}")

        await asyncio.sleep(settings.loop_seconds)


async def publish_watch_report(frames, tf_results, zones, reason: str, cd: Cooldowns, art_dir: Path, ai_narrative: str):
    try:
        m15_df = frames["M15"]
        liq = detect_liquidity_sweep(m15_df, lookback=60, wick_ratio=0.55)
        fake = detect_fake_breakout(m15_df, lookback=60)

        image_path = (art_dir / f"{settings.symbol}_M15_watch.png").as_posix()

        # Render Chart (No Plan)
        render_swing_chart(
            df=m15_df,
            zones=zones["M15"],
            plan=None,
            out_path=image_path,
            last_n=settings.chart_last_n,
        )

        title = f"🛰️ MARKET WATCH — {settings.symbol}"
        desc = _format_watch_report(settings.symbol, tf_results, zones["M15"], liq, fake, reason, ai_analysis=ai_narrative)
        footer = f"{utc_now()} | Next Check: {settings.watch_status_seconds}s"

        await send_discord_embed_with_image(
            title=title,
            description=desc,
            image_path=image_path,
            color=0x34495E, # Dark Blue for Watch
            footer=footer,
        )
        print("✅ Watch Report Sent (with AI)")
        cd.mark_watch()
    except Exception as e:
        print(f"⚠️ Watch Report Failed: {e}")


def main():
    print("===================================")
    print(" XAUUSD SIGNAL ENGINE - REBOOT ")
    print(" Mode :", settings.mode)
    print(f" AI Brain:", f"{settings.megallm_model}" if settings.use_ai_narrative else "OFF")
    print("===================================")

    feed = MT5Feed(symbol=settings.symbol)
    status = feed.connect()

    if not status.ok:
        print(f"🔴 MT5 ERROR: {status.reason}")
        return

    print("🟢 MT5 READY")
    try:
        asyncio.run(engine_loop(feed))
    finally:
        feed.shutdown()


if __name__ == "__main__":
    main()