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
from app.signal.entry_engine import build_trade_plan

from app.notify.discord_bot import send_discord_embed, send_discord_embed_with_image
from app.visual.chart_renderer import render_swing_chart, RenderPlan


# =========================================================
# SWING CONFIG (M15 entry, HTF = H1 + H4)
# =========================================================
TIMEFRAMES = ["M1", "M5", "M15", "H1", "H4"]
ZONE_TFS = ["M15", "H1", "H4"]  # zones are meaningful here


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
    Swing rules:
    - optional: disallow ranging on H4
    - optional: disallow ranging on H1
    - optional: require alignment H1 == H4 (and both not ranging)
    """
    h4_bias = tf_results["H4"]["bias"]
    h1_bias = tf_results["H1"]["bias"]

    if settings.swing_disallow_h4_ranging and h4_bias == "RANGING":
        return False, "HTF ranging (H4)"

    if settings.swing_disallow_h1_ranging and h1_bias == "RANGING":
        return False, "HTF ranging (H1)"

    if settings.swing_require_htf_alignment:
        # if either is ranging, still reject (safer)
        if h4_bias == "RANGING" or h1_bias == "RANGING":
            return False, f"HTF ranging (H4={h4_bias}, H1={h1_bias})"
        if h1_bias != h4_bias:
            return False, f"HTF conflict (H1={h1_bias} vs H4={h4_bias})"

    return True, "OK"


def _format_trade_text(plan) -> str:
    side = plan.side.upper()
    entry_lo, entry_hi = plan.entry_zone

    lines = []
    lines.append(f"**PAIR**: `{plan.symbol}`")
    lines.append(f"**MODE**: `SWING` | **ENTRY TF**: `M15` | **ENGINE**: `{settings.mode}`")
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
    lines.append(f"**Reason**: {plan.reason}")
    return "\n".join(lines)


def _format_watch_report(symbol: str, tf_results: dict, zones_m15: dict, liq: dict, fake: dict, reason: str) -> str:
    lines = []
    lines.append(f"**PAIR**: `{symbol}`")
    lines.append(f"**MODE**: `SWING WATCH` | **ENTRY TF**: `M15` | **ENGINE**: `{settings.mode}`")
    lines.append("")
    lines.append("**HTF STATUS**")
    lines.append(f"- H4: `{tf_results['H4']['bias']}` `{tf_results['H4']['event']}` | ATR `{tf_results['H4']['atr']:.2f}` | VOLz `{tf_results['H4']['vol_z']:.2f}`")
    lines.append(f"- H1: `{tf_results['H1']['bias']}` `{tf_results['H1']['event']}` | ATR `{tf_results['H1']['atr']:.2f}` | VOLz `{tf_results['H1']['vol_z']:.2f}`")
    lines.append("")
    lines.append("**ENTRY TF (M15)**")
    lines.append(f"- M15: `{tf_results['M15']['bias']}` `{tf_results['M15']['event']}` | ATR `{tf_results['M15']['atr']:.2f}` | VOLz `{tf_results['M15']['vol_z']:.2f}`")
    lines.append(f"- Zones: `{_fmt_zones(zones_m15)}`")
    lines.append("")
    lines.append("**LIQUIDITY CHECK (M15)**")
    lines.append(f"- Sweep: `{liq.get('sweep')}` | Level: `{liq.get('level')}` | Notes: `{liq.get('notes')}`")
    lines.append(f"- Fakeout: `{fake.get('fake')}` | Level: `{fake.get('level')}`")
    lines.append("")
    lines.append(f"🚫 **NO TRADE** — {reason}")
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

    # Create artifacts dir
    art_dir = Path(settings.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    # Boot message
    if settings.discord_enabled:
        try:
            await send_discord_embed(
                title="🤖 Engine ONLINE — Swing Visual + Watch",
                description=(
                    f"PAIR `{settings.symbol}` • Entry `M15` • "
                    f"Sinyal valid akan dikirim + chart. Kalau NO TRADE, akan kirim WATCH report + chart tiap interval."
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

        # 1) fetch & analyze per TF
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

            print(
                f"  - {settings.symbol} {tf} | {tf_results[tf]['bias']} {tf_results[tf]['event']} "
                f"| ATR={atr:.2f} VOLz={vol_z:.2f} | {_fmt_zones(zones[tf])}"
            )

        # 2) sanity gate: ATR floors for swing relevant TFs
        atr_allowed, atr_reason = no_trade_gate(
            {k: v for k, v in tf_results.items() if k in ["M15", "H1", "H4"]},
            settings.min_atr,
        )
        if not atr_allowed:
            reason = atr_reason
            print(f"🚫 NO TRADE — {reason}")
            await publish_watch_if_needed(frames, tf_results, zones, reason, cd, art_dir)
            await asyncio.sleep(settings.loop_seconds)
            continue

        # 3) swing HTF gate
        ok, reason_swing = _swing_gate(tf_results)
        if not ok:
            reason = reason_swing
            print(f"🚫 NO TRADE — {reason}")
            await publish_watch_if_needed(frames, tf_results, zones, reason, cd, art_dir)
            await asyncio.sleep(settings.loop_seconds)
            continue

        # 4) liquidity checks (timing)
        m15_df = frames["M15"]
        h1_df = frames["H1"]

        liq = detect_liquidity_sweep(m15_df, lookback=60, wick_ratio=0.55)
        fake = detect_fake_breakout(m15_df, lookback=60)

        print(f"🧪 LIQUIDITY M15: sweep={liq.get('sweep')} level={liq.get('level')} notes={liq.get('notes')}")
        print(f"🧪 FAKEOUT   M15: fake={fake.get('fake')} level={fake.get('level')}")

        # 5) build plan
        htf_bias = tf_results["H4"]["bias"]
        ltf_bias = tf_results["M15"]["bias"]

        plan = build_trade_plan(
            symbol=settings.symbol,
            htf_bias=htf_bias,
            ltf_bias=ltf_bias,
            m15_df=m15_df,
            h1_df=h1_df,
            zones_m15=zones["M15"],
            liquidity_m15=liq,
            fakeout_m15=fake,
            mode=settings.mode,
        )

        if plan is None:
            reason = "No valid setup after liquidity + zone filters"
            print(f"🚫 NO TRADE — {reason}")
            await publish_watch_if_needed(frames, tf_results, zones, reason, cd, art_dir)
            await asyncio.sleep(settings.loop_seconds)
            continue

        # 6) publish signal with cooldown
        if not cd.can_signal():
            print(f"⏳ Signal cooldown active ({settings.signal_cooldown_seconds}s). Skip publish.")
            await asyncio.sleep(settings.loop_seconds)
            continue

        image_path = (art_dir / f"{settings.symbol}_M15_swing_signal.png").as_posix()

        rp = RenderPlan(
            symbol=plan.symbol,
            tf="M15",
            side=plan.side,
            entry=float(sum(plan.entry_zone) / 2.0),
            sl=float(plan.sl),
            tp1=float(plan.tp1),
            tp2=float(plan.tp2),
            tp3=float(plan.tp3),
            entry_zone=plan.entry_zone,
            market_bias=plan.market_bias,
            confidence=float(plan.confidence),
            reason=plan.reason,
        )

        try:
            render_swing_chart(
                df=m15_df,
                zones=zones["M15"],
                plan=rp,
                out_path=image_path,
                last_n=settings.chart_last_n,
            )
        except Exception as e:
            print(f"❌ Chart render failed: {e}")
            await asyncio.sleep(settings.loop_seconds)
            continue

        if settings.discord_enabled:
            title = f"📌 SWING SIGNAL (M15) — {plan.symbol} — {plan.side.upper()}"
            desc = _format_trade_text(plan)
            footer = f"HTF: H4={tf_results['H4']['bias']} | H1={tf_results['H1']['bias']} | {utc_now()}"
            color = 0x2ECC71 if plan.side.upper() == "BUY" else 0xE74C3C

            try:
                await send_discord_embed_with_image(
                    title=title,
                    description=desc,
                    image_path=image_path,
                    color=color,
                    footer=footer,
                )
                print("✅ Discord signal sent (embed + chart)")
                cd.mark_signal()
            except Exception as e:
                print(f"❌ Discord send failed: {e}")

        await asyncio.sleep(settings.loop_seconds)


async def publish_watch_if_needed(frames, tf_results, zones, reason: str, cd: Cooldowns, art_dir: Path):
    """
    Publish WATCH report (NO TRADE) with chart + zones + EMA
    Controlled by watch_status_seconds cooldown.
    """
    if not settings.discord_enabled:
        return
    if not cd.can_watch():
        return

    try:
        m15_df = frames["M15"]
        liq = detect_liquidity_sweep(m15_df, lookback=60, wick_ratio=0.55)
        fake = detect_fake_breakout(m15_df, lookback=60)

        image_path = (art_dir / f"{settings.symbol}_M15_watch.png").as_posix()

        # Chart without plan overlays (still shows EMA + zones)
        render_swing_chart(
            df=m15_df,
            zones=zones["M15"],
            plan=None,
            out_path=image_path,
            last_n=settings.chart_last_n,
        )

        title = f"🛰️ SWING WATCH (M15) — {settings.symbol}"
        desc = _format_watch_report(settings.symbol, tf_results, zones["M15"], liq, fake, reason)
        footer = f"{utc_now()} | Watch every {settings.watch_status_seconds}s"

        await send_discord_embed_with_image(
            title=title,
            description=desc,
            image_path=image_path,
            color=0x95A5A6,
            footer=footer,
        )
        print("✅ Discord watch report sent (embed + chart)")
        cd.mark_watch()
    except Exception as e:
        print(f"⚠️ Watch report failed: {e}")


def main():
    print("===================================")
    print(" XAUUSD SIGNAL ENGINE - BOOTING UP ")
    print(" Mode :", settings.mode)
    print(" Symbol:", settings.symbol)
    print(" TF Entry:", "M15 (SWING)")
    print("===================================")

    feed = MT5Feed(symbol=settings.symbol)
    status = feed.connect()

    if not status.ok:
        print(f"🔴 MT5 CONNECT FAILED: {status.reason}")
        return

    print("🟢 MT5 CONNECTED")
    print(f"  Account: {status.account.login} | Server: {status.account.server} | Currency: {status.account.currency}")
    print(f"  Symbol : {status.symbol.symbol} | Digits: {status.symbol.digits} | Point: {status.symbol.point}")

    try:
        asyncio.run(engine_loop(feed))
    finally:
        feed.shutdown()


if __name__ == "__main__":
    main()
