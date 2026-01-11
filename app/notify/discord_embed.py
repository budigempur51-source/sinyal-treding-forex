from datetime import datetime, timezone


def build_trade_embed(plan, symbol: str, timeframe: str, mode: str) -> dict:
    side = str(plan.side).upper()
    color = 0x2ECC71 if side == "BUY" else 0xE74C3C

    el, eh = plan.entry_zone

    return {
        "title": f"{symbol} • {timeframe} • {side}",
        "description": f"**Market Bias:** {plan.market_bias}\n**Mode:** {mode}",
        "color": color,
        "fields": [
            {"name": "Entry Zone", "value": f"{el:.2f} - {eh:.2f}", "inline": False},
            {"name": "Stop Loss", "value": f"{plan.sl:.2f}", "inline": True},
            {"name": "TP1", "value": f"{plan.tp1:.2f}", "inline": True},
            {"name": "TP2", "value": f"{plan.tp2:.2f}", "inline": True},
            {"name": "TP3", "value": f"{plan.tp3:.2f}", "inline": True},
            {"name": "RR (TP2)", "value": f"{plan.rr:.2f}", "inline": True},
            {"name": "Confidence", "value": f"{plan.confidence:.1f}%", "inline": True},
            {"name": "Reason", "value": str(plan.reason)[:900], "inline": False},
        ],
        "footer": {"text": "Auto Signal Engine • Confirm > Predict"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
