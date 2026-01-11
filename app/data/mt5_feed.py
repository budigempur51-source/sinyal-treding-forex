from __future__ import annotations

import MetaTrader5 as mt5
import pandas as pd
from datetime import timezone
from typing import Dict

from app.config import settings
from app.data.models import FeedSnapshot, FeedStatus, MT5AccountInfo, MT5SymbolInfo, Timeframe

TF_MAP: Dict[Timeframe, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}

class MT5Feed:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def connect(self) -> FeedStatus:
        """
        Connect to MT5 terminal using explicit terminal path to avoid IPC timeout.
        MT5 MUST be installed and opened (same Windows user, same admin level).
        """
        # Initialize with explicit path (CRITICAL FIX)
        if settings.mt5_path:
            ok = mt5.initialize(path=settings.mt5_path)
        else:
            ok = mt5.initialize()

        if not ok:
            return FeedStatus(ok=False, reason=f"MT5 initialize failed: {mt5.last_error()}")

        # Login (safe even if already logged in)
        if settings.mt5_login and settings.mt5_password and settings.mt5_server:
            logged = mt5.login(
                login=settings.mt5_login,
                password=settings.mt5_password,
                server=settings.mt5_server
            )
            if not logged:
                return FeedStatus(ok=False, reason=f"MT5 login failed: {mt5.last_error()}")

        acc = mt5.account_info()
        if acc is None:
            return FeedStatus(ok=False, reason="account_info() returned None (MT5 not logged in)")

        account = MT5AccountInfo(
            login=int(acc.login),
            server=str(acc.server),
            name=str(getattr(acc, "name", "")),
            currency=str(getattr(acc, "currency", "")),
        )

        # Ensure symbol exists & enabled
        if not mt5.symbol_select(self.symbol, True):
            return FeedStatus(
                ok=False,
                reason=f"symbol_select failed for {self.symbol}: {mt5.last_error()}",
                account=account
            )

        info = mt5.symbol_info(self.symbol)
        if info is None:
            return FeedStatus(
                ok=False,
                reason=f"symbol_info() is None for {self.symbol}",
                account=account
            )

        symbol_info = MT5SymbolInfo(
            symbol=self.symbol,
            description=str(getattr(info, "description", "")),
            digits=int(getattr(info, "digits", 0)),
            point=float(getattr(info, "point", 0.0)),
            trade_mode=int(getattr(info, "trade_mode", 0)),
        )

        return FeedStatus(ok=True, reason="OK", account=account, symbol=symbol_info)

    def shutdown(self) -> None:
        mt5.shutdown()

    def fetch_ohlcv(self, tf: Timeframe, n: int = 500) -> pd.DataFrame:
        timeframe = TF_MAP[tf]
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, n)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates for {self.symbol} {tf}. last_error={mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(timezone.utc)

        if "tick_volume" in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
        if "volume" not in df.columns:
            df["volume"] = 0

        return df[["time", "open", "high", "low", "close", "volume"]]

    def snapshot(self, tf: Timeframe, n: int = 300) -> FeedSnapshot:
        df = self.fetch_ohlcv(tf, n)
        last = df.iloc[-1]
        return FeedSnapshot(
            symbol=self.symbol,
            timeframe=tf,
            last_close=float(last["close"]),
            last_time_utc=last["time"].strftime("%Y-%m-%d %H:%M:%S UTC"),
            bars=int(len(df)),
        )
