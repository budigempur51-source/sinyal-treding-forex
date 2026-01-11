from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

Timeframe = Literal["M1", "M5", "M15", "H1", "H4"]

@dataclass(frozen=True)
class MT5AccountInfo:
    login: int
    server: str
    name: str = ""
    currency: str = ""

@dataclass(frozen=True)
class MT5SymbolInfo:
    symbol: str
    description: str = ""
    digits: int = 0
    point: float = 0.0
    trade_mode: int = 0

@dataclass(frozen=True)
class FeedSnapshot:
    symbol: str
    timeframe: Timeframe
    last_close: float
    last_time_utc: str
    bars: int

@dataclass(frozen=True)
class FeedStatus:
    ok: bool
    reason: str
    account: Optional[MT5AccountInfo] = None
    symbol: Optional[MT5SymbolInfo] = None
