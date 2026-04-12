# ============================================================================
# VirtualMiner / VHW
# ASCII-ONLY SOURCE FILE
# File: market_utils.py
# Version: v4.8.7+ContextualWindow
# ============================================================================
"""
Purpose
-------
Market utilities for telemetry-driven trading and prediction support.
- Provides stateful market snapshots.
- Integrates contextual rolling-window and EMA smoothing.
- Thread-safe, BIOS-gated, EventBus and VSDManager compatible.

Features
--------
- ContextualWindow for recent price/volume changes.
- EMA smoothing for key metrics (price, volume, spread).
- Optional alert triggers based on config thresholds.
- Fully ASCII logging.
- Safe fallbacks for missing dependencies.
"""

from __future__ import annotations
from typing import Dict, Any, List, Callable, Optional
import threading
import time
import math
import hashlib

# ---------------------------------------------------------------------------
# Fallback imports
# ---------------------------------------------------------------------------
try:
    from VHW.vsd_manager import VSDManager
except Exception:
    class VSDManager:
        def get(self, key: str, default: Any = None) -> Any: return default
        def store(self, key: str, value: Any) -> None: pass

try:
    from core.utils import get, store  # noqa: F401
except Exception:
    def get(*a, **k): return {}
    def store(*a, **k): return None

try:
    from core.rule_params import RuleParams
except Exception:
    class RuleParams:
        def __init__(self) -> None:
            self._ema_alpha = 0.2
            self._headroom = 0.12
            self._alert_bias = 0.0
        @property
        def ema_alpha(self) -> float: return self._ema_alpha
        @property
        def headroom(self) -> float: return self._headroom
        @property
        def alert_bias(self) -> float: return self._alert_bias

try:
    from config.manager import ConfigManager
except Exception:
    class ConfigManager:
        def __init__(self) -> None: self._cfg = {}
        def get(self, key: str, default: Any = None) -> Any:
            return self._cfg.get(key, default)

try:
    from core.event_bus import get_event_bus
except Exception:
    def get_event_bus():
        class _NoBus:
            def publish(self, topic: str, payload: Dict[str, Any]) -> None: return
        return _NoBus()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ASCII_MIN = 32
ASCII_MAX = 126

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _now() -> float:
    return float(time.time())

def _sha256_ascii(s: str) -> str:
    try:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# EMA Filter
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self._value: Optional[float] = None
        self._lock = threading.RLock()

    def update(self, x: float) -> float:
        with self._lock:
            xv = float(x)
            if self._value is None:
                self._value = xv
            else:
                self._value = self.alpha * xv + (1.0 - self.alpha) * self._value
            return self._value

    def current(self) -> float:
        with self._lock:
            return self._value if self._value is not None else 0.0

# ---------------------------------------------------------------------------
# Contextual Rolling Window
# ---------------------------------------------------------------------------
class ContextualWindow:
    def __init__(self, max_len: int = 1000):
        self._lock = threading.RLock()
        self._buffer: List[Any] = []
        self.max_len = int(max_len)

    def append(self, item: Any, relevant: bool = False) -> None:
        with self._lock:
            if relevant:
                self._buffer.append(item)
                if len(self._buffer) > self.max_len:
                    self._buffer = self._buffer[-self.max_len:]

    def snapshot(self) -> List[Any]:
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

# ---------------------------------------------------------------------------
# Market Utils
# ---------------------------------------------------------------------------
class MarketUtils:
    def __init__(self, vsd: Optional[VSDManager] = None):
        self.vsd = vsd or VSDManager()
        self.rule = RuleParams()
        self.cfg = ConfigManager()
        self.bus = get_event_bus()
        self._lock = threading.RLock()

        alpha = float(getattr(self.rule, "ema_alpha", 0.2))
        self.ema_price = EMA(alpha)
        self.ema_volume = EMA(alpha)
        self.ema_spread = EMA(alpha)

        self.context = ContextualWindow(max_len=1000)
        self.alert_bias = float(self.cfg.get("market.alert_bias", getattr(self.rule, "alert_bias", 0.0)))

    def add_tick(self, symbol: str, price: float, volume: float, spread: float, relevant: bool = False):
        with self._lock:
            # Update EMAs
            price_smooth = self.ema_price.update(price)
            vol_smooth = self.ema_volume.update(volume)
            spread_smooth = self.ema_spread.update(spread)

            # Store contextual event
            event = {
                "symbol": symbol,
                "price": price,
                "price_ema": price_smooth,
                "volume": volume,
                "volume_ema": vol_smooth,
                "spread": spread,
                "spread_ema": spread_smooth,
                "ts": _now()
            }
            self.context.append(event, relevant=relevant)

            # Optionally store in VSD
            key = f"market/ticks/{symbol}"
            self.vsd.append_bounded(key, event, max_len=500)

            # Alert if spread exceeds threshold
            if spread_smooth > self.alert_bias:
                self.bus.publish("market.alert", {"symbol": symbol, "spread": spread_smooth, "ts": _now()})

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ema_price": self.ema_price.current(),
                "ema_volume": self.ema_volume.current(),
                "ema_spread": self.ema_spread.current(),
                "context": self.context.snapshot()
            }

    def clear_context(self) -> None:
        with self._lock:
            self.context.clear()
