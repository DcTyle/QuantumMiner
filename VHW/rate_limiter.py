# ============================================================================
# Quantum Application / VHW
# ASCII-ONLY SOURCE FILE
# File: rate_limiter.py
# Version: v1.0 (Deterministic, EventBus-Aware, No-Stubs)
# ============================================================================
"""
Purpose
-------
Deterministic rate limiter for API and subsystem throttling.

Design
------
- No fallbacks, no stubs.
- ASCII-only.
- Uses EventBus for telemetry.
- Enforces a fixed maximum number of allowed calls inside a sliding window.
- Used by:
      AI_processor
      prediction_engine
      miner.api_client
      any module requiring throttling

EventBus Topics
---------------
rate_limiter.hit
rate_limiter.block
"""

from __future__ import annotations
from typing import List, Dict
import time

from bios.event_bus import get_event_bus

class RateLimiter:
    def __init__(self, max_calls: int = 100, window_seconds: float = 60.0) -> None:
        self.max_calls = int(max_calls)
        self.window = float(window_seconds)
        self.events: List[float] = []
        self.bus = get_event_bus()

    def allow(self) -> bool:
        now = time.time()
        cutoff = now - self.window

        # purge old timestamps
        self.events = [t for t in self.events if t >= cutoff]

        if len(self.events) < self.max_calls:
            self.events.append(now)
            self.bus.publish(
                "rate_limiter.hit",
                {"ts": now, "window": self.window, "count": len(self.events)}
            )
            return True

        self.bus.publish(
            "rate_limiter.block",
            {"ts": now, "window": self.window, "count": len(self.events)}
        )
        return False

    def usage_fraction(self) -> float:
        """Return fraction of window capacity consumed."""
        now = time.time()
        cutoff = now - self.window
        self.events = [t for t in self.events if t >= cutoff]
        return len(self.events) / float(self.max_calls)
