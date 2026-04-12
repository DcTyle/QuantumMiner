# ============================================================================
# Quantum Application / miner
# ASCII-ONLY SOURCE FILE
# File: miner/__init__.py
# Version: v4.8.9.2 Hybrid (Package Init + EventBus-safe)
# ============================================================================
"""
Miner package initializer.

Purpose
-------
- Provide a clean import surface for miner components.
- Avoid circular imports with BIOS and core.
- Offer optional EventBus access for higher-level wiring.
- ASCII-only, ADA v4.7 Hybrid compatible.

Notes
-----
This module deliberately avoids importing bios.main_runtime or core.__init__.
It only pulls in miner-local modules and a few lightweight helpers.
"""

from __future__ import annotations

from typing import Any, Dict
import logging
import sys

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def _init_logger() -> logging.Logger:
    log = logging.getLogger("miner.init")
    if not log.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log


_logger = _init_logger()

# ---------------------------------------------------------------------------
# Core utilities (minimal dependency surface)
# ---------------------------------------------------------------------------
try:
    from core.utils import hw_snapshot, append_telemetry  # type: ignore
except Exception as exc:  # pragma: no cover
    _logger.warning("core.utils unavailable for miner.init: %s", exc)

    def hw_snapshot() -> Dict[str, Any]:  # type: ignore
        return {}

    def append_telemetry(_scope: str, _payload: Dict[str, Any]) -> None:  # type: ignore
        return

# ---------------------------------------------------------------------------
# BIOS EventBus access (no BIOSUtilities, only get_event_bus)
# ---------------------------------------------------------------------------
try:
    from bios.event_bus import get_event_bus  # type: ignore
except Exception as exc:  # pragma: no cover

    _logger.warning("bios.event_bus unavailable for miner.init: %s", exc)

    def get_event_bus() -> Any:  # type: ignore
        class _NoBus:
            def publish(self, _topic: str, _data: Dict[str, Any] | None = None) -> None:
                return

            def subscribe(self, *_a: Any, **_kw: Any) -> None:
                return

        return _NoBus()

# ---------------------------------------------------------------------------
# Miner-local imports
# ---------------------------------------------------------------------------
try:
    from miner.miner_engine import MinerEngine, register_miner_autostart  # type: ignore
except Exception as exc:  # pragma: no cover
    _logger.error("miner_engine import failed in miner.init: %s", exc)
    MinerEngine = None  # type: ignore
    register_miner_autostart = None  # type: ignore

try:
    from miner.failsafe import FailsafeGovernor  # type: ignore
except Exception as exc:  # pragma: no cover
    _logger.error("miner.failsafe import failed in miner.init: %s", exc)
    FailsafeGovernor = None  # type: ignore

try:
    from prediction_engine.crypto_com_api import CryptoComAPI  # type: ignore
except Exception as exc:  # pragma: no cover
    _logger.error("miner.crypto_com_api import failed in miner.init: %s", exc)
    CryptoComAPI = None  # type: ignore

# If you have additional miner modules (e.g., telemetry console, market data),
# they can be added here following the same guarded-import pattern.

# ---------------------------------------------------------------------------
# Version and public exports
# ---------------------------------------------------------------------------
version: str = "v4.8.9.2"

__all__ = [
    "version",
    # core helpers
    "hw_snapshot",
    "append_telemetry",
    "get_event_bus",
    # miner core
    "MinerEngine",
    "register_miner_autostart",
    "FailsafeGovernor",
    "CryptoComAPI",
]
