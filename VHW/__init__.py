# ============================================================================
# Quantum Application / VHW
# ASCII-ONLY SOURCE FILE
# File: VHW/__init__.py
# Version: v4.8.9 Init (No core.utils dependency)
# ============================================================================
"""
Package init for Virtual Hardware (VHW).

Responsibilities
----------------
- Provide a stable logger for VHW init.
- Expose the unified VSD/EventBus kernel from VHW.vsd_manager.
- Expose VHWCore and global_core from VHW.vhw_core when available.
- Avoid ANY import from core.utils to prevent circular imports.

Exports
-------
get_event_bus() -> EventBus
VSD -> global VSDManager instance
VSDManager -> class
VHWCore -> main core orchestrator (if available)
global_core() -> convenience singleton accessor
"""

from __future__ import annotations
from typing import Any, Dict
import logging
import sys

# ----------------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------------
_logger = logging.getLogger("VHW.init")
if not _logger.handlers:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] VHW.init %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    ))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)


# ----------------------------------------------------------------------------
# Import VSD/EventBus kernel (no core.utils)
# ----------------------------------------------------------------------------
try:
    from VHW.vsd_manager import get_event_bus, VSD, VSDManager  # type: ignore
except Exception as exc:
    _logger.warning("VHW.vsd_manager unavailable, using local stubs: %s", exc)

    def get_event_bus() -> Any:  # type: ignore
        class _Bus:
            def publish(self, *args, **kwargs) -> None:
                return None
            def subscribe(self, *args, **kwargs) -> None:
                return None
        return _Bus()

    class VSDManager:  # type: ignore
        def __init__(self) -> None:
            self._m: Dict[str, Any] = {}
        def get(self, key: str, default: Any = None) -> Any:
            return self._m.get(str(key), default)
        def store(self, key: str, value: Any) -> None:
            self._m[str(key)] = value
        def delete(self, key: str) -> None:
            self._m.pop(str(key), None)
        def get(self, key: str, default: Any = None) -> Any:
            return self.get(key, default)
        def set(self, key: str, value: Any) -> None:
            self.store(key, value)

    VSD = VSDManager()  # type: ignore


# ----------------------------------------------------------------------------
# Import VHW core orchestrator (optional)
# ----------------------------------------------------------------------------
try:
    from VHW.vhw_core import VHWCore, global_core  # type: ignore
except Exception as exc:
    _logger.warning("VHWCore unavailable, using stub: %s", exc)

    class VHWCore:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return
        def start(self) -> None:
            return
        def stop(self, timeout: float = 2.0) -> None:
            return
        def status(self) -> Dict[str, Any]:
            return {"ready": False, "error": "VHWCore stub"}

    def global_core() -> VHWCore:  # type: ignore
        return VHWCore()


__all__ = [
    "get_event_bus",
    "VSD",
    "VSDManager",
    "VHWCore",
    "global_core",
]
