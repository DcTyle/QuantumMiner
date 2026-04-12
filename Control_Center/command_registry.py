# ============================================================================
# Quantum Application / Control Center
# ASCII-ONLY SOURCE FILE
# File: Control_Center/command_registry.py
# Version: v1.0 Command Registry
# ----------------------------------------------------------------------------
# Purpose
# -------
# Central registry that maps human-friendly commands to actions. Actions can:
#  - Publish Control Center UI events
#  - Invoke internal Python entry points
#  - Execute safe scripts in a subprocess
#  - Query VSD for status and publish results to panels
#
# Integration
# -----------
# - ControlCenterApp wires voice/text input to this registry as a fallback
#   (Neuralis CommandRouter remains primary if available).
# - Results are published to bus topics so panels can render messages.
# ============================================================================
from __future__ import annotations
from typing import Callable, Dict, Any, List, Tuple
from enum import Enum
import subprocess
import time

Bus = Any  # duck-typed EventBus
VSD = Any  # duck-typed VSD


class Command(Enum):
    SHOW_HASH_RATES = 1
    GPU_STATUS = 2
    RUN_VERIFY = 3
    EXPORT_STATS = 4
    PAUSE_MINER = 5
    RUN_TEST_SIMULATION = 6
    SET_SYMBOL = 7


class CommandRegistry:
    def __init__(self, bus: Bus, vsd: VSD) -> None:
        self.bus = bus
        self.vsd = vsd
        self._handlers: Dict[Command, Callable[[Dict[str, Any]], Any]] = {}
        self._init_defaults()

    # ------------------------------------------------------------------
    # Public API (enum-based only)
    # ------------------------------------------------------------------
    def run_enum(self, cmd: Command, args: Dict[str, Any] | None = None) -> Any:
        handler = self._handlers.get(cmd)
        if handler is None:
            self._emit("Unknown command: %s" % cmd.name, level="warn")
            return None
        try:
            return handler(dict(args or {}))
        except Exception as exc:
            self._emit("Command error: %s" % str(exc), level="error")
            return None

    # ------------------------------------------------------------------
    # Default command set
    # ------------------------------------------------------------------
    def _init_defaults(self) -> None:
        self._handlers = {
            Command.SHOW_HASH_RATES: lambda _a: self._emit(self._format_rates(), level="ok", panel="miner"),
            Command.GPU_STATUS: self._cmd_gpu_status,
            Command.RUN_VERIFY: self._cmd_verify,
            Command.EXPORT_STATS: self._cmd_export_stats,
            Command.PAUSE_MINER: self._cmd_pause_miner,
            Command.RUN_TEST_SIMULATION: self._cmd_run_test_sim,
            Command.SET_SYMBOL: self._cmd_set_symbol,
        }

    # ------------------------------------------------------------------
    # Helpers / Actions
    # ------------------------------------------------------------------
    def _format_rates(self) -> str:
        try:
            idx = self.vsd.get("telemetry/metrics/index", []) or []
            nets = [str(x).upper() for x in idx] if isinstance(idx, list) else []
            parts: List[str] = []
            for n in nets:
                cur = self.vsd.get(f"telemetry/metrics/{n}/current", {}) or {}
                hs = float(cur.get("hashes_submitted_hs", 0.0))
                parts.append("%s: %.2f h/s" % (n, hs))
            return " | ".join(parts) if parts else "No miner metrics yet"
        except Exception:
            return "No miner metrics yet"

    def _publish_ui_action(self, payload: Dict[str, Any]) -> None:
        try:
            if self.bus is not None and hasattr(self.bus, "publish"):
                self.bus.publish("control_center.ui.action", payload)
        except Exception:
            pass

    def _cmd_gpu_status(self, _a: Dict[str, Any]) -> None:
        rc, out, err = self._run_cmd(["python3", "scripts/gpu_initiator_status.py"]) 
        msg = out.strip() if rc == 0 and out.strip() else (err.strip() or ("rc=" + str(rc)))
        self._emit("GPU: " + msg, level=("ok" if rc == 0 else "warn"))

    def _cmd_verify(self, _a: Dict[str, Any]) -> None:
        rc, out, err = self._run_cmd(["bash", "-lc", "STRICT_BOUNDARIES=1 ./scripts/verify.sh"], timeout=120.0)
        lvl = "ok" if rc == 0 else "error"
        self._emit("Verify: rc=%d" % rc, level=lvl)
        if out:
            self._emit(out[-400:], level=lvl)
        if err and rc != 0:
            self._emit(err[-400:], level="error")

    def _cmd_export_stats(self, _a: Dict[str, Any]) -> None:
        try:
            snap = {
                "ts": time.time(),
                "note": "export requested",
            }
            self.vsd.store("telemetry/exports/last", snap)
            self._emit("Stats export flagged in VSD", level="ok")
        except Exception:
            self._emit("Export failed", level="error")

    def _cmd_pause_miner(self, _a: Dict[str, Any]) -> None:
        note = str((_a or {}).get("note", "Miner pause requested"))
        try:
            if self.bus is not None and hasattr(self.bus, "publish"):
                self.bus.publish("control_center.cmd.pause_miner", {
                    "ts": time.time(),
                    "note": note,
                    "source": "control_center",
                })
        except Exception:
            pass
        self._emit(note, level="warn", panel="miner")

    def _cmd_run_test_sim(self, _a: Dict[str, Any]) -> None:
        # Publish a standard event; prediction engine or simulator can react.
        try:
            if self.bus is not None and hasattr(self.bus, "publish"):
                self.bus.publish("control_center.cmd.run_test_simulation", {"ts": time.time()})
            self._emit("Test simulation requested", level="ok", panel="prediction")
        except Exception:
            self._emit("Test simulation dispatch failed", level="error")

    def _cmd_set_symbol(self, args: Dict[str, Any]) -> None:
        try:
            sym = str(args.get("symbol", "BTCUSDT")).upper()
            self.vsd.store("control_center/timeseries/selected_symbol", sym)
            self._emit("Symbol set: %s" % sym, level="ok")
        except Exception:
            self._emit("Could not set symbol", level="error")

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _emit(self, text: str, level: str = "info", panel: str = "neuralis") -> None:
        try:
            if self.bus is not None and hasattr(self.bus, "publish"):
                self.bus.publish("control_center.telemetry.%s" % panel, {
                    "ts": time.time(),
                    "text": text,
                    "level": level,
                })
        except Exception:
            pass

    def _run_cmd(self, argv: List[str], timeout: float = 10.0) -> Tuple[int, str, str]:
        try:
            p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            out = (p.stdout or b"").decode("ascii", errors="ignore")
            err = (p.stderr or b"").decode("ascii", errors="ignore")
            return p.returncode, out, err
        except Exception as exc:
            return 1, "", str(exc)
