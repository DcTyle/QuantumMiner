# ============================================================================
# Quantum Application / Neuralis_AI
# ASCII-ONLY SOURCE FILE
# File: ai_core.py
# Version: v4.8.5 (Phases 1,3,4 applied)
# Jarvis ADA v4.8.5 Hybrid Ready
# ============================================================================
"""
Purpose
-------
Neuralis AI core loader and installer utilities. Handles conditional
rehydration of the AI "brain" and the prediction engine memory based on the
BIOS boot signal.

Changes in v4.8.5
-----------------
Phase 1 (Safety):
    - All public operations refuse to run unless VSD key
      system/bios_boot_ok == True.
Phase 3 (Quality):
    - Structured logging (UTC) + typed returns.
Phase 4 (Architecture):
    - EventBus publish after successful rehydration:
        'ai.rehydrated' and 'pe.rehydrated' with payloads.
    - Interface-style decoupling via safe try-imports.

Public API (stable)
-------------------
load_on_activation(context: dict) -> dict
load_ai_fullstate(path: str, *, kind: str = "neuralis") -> dict
install_from_statevector(text: str, *, kind: str = "neuralis") -> dict
status() -> dict
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import logging
import time
import re

# ----------------------------------------------------------------------------
# Structured logging (Phase 3)
# ----------------------------------------------------------------------------
def _init_logger() -> logging.Logger:
    logger = logging.getLogger("Neuralis.AICore")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="[%(asctime)s] NEURALIS_AICORE %(levelname)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = _init_logger()

# ----------------------------------------------------------------------------
# Safe imports / fallbacks (Phase 1,4)
# ----------------------------------------------------------------------------
try:
    from VHW.vsd_manager import VSDManager
except Exception:
    class VSDManager:  # minimal shim
        def get(self, key: str, default: Any = None) -> Any:
            return default
        def store(self, key: str, value: Any) -> None:
            return

try:
    # Optional dehydrator/rehydrator (used only if present)
    from Neuralis_AI.AI_processor import Rehydrator, Dehydrator
except Exception:
    Rehydrator = None  # type: ignore
    Dehydrator = None  # type: ignore

try:
    from bios.event_bus import get_event_bus
except Exception:
    def get_event_bus():
        class _NoBus:
            def publish(self, event: str, data: Dict[str, Any]) -> None:
                return
        return _NoBus()

# ----------------------------------------------------------------------------
# Optional: Register Neuralis cognition tick listener (minimal, Neuralis-only)
# ----------------------------------------------------------------------------
try:
    from Neuralis_AI.cognition_tick_adapter import register as _register_cogtick
    from Neuralis_AI.cognition_tick_adapter import CognitionDomain as _CogDom
    _register_cogtick(_CogDom.NEURAL_NETWORK)
except Exception:
    pass

# ----------------------------------------------------------------------------
# Local helpers
# ----------------------------------------------------------------------------
def _bios_ready() -> bool:
    try:
        return bool(VSDManager().get("system/bios_boot_ok", False))
    except Exception:
        return False

def _publish(evt: str, payload: Dict[str, Any]) -> None:
    try:
        get_event_bus().publish(evt, payload)
    except Exception:
        logger.warning("EventBus publish failed for %s", evt)


# ----------------------------------------------------------------------------
# Command Router (Voice/Text -> UI Actions + Replies)
# ----------------------------------------------------------------------------
_ROUTER_STARTED = False


def _publish_reply(text: str, level: str = "ok") -> None:
    try:
        get_event_bus().publish(
            "control_center.telemetry.neuralis",
            {"ts": time.time(), "text": str(text), "level": str(level)}
        )
    except Exception:
        pass


def _route_action(cmd: str, vsd: Any) -> bool:
    """Parse simple commands and emit UI actions; return True if handled."""
    s = cmd.strip().lower()
    bus = get_event_bus()

    # Panel switching
    panels = {
        "miner": "miner",
        "prediction": "prediction",
        "neuralis": "neuralis",
        "system": "system",
        "settings": "settings",
    }
    m = re.search(r"switch\s+(?:to\s+)?(miner|prediction|neuralis|system|settings)", s)
    if m:
        p = panels.get(m.group(1), "miner")
        bus.publish("control_center.ui.action", {"type": "switch_panel", "panel": p, "note": "Switched to %s" % p})
        return True

    # Sidebar controls
    if "collapse sidebar" in s:
        bus.publish("control_center.ui.action", {"type": "sidebar", "mode": "collapse", "note": "Sidebar collapsed"})
        return True
    if "expand sidebar" in s:
        bus.publish("control_center.ui.action", {"type": "sidebar", "mode": "expand", "note": "Sidebar expanded"})
        return True
    if "toggle sidebar" in s:
        bus.publish("control_center.ui.action", {"type": "sidebar", "mode": "toggle", "note": "Sidebar toggled"})
        return True

    # Logs / mic / quit
    if "clear log" in s or "clear logs" in s:
        bus.publish("control_center.ui.action", {"type": "clear_log", "note": "Logs cleared"})
        return True
    if "toggle voice" in s or "toggle mic" in s:
        bus.publish("control_center.ui.action", {"type": "toggle_mic", "note": "Voice toggled"})
        return True
    if s in ("quit", "exit", "shutdown") or "quit app" in s or "exit app" in s:
        bus.publish("control_center.ui.action", {"type": "shutdown", "note": "Shutting down"})
        return True

    return False


def _status_reply(which: str, vsd: Any) -> str:
    """Build a compact status line for miner/prediction/system/ai."""
    w = which.lower().strip()
    try:
        if w == "miner":
            idx = vsd.get("telemetry/metrics/index", []) or []
            nets = [str(x).upper() for x in idx] if isinstance(idx, list) else []
            rows = []
            for n in nets[:5]:
                cur = vsd.get(f"telemetry/metrics/{n}/current", {}) or {}
                hs = float(cur.get("hashes_submitted_hs", 0.0))
                rows.append(f"{n}:{hs:.1f}h/s")
            return "Miner " + (", ".join(rows) if rows else "no data")
        if w == "prediction":
            sigs = vsd.get("telemetry/predictions/latest", []) or []
            if isinstance(sigs, list) and sigs:
                s = sigs[0]
                return "Prediction %s conf=%.2f" % (str(s.get("symbol", "")), float(s.get("avg_confidence", 0.0)))
            return "Prediction no data"
        if w == "system":
            dev = vsd.get("system/device_snapshot", {}) or {}
            cpu = float(((dev.get("cpu") or {}).get("util") or 0.0)) * 100.0
            mem = float(((dev.get("memory") or {}).get("util") or 0.0)) * 100.0
            gpu = float(((dev.get("gpu") or {}).get("util") or 0.0)) * 100.0
            return "System CPU=%.1f%% MEM=%.1f%% GPU=%.1f%%" % (cpu, mem, gpu)
        if w in ("ai", "neuralis"):
            meta = vsd.get("neuralis/fullstate/meta", {}) or {}
            path = vsd.get("neuralis/fullstate/path", "")
            return "AI path=%s dims=%s" % (str(path or ""), str(meta.get("dims", "?")))
    except Exception:
        pass
    return which + " status not available"


def _on_voice_input(payload: Dict[str, Any]) -> None:
    try:
        vsd = VSDManager()
        txt = str((payload or {}).get("text", "")).strip()
        if not txt:
            return
        if not _bios_ready():
            _publish_reply("BIOS not ready", level="warn")
            return

        s = txt.lower()
        # Routing actions
        if _route_action(s, vsd):
            VSDManager().store("neuralis/last_class", "action")
            return

        # Status queries
        m = re.search(r"(miner|prediction|system|ai)\s+status", s)
        if m or s in ("status",):
            which = m.group(1) if m else "system"
            msg = _status_reply(which, vsd)
            _publish_reply(msg, level="ok")
            VSDManager().store("neuralis/last_class", "status")
            return

        # Help
        if "help" in s or s in ("?",):
            _publish_reply("try: switch to miner | collapse sidebar | clear log | status | miner status | quit", level="ok")
            VSDManager().store("neuralis/last_class", "help")
            return

        # Fallback echo
        _publish_reply("echo: " + txt, level="ok")
        VSDManager().store("neuralis/last_class", "echo")
    except Exception:
        pass


def start_command_router() -> bool:
    """Start Neuralis command router once; returns True if started now."""
    global _ROUTER_STARTED
    if _ROUTER_STARTED:
        return False
    try:
        bus = get_event_bus()
        bus.subscribe("neuralis.voice.input", _on_voice_input, priority=0)
        _ROUTER_STARTED = True
        logger.info("Neuralis CommandRouter started")
        return True
    except Exception:
        return False

# ----------------------------------------------------------------------------
# API: Load paths
# ----------------------------------------------------------------------------
def load_on_activation(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attempt to rehydrate AI at activation time from context hints.
    Returns dict with 'ok' flag and details.
    """
    if not _bios_ready():
        logger.warning("AICore load_on_activation aborted: BIOS not ready")
        return {"ok": False, "error": "bios_not_ready"}

    vsd = VSDManager()
    t0 = time.time()
    try:
        ai_hint = context.get("ai_fullstate_path") or vsd.get("ai/fullstate/path", "")
        pe_hint = context.get("pe_fullstate_path") or vsd.get("pe/fullstate/path", "")

        result: Dict[str, Any] = {"ok": True, "ai": None, "pe": None}
        if ai_hint:
            result["ai"] = load_ai_fullstate(ai_hint, kind="neuralis")
        if pe_hint:
            result["pe"] = load_ai_fullstate(pe_hint, kind="prediction")

        dt = time.time() - t0
        logger.info("AICore load_on_activation complete in %.3fs", dt)
        return result
    except Exception as e:
        logger.exception("AICore load_on_activation failed")
        return {"ok": False, "error": str(e)}

def load_ai_fullstate(path: str, *, kind: str = "neuralis") -> Dict[str, Any]:
    """
    Load a fullstate file (typically JSON) from disk/VSD and register it.
    'kind' may be 'neuralis' or 'prediction'.
    """
    if not _bios_ready():
        logger.warning("AICore load_ai_fullstate aborted: BIOS not ready")
        return {"ok": False, "error": "bios_not_ready", "path": path, "kind": kind}

    t0 = time.time()
    try:
        if Rehydrator is None:
            logger.warning("Rehydrator not available; skipping fullstate load")
            return {"ok": False, "error": "rehydrator_missing", "path": path, "kind": kind}

        rh = Rehydrator()
        meta = rh.load(path)
        # Persist reference for later
        VSDManager().store(f"{kind}/fullstate/path", path)
        VSDManager().store(f"{kind}/fullstate/meta", meta)
        _publish(f"{'ai' if kind=='neuralis' else 'pe'}.rehydrated",
                 {"ts": time.time(), "path": path, "kind": kind})
        dt = time.time() - t0
        logger.info("AICore %s fullstate loaded in %.3fs: %s", kind, dt, path)
        return {"ok": True, "path": path, "kind": kind, "meta": meta}
    except Exception as e:
        logger.exception("AICore load_ai_fullstate failed for %s", path)
        return {"ok": False, "error": str(e), "path": path, "kind": kind}

def install_from_statevector(text: str, *, kind: str = "neuralis") -> Dict[str, Any]:
    """
    Install from a state-vector text (ascii_floatmap_v1) into a fullstate file
    using Dehydrator->Rehydrator pipeline when available.
    """
    if not _bios_ready():
        logger.warning("AICore install_from_statevector aborted: BIOS not ready")
        return {"ok": False, "error": "bios_not_ready", "kind": kind}

    try:
        if Dehydrator is None or Rehydrator is None:
            logger.warning("Dehydrator/Rehydrator missing; cannot install")
            return {"ok": False, "error": "rehydration_pipeline_missing", "kind": kind}

        # Dehydrate to a temp json path recorded in VSD
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        out_path = f"VHW/VD/state_vectors/{kind}_fullstate_{stamp}.json"
        Dehydrator().pack(".", out_path, text)
        VSDManager().store(f"{kind}/fullstate/path", out_path)

        # Immediately rehydrate to verify integrity and capture meta
        meta = Rehydrator().load(out_path)
        VSDManager().store(f"{kind}/fullstate/meta", meta)

        _publish(f"{'ai' if kind=='neuralis' else 'pe'}.rehydrated",
                 {"ts": time.time(), "path": out_path, "kind": kind})
        logger.info("AICore %s installed from state-vector: %s", kind, out_path)
        return {"ok": True, "kind": kind, "path": out_path, "meta": meta}
    except Exception as e:
        logger.exception("AICore install_from_statevector failed")
        return {"ok": False, "error": str(e), "kind": kind}

# ----------------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------------
def status() -> Dict[str, Any]:
    try:
        vsd = VSDManager()
        return {
            "bios_ready": _bios_ready(),
            "ai_path": vsd.get("neuralis/fullstate/path", ""),
            "pe_path": vsd.get("prediction/fullstate/path", ""),
            "ai_meta": vsd.get("neuralis/fullstate/meta", {}),
            "pe_meta": vsd.get("prediction/fullstate/meta", {}),
            "ts": time.time(),
        }
    except Exception as e:
        logger.warning("AICore status failed: %s", str(e))
        return {"bios_ready": _bios_ready(), "error": str(e)}
# ============================================================================
# End of Neuralis_AI/ai_core.py
# ============================================================================
