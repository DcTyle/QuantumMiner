# ==========================================================================
# Imports (must be at top)
# ==========================================================================
from __future__ import annotations
from typing import Dict, Any
import time
import sys
import logging
import os

# ---------------------------------------------------------------------------
# Attach Engines (Miner + Prediction)
# ---------------------------------------------------------------------------
def _attach_engines(ctx: Dict[str, Any], boot_obj: Any) -> None:
    # Miner Engine
    try:
        from miner.miner_engine import MinerEngine, register_miner_autostart
        runtime_cfg = dict(getattr(boot_obj, "runtime_cfg", {}) or {})
        system_cfg = dict(runtime_cfg.get("system", {}) or {}) if isinstance(runtime_cfg, dict) else {}
        tick_interval_s = 0.25
        try:
            tick_interval_s = float(system_cfg.get("mining_period_s", runtime_cfg.get("tick_interval_s", 0.25)) or 0.25)
        except Exception:
            tick_interval_s = 0.25
        engine = MinerEngine(
            compute_manager=getattr(boot_obj, "compute_manager", None),
            share_allocator=getattr(boot_obj, "share_allocator", None),
            submitter=getattr(boot_obj, "submitter", None),
            failsafe=getattr(boot_obj, "failsafe", None),
            read_telemetry=ctx["env"]["reader"],
            vsd=_vsd,
            handler_miner=getattr(boot_obj, "handler_miner", None),
            tick_interval_s=tick_interval_s,
            network_cfg=dict(runtime_cfg.get("network", {}) or {}),
            nonce_cfg=dict(getattr(boot_obj, "nonce_cfg", {}) or {}),
        )
        register_miner_autostart(engine)
        ctx["miner_engine"] = engine
        _logger.info("MinerEngine attached and autostart registered.")
    except ImportError as ie:
        _logger.error("MinerEngine module import failed: %s", ie, exc_info=True)
        ctx["miner_engine"] = None
    except Exception as exc:
        _logger.error("Failed to attach MinerEngine: %s", exc, exc_info=True)
        ctx["miner_engine"] = None

    # Prediction Engine
    try:
        from prediction_engine.prediction_engine import PredictionEngine
        from bios.scheduler import MurphyWatchdog
        from config.manager import ConfigManager
        env = {
            "vsd": _vsd,
            "failsafe": getattr(boot_obj, "failsafe", None),
            "allocator_ref": getattr(boot_obj, "share_allocator", None),
            "telemetry_reader": ctx["env"]["reader"],
        }
        api_key = str(ConfigManager.get("prediction.api_key", "") or "")
        api_secret = str(ConfigManager.get("prediction.api_secret", "") or "")
        if api_key and not os.environ.get("CRYPTOCOM_API_KEY"):
            os.environ["CRYPTOCOM_API_KEY"] = api_key
        if api_secret and not os.environ.get("CRYPTOCOM_API_SECRET"):
            os.environ["CRYPTOCOM_API_SECRET"] = api_secret
        wd = MurphyWatchdog(
            env=env,
            failsafe=env["failsafe"],
            allocator=env["allocator_ref"],
            vsd=_vsd,
        )
        pe = PredictionEngine(
            watchdog=wd,
            vsd=_vsd,
            base_quote=str(ConfigManager.get("prediction.base_quote", "USDT") or "USDT").upper(),
            max_assets=int(ConfigManager.get("prediction.max_assets", 12) or 12),
            lanes_per_cluster=int(ConfigManager.get("prediction.lanes_per_cluster", 6) or 6),
            max_clusters=int(ConfigManager.get("prediction.max_clusters", 4) or 4),
            min_confidence=float(ConfigManager.get("prediction.min_confidence", 0.99) or 0.99),
            candle_timeframe=str(ConfigManager.get("prediction.candle_timeframe", "1h") or "1h"),
            candle_limit=int(ConfigManager.get("prediction.candle_limit", 400) or 400),
        )
        pe.initialize()
        ctx["prediction_engine"] = pe
        _logger.info("PredictionEngine attached and initialized.")
    except ImportError as ie:
        _logger.error("PredictionEngine module import failed: %s", ie, exc_info=True)
        ctx["prediction_engine"] = None
    except Exception as exc:
        _logger.error("Failed to attach PredictionEngine: %s", exc, exc_info=True)
        ctx["prediction_engine"] = None
# ============================================================================
# Quantum Application / bios
# ASCII-ONLY SOURCE FILE
# File: main_runtime.py
# Version: v7.5 - BIOS Silent-Error Purge (Miner + Prediction Autostart Replay Fix)
# ============================================================================

from typing import Dict, Any
import time
import sys
import logging
import os

# ---------------------------------------------------------------------------
def _init_logger() -> logging.Logger:
    logger = logging.getLogger("bios.main_runtime")
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

_logger = _init_logger()

# ---------------------------------------------------------------------------
# BIOS boot entry (deferred import to avoid circular warnings)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# EventBus (strict import)
# ---------------------------------------------------------------------------
from bios.event_bus import get_event_bus
_bus = get_event_bus()

# ---------------------------------------------------------------------------
# VSD Manager (strict import)
# ---------------------------------------------------------------------------
from VHW.vsd_manager import VSDManager
_vsd = VSDManager()

# ---------------------------------------------------------------------------
# Telemetry Console (optional, no stub fallback)
# ---------------------------------------------------------------------------
try:
    from Control_Center.telemetry_console_live import LiveConsoleRunner
    CONSOLE_OK = True
except Exception as exc:
    _logger.warning(
        "Failed to import LiveConsoleRunner; live console disabled: %s",
        exc,
        exc_info=True,
    )
    CONSOLE_OK = False

# ---------------------------------------------------------------------------
# Telemetry Reader (Monitor first, fallback to VSD)
# ---------------------------------------------------------------------------
def _make_reader(boot_obj: Any):
    if hasattr(boot_obj, "monitor") and hasattr(boot_obj.monitor, "frame"):
        def _reader() -> Dict[str, Any]:
            try:
                frame = dict(boot_obj.monitor.frame() or {})
            except Exception:
                frame = {}
            try:
                live = _vsd.get("telemetry/global", {}) or {}
            except Exception:
                live = {}
            merged = dict(live) if isinstance(live, dict) else {}
            if isinstance(frame, dict):
                merged.update(frame)
            return merged

        return _reader
    raise RuntimeError("Telemetry reader unavailable: boot_obj.monitor.frame is required")


_runtime_ctx: Dict[str, Any] | None = None


def _emit_miner_control(text: str, level: str = "info") -> None:
    try:
        _bus.publish("control_center.telemetry.miner", {
            "ts": time.time(),
            "text": str(text),
            "level": str(level),
        })
    except Exception:
        pass


def _store_miner_control_state(
    phase: str,
    paused: bool,
    note: str = "",
    source: str = "bios.main_runtime",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "ts": time.time(),
        "phase": str(phase),
        "paused": bool(paused),
        "note": str(note or ""),
        "source": str(source or "bios.main_runtime"),
    }
    if isinstance(extra, dict):
        state.update(extra)
    try:
        _vsd.store("miner/control/state", state)
        _vsd.store("system/miner_paused", bool(paused))
    except Exception:
        pass
    return state


def pause_miner(ctx: Dict[str, Any], note: str = "", source: str = "control_center") -> Dict[str, Any]:
    if not isinstance(ctx, dict):
        return {}

    current = _vsd.get("miner/control/state", {}) or {}
    if isinstance(current, dict) and bool(current.get("paused", False)):
        state = _store_miner_control_state(
            "paused",
            True,
            note or str(current.get("note", "")),
            source,
            {"already_paused": True},
        )
        _emit_miner_control("Miner already paused", level="warn")
        return state

    _store_miner_control_state("quiescing", False, note, source)

    engine = ctx.get("miner_engine")
    adapter = ctx.get("stratum_adapter")
    boot_obj = ctx.get("boot")
    submitter = getattr(boot_obj, "submitter", None) if boot_obj is not None else None

    if engine is not None and hasattr(engine, "pause"):
        try:
            engine.pause(note=note, source=source)
        except Exception as exc:
            _logger.warning("Failed to pause MinerEngine: %s", exc, exc_info=True)

    if adapter is not None and hasattr(adapter, "pause"):
        try:
            adapter.pause(note=note, source=source)
        except Exception as exc:
            _logger.warning("Failed to pause StratumAdapter: %s", exc, exc_info=True)

    queue_drained = False
    queue_depth = 0
    if submitter is not None and hasattr(submitter, "pause"):
        try:
            queue_drained = bool(submitter.pause(note=note, source=source, drain_timeout_s=1.0))
            queue_depth = int(getattr(getattr(submitter, "q", None), "qsize", lambda: 0)())
        except Exception as exc:
            _logger.warning("Failed to pause Submitter: %s", exc, exc_info=True)

    state = _store_miner_control_state(
        "paused",
        True,
        note,
        source,
        {
            "engine_paused": bool(getattr(engine, "is_paused", lambda: False)()),
            "adapter_paused": bool(getattr(adapter, "is_paused", lambda: False)()) if adapter is not None else False,
            "submitter_paused": bool(getattr(submitter, "is_paused", lambda: False)()) if submitter is not None else False,
            "queue_drained": bool(queue_drained),
            "queue_depth": int(queue_depth),
        },
    )
    try:
        _bus.publish("miner.control.paused", dict(state))
    except Exception:
        pass
    _emit_miner_control("Miner paused", level="warn")
    return state


def _wire_control_center_commands(ctx: Dict[str, Any]) -> None:
    if not isinstance(ctx, dict) or bool(ctx.get("_control_center_commands_wired", False)):
        return

    def _on_pause_command(payload: Dict[str, Any]) -> None:
        try:
            note = str((payload or {}).get("note", "Miner pause requested"))
            source = str((payload or {}).get("source", "control_center"))
            pause_miner(ctx, note=note, source=source)
        except Exception as exc:
            _logger.warning("Pause command handler failed: %s", exc, exc_info=True)

    try:
        _bus.subscribe("control_center.cmd.pause_miner", _on_pause_command, priority=0)
        ctx["_control_center_commands_wired"] = True
    except Exception as exc:
        _logger.warning("Failed to wire Control Center commands: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# RUN BIOS + RUNTIME
# ---------------------------------------------------------------------------
def run(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    global _runtime_ctx
    from bios.boot import start as start_boot
    boot_obj = start_boot(vsd=_vsd)
    required_attrs = ["compute_manager", "share_allocator", "submitter", "failsafe"]
    missing = [attr for attr in required_attrs if not hasattr(boot_obj, attr) or getattr(boot_obj, attr) is None]
    if boot_obj is None or missing:
        _logger.critical(f"Production boot failed: missing attributes: {missing}")
        raise RuntimeError(f"Production boot failed: missing attributes: {missing}")
    # ASCII-only enforcement for boot attributes
    def _ascii_check(val):
        if isinstance(val, str):
            val.encode('ascii')
        elif isinstance(val, dict):
            for k, v in val.items():
                _ascii_check(k)
                _ascii_check(v)
        elif isinstance(val, list):
            for item in val:
                _ascii_check(item)
    for attr in required_attrs:
        _ascii_check(getattr(boot_obj, attr))

    env = {"reader": _make_reader(boot_obj), "vsd": _vsd}

    # GPU Initiator: anchor virtual hardware on GPU with live photonic actuation.
    try:
        from VHW.gpu_initiator import start_gpu_initiator
        runtime_cfg = dict(getattr(boot_obj, "runtime_cfg", {}) or {})
        initiator_cfg = dict(runtime_cfg.get("vhw_gpu_initiator", {}) or {})
        sustain_pct = 0.05
        try:
            sustain_pct = float(
                os.environ.get(
                    "VHW_GPU_SUSTAIN_PCT",
                    initiator_cfg.get("sustain_pct", 0.05),
                )
            )
        except Exception:
            sustain_pct = 0.05
        photonic_profile = dict(initiator_cfg)
        gpu_init = start_gpu_initiator(
            vsd=_vsd,
            sustain_pct=sustain_pct,
            actuation_profile=photonic_profile,
        )
        _logger.info(
            "GPU initiator started (sustain_pct=%.3f, profile=%s, backend=%s)",
            sustain_pct,
            str(photonic_profile.get("profile_name", "photonic_actuation")),
            str(photonic_profile.get("actuation_backend", "vulkan_calibration")),
        )
    except Exception as exc:
        gpu_init = None
        _logger.warning("GPU initiator unavailable: %s", exc)

    ctx: Dict[str, Any] = {
        "boot": boot_obj,
        "console": None,
        "env": env,
        "gpu_initiator": gpu_init,
        "vsd": _vsd,
        "bus": _bus,
    }

    _attach_engines(ctx, boot_obj)

    # Enforce fatal policy in production: both engines must attach.
    try:
        prod_mode = bool(config and str(config.get("mode", "")).lower() == "production")
    except Exception:
        prod_mode = False
    if prod_mode:
        if not ctx.get("miner_engine"):
            _logger.critical("Fatal: miner_engine missing in production mode.")
            raise RuntimeError("Fatal: miner_engine missing in production mode.")
        if not ctx.get("prediction_engine"):
            _logger.critical("Fatal: prediction_engine missing in production mode.")
            raise RuntimeError("Fatal: prediction_engine missing in production mode.")

    # 3-Agent Continuous Integration: CI hooks
    try:
        from bios.scheduler import MurphyWatchdog
        wd = MurphyWatchdog(
            env={
                "vsd": _vsd,
                "failsafe": getattr(boot_obj, "failsafe", None),
                "allocator_ref": getattr(boot_obj, "share_allocator", None),
            },
            failsafe=getattr(boot_obj, "failsafe", None),
            allocator=getattr(boot_obj, "share_allocator", None),
            vsd=_vsd,
        )
        _bus.publish("ci.cycle.start", {"ts": time.time(), "source": "bios.main_runtime"})
        # Simulate CI event triggers
        for event in ["commit", "push", "pull_request"]:
            _bus.publish(f"ci.event.{event}", {"ts": time.time(), "source": "bios.main_runtime"})
            # Neuralis: repo-wide checks
            _bus.publish(f"ci.neuralis.check.{event}", {"ts": time.time(), "status": "pending"})
            # Jarvis: architecture enforcement
            _bus.publish(f"ci.jarvis.validate.{event}", {"ts": time.time(), "status": "pending"})
            # Codex: stage/apply approved changes
            _bus.publish(f"ci.codex.apply.{event}", {"ts": time.time(), "status": "pending"})
        # Murphy Watchdog: monitor and rollback
        _bus.publish("ci.murphy.monitor", {"ts": time.time(), "status": "active"})
        # EventBus: publish CI events to VSD
        _bus.publish("ci.events.published", {"ts": time.time(), "source": "bios.main_runtime"})
    except Exception as exc:
        _logger.warning("MurphyWatchdog unavailable for CI routing: %s", exc)

    # -----------------------------------------------------------------------
    # Start Stratum Adapter to feed real jobs into MinerEngine
    # -----------------------------------------------------------------------
    try:
        from miner.stratum_adapter import StratumAdapter
        engine = ctx.get("miner_engine")
        submitter = getattr(boot_obj, "submitter", None)
        if engine and submitter:
            adapter = StratumAdapter(engine=engine, submitter=submitter, vsd=_vsd, bus=_bus)
            adapter.start()
            # Force lane assignment after engine is started (fix for empty lanes)
            adapter.assign_lanes()
            ctx["stratum_adapter"] = adapter
            _logger.info("StratumAdapter started and wired to Submitter resolver.")
    except Exception as exc:
        _logger.warning("StratumAdapter failed to start: %s", exc, exc_info=True)

    _wire_control_center_commands(ctx)
    _store_miner_control_state("running", False, "Miner runtime active", "bios.main_runtime")
    _runtime_ctx = ctx

    # -----------------------------------------------------------------------
    # CRITICAL FIX:
    # BIOS publishes boot.complete BEFORE engines subscribe, so we replay it.
    # -----------------------------------------------------------------------
    try:
        _bus.publish(
            "boot.complete",
            {
                "ts": time.time(),
                "source": "bios.main_runtime",
                "replayed": True,
            },
        )
    except Exception as exc:
        _logger.warning(
            "Failed to replay boot.complete event: %s",
            exc,
            exc_info=True,
        )

    # Optional Telemetry Console
    try:
        disable_live_console = bool(config and config.get("disable_live_console", False))
        if CONSOLE_OK and not disable_live_console:
            console = LiveConsoleRunner(env, refresh_s=0.5)
            console.start()
            ctx["console"] = console
            _logger.info("Live telemetry console started.")
    except Exception as exc:
        _logger.warning("Live console failed to start: %s", exc, exc_info=True)
        ctx["console"] = None

    ctx["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return ctx

# ---------------------------------------------------------------------------
# SHUTDOWN
# ---------------------------------------------------------------------------
def shutdown(ctx: Dict[str, Any]) -> None:
    if not isinstance(ctx, dict):
        return
    try:
        c = ctx.get("console")
        if c:
            c.stop()
    except Exception as exc:
        _logger.warning(
            "Shutdown cleanup failed while stopping console: %s",
            exc,
            exc_info=True,
        )
    try:
        _bus.publish(
            "system.shutdown",
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "bios.main_runtime",
            },
        )
    except Exception as exc:
        _logger.warning(
            "Failed to publish system.shutdown event: %s",
            exc,
            exc_info=True,
        )

    try:
        adapter = ctx.get("stratum_adapter")
        if adapter is not None and hasattr(adapter, "stop"):
            adapter.stop()
    except Exception as exc:
        _logger.warning("Failed to stop StratumAdapter: %s", exc, exc_info=True)

    try:
        engine = ctx.get("miner_engine")
        if engine is not None and hasattr(engine, "stop"):
            engine.stop()
    except Exception as exc:
        _logger.warning("Failed to stop MinerEngine: %s", exc, exc_info=True)

    try:
        boot_obj = ctx.get("boot")
        submitter = getattr(boot_obj, "submitter", None) if boot_obj is not None else None
        if submitter is not None and hasattr(submitter, "stop"):
            submitter.stop()
    except Exception as exc:
        _logger.warning("Failed to stop Submitter: %s", exc, exc_info=True)

    # Stop GPU initiator if present
    try:
        gi = ctx.get("gpu_initiator") if isinstance(ctx, dict) else None
        if gi is not None:
            from VHW.gpu_initiator import stop_gpu_initiator
            stop_gpu_initiator()
            _logger.info("GPU initiator stopped on shutdown")
    except Exception as exc:
        _logger.warning("Failed to stop GPU initiator: %s", exc)
