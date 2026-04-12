# ============================================================================
# VirtualMiner / VHW
# ASCII-ONLY SOURCE FILE
# File: gpu_initiator.py
# Purpose:
#   GPU-anchored initiator for Virtual Hardware. The initiator now performs a
#   live photonic substrate cycle instead of a passive sustain-only heartbeat.
#   It probes for a GPU, actuates the pulse runtime, and stores the emergent
#   lattice/trace metrics into VSD for BIOS and Control Center consumers.
#
#   - ASCII-only
#   - No hard dependency on CUDA libraries
#   - Safe to run on systems without GPUs (becomes a lightweight no-op)
# ----------------------------------------------------------------------------
from __future__ import annotations

from typing import Optional, Dict, Any
import threading
import subprocess
import time
import os
import logging


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
_logger = logging.getLogger("VHW.gpu_initiator")
if not _logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter(
        fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logging.Formatter.converter = time.gmtime
    h.setFormatter(fmt)
    _logger.addHandler(h)
_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# VSD FALLBACK
# ---------------------------------------------------------------------------
try:
    from VHW.vsd_manager import VSDManager
except Exception:
    class VSDManager:
        def __init__(self):
            self._kv = {}

        def get(self, k, d=None):
            return self._kv.get(k, d)

        def store(self, k, v):
            self._kv[k] = v


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


class _GPUInitiator:
    def __init__(
        self,
        vsd: Optional[Any],
        sustain_pct: float = 0.05,
        actuation_profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vsd = vsd or VSDManager()
        self.sustain_pct = max(0.0, min(1.0, float(sustain_pct)))
        self.actuation_profile = self._normalize_profile(actuation_profile)
        self._stop = False
        self._thr: Optional[threading.Thread] = None
        self._gpu_available = False
        self._last_probe = 0.0
        self._trace_state: Dict[str, Any] = {}
        self._previous_profile: Dict[str, Any] = {}
        self._previous_memory_basin_state: Dict[str, Any] = {}
        self._previous_scheduler_state: Dict[str, Any] = {}
        self._previous_process_state: Dict[str, Any] = {}
        self._frame_history: list[Dict[str, Any]] = []

    def _normalize_profile(self, profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(profile or {})
        sample_period_s = max(
            0.01,
            _safe_float(cfg.get("telemetry_sample_period_s", max(0.05, 0.50 - (0.35 * self.sustain_pct))), 0.10),
        )
        loop_period_s = max(
            0.05,
            _safe_float(cfg.get("loop_period_s", max(sample_period_s, 0.25)), max(sample_period_s, 0.25)),
        )
        cfg.setdefault("program_id", "gpu_initiator_photonic")
        cfg.setdefault("profile_name", "photonic_actuation")
        cfg.setdefault("telemetry_mode", "live_startup")
        cfg.setdefault("actuation_backend", "vulkan_calibration")
        cfg.setdefault("capture_sleep", False)
        cfg.setdefault("telemetry_sample_period_s", float(sample_period_s))
        cfg.setdefault("history_size", 8)
        cfg.setdefault("horizon_frames", 2)
        cfg.setdefault("pulse_cycles", 1)
        cfg.setdefault("actuation_settle_s", 0.004)
        cfg.setdefault("loop_period_s", float(loop_period_s))
        cfg.setdefault("vulkan_element_count", 4096)
        cfg.setdefault("vulkan_iterations", 12)
        cfg.setdefault("vulkan_frequency", 0.245)
        cfg.setdefault("vulkan_amplitude", 0.18)
        cfg.setdefault("vulkan_voltage", 0.33)
        cfg.setdefault("vulkan_current", 0.33)
        cfg.setdefault("force_gpu_available", False)
        return cfg

    def start(self) -> None:
        if self._thr:
            return
        self._probe_gpu()
        self._mark_state(init=True)
        self._thr = threading.Thread(target=self._loop, daemon=True, name="gpu_initiator")
        self._thr.start()
        _logger.info(
            "GPU Initiator started (available=%s, sustain_pct=%.3f, profile=%s, mode=%s)",
            self._gpu_available,
            self.sustain_pct,
            str(self.actuation_profile.get("profile_name", "photonic_actuation")),
            str(self.actuation_profile.get("actuation_backend", "passive")),
        )

    def stop(self, timeout: float = 2.0) -> None:
        self._stop = True
        if self._thr:
            try:
                self._thr.join(timeout=timeout)
            except Exception:
                pass
            self._thr = None
        self._mark_state(init=False)
        _logger.info("GPU Initiator stopped")

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        period = max(0.05, _safe_float(self.actuation_profile.get("loop_period_s", 0.25), 0.25))
        while not self._stop:
            started = time.perf_counter()
            try:
                now = time.time()
                if now - self._last_probe > 10.0:
                    self._probe_gpu()
                if self._gpu_available:
                    self._run_photonic_cycle(now)
                else:
                    self._store("vhw/gpu/heartbeat", {
                        "ts": now,
                        "gpu_available": False,
                        "sustain_pct": float(self.sustain_pct),
                        "profile_name": str(self.actuation_profile.get("profile_name", "photonic_actuation")),
                        "mode": "idle_probe",
                    })
            except Exception as exc:
                self._store("vhw/gpu/photonic_error", {
                    "ts": time.time(),
                    "error": str(exc),
                })
                _logger.warning("GPU initiator cycle failed: %s", exc, exc_info=True)
            elapsed = max(0.0, float(time.perf_counter() - started))
            sleep_s = max(0.05, period - elapsed)
            time.sleep(sleep_s)

    def _run_photonic_cycle(self, now: float) -> None:
        from VHW.gpu_pulse_runtime import run_live_photonic_substrate_cycle

        cycle_payload = dict(self.actuation_profile)
        cycle = run_live_photonic_substrate_cycle(
            payload=cycle_payload,
            previous_trace_state=self._trace_state,
            previous_profile=self._previous_profile,
            frame_history=self._frame_history,
            previous_memory_basin_state=self._previous_memory_basin_state,
            previous_scheduler_state=self._previous_scheduler_state,
            previous_process_state=self._previous_process_state,
        )
        self._trace_state = dict(cycle.get("trace_state", {}) or {})
        self._previous_profile = dict(cycle.get("profile", {}) or {})
        self._previous_memory_basin_state = dict(cycle.get("memory_basin_state", {}) or {})
        self._previous_scheduler_state = dict(cycle.get("scheduler_state", {}) or {})
        self._previous_process_state = dict(cycle.get("process_state", {}) or {})
        self._frame_history = list(cycle.get("frames", []) or [])[-max(4, int(self.actuation_profile.get("history_size", 8) or 8)):]

        frame = dict(cycle.get("frame", {}) or {})
        latency = dict(cycle.get("latency_calibration", {}) or {})
        actuation = dict(cycle.get("actuation_summary", {}) or {})
        trace = dict(self._trace_state or {})
        result = dict(cycle.get("result", {}) or {})
        memory_basin = dict(cycle.get("memory_basin_state", {}) or {})
        scheduler = dict(cycle.get("scheduler_state", {}) or {})
        process_state = dict(cycle.get("process_state", {}) or {})

        heartbeat = {
            "ts": now,
            "gpu_available": bool(self._gpu_available),
            "sustain_pct": float(self.sustain_pct),
            "profile_name": str(self.actuation_profile.get("profile_name", "photonic_actuation")),
            "mode": "photonic_actuation",
            "trace_gate": float(trace.get("trace_gate", 0.0)),
            "trace_alignment": float(trace.get("trace_alignment", 0.0)),
            "phase_ring_strength": float(latency.get("phase_ring_strength", frame.get("phase_ring_strength", 0.0))),
            "shared_vector_collapse_gate": float(latency.get("shared_vector_collapse_gate", frame.get("shared_vector_collapse_gate", 0.0))),
            "zero_point_crossover_gate": float(latency.get("zero_point_crossover_gate", frame.get("zero_point_crossover_gate", 0.0))),
            "temporal_relativity_norm": float(latency.get("temporal_relativity_norm", frame.get("temporal_relativity_norm", 0.0))),
            "field_interference_norm": float(latency.get("field_interference_norm", frame.get("field_interference_norm", 0.0))),
            "resonant_interception_inertia": float(latency.get("resonant_interception_inertia", frame.get("resonant_interception_inertia", 0.0))),
            "gpu_pulse_phase_effect": float(latency.get("gpu_pulse_phase_effect", frame.get("gpu_pulse_phase_effect", 0.0))),
            "scheduler_mode": str(scheduler.get("scheduling_mode", "")),
            "process_mode": str(process_state.get("process_mode", "")),
            "mining_resonance_gate": float(process_state.get("mining_resonance_gate", 0.0)),
            "actuation_mode": str(actuation.get("mode", "passive")),
            "dispatch_elapsed_ms_mean": float(actuation.get("dispatch_elapsed_ms_mean", 0.0)),
        }
        self._store("vhw/gpu/heartbeat", heartbeat)
        self._store("vhw/gpu/photonic_cycle", {
            "ts": now,
            "telemetry_source": str(cycle.get("telemetry_source", "live_startup")),
            "result": result,
            "actuation_summary": actuation,
        })
        self._store("vhw/gpu/photonic_frame", frame)
        self._store("vhw/gpu/photonic_latency", latency)
        self._store("vhw/gpu/photonic_trace", trace)
        self._store("vhw/gpu/photonic_memory_basin", memory_basin)
        self._store("vhw/gpu/photonic_scheduler", scheduler)
        self._store("vhw/gpu/photonic_process", process_state)
        self._store("vhw/gpu/photonic_forecast", list(cycle.get("forecast_preview", []) or []))
        self._store("vhw/gpu/photonic_status", {
            "ts": now,
            "profile_name": str(self.actuation_profile.get("profile_name", "photonic_actuation")),
            "phase_ring_strength": float(latency.get("phase_ring_strength", 0.0)),
            "phase_ring_closure": float(latency.get("phase_ring_closure", 0.0)),
            "shared_vector_phase_lock": float(latency.get("shared_vector_phase_lock", 0.0)),
            "inertial_basin_strength": float(latency.get("inertial_basin_strength", 0.0)),
            "temporal_relativity_norm": float(latency.get("temporal_relativity_norm", 0.0)),
            "zero_point_line_distance": float(latency.get("zero_point_line_distance", 0.0)),
            "field_interference_norm": float(latency.get("field_interference_norm", 0.0)),
            "resonant_interception_inertia": float(latency.get("resonant_interception_inertia", 0.0)),
            "active_basin_name": str(memory_basin.get("active_basin_name", "")),
            "active_zone_name": str(scheduler.get("active_zone_name", "")),
            "process_mode": str(process_state.get("process_mode", "")),
            "mining_resonance_gate": float(process_state.get("mining_resonance_gate", 0.0)),
            "gradient_spectral_id": str(latency.get("gradient_spectral_id", "")),
            "trajectory_spectral_id": str(latency.get("trajectory_spectral_id", "")),
        })

    # ------------------------------------------------------------------
    def _probe_gpu(self) -> None:
        self._last_probe = time.time()
        if bool(self.actuation_profile.get("force_gpu_available", False)):
            self._gpu_available = True
            return
        try:
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if vis is not None and vis.strip() == "":
                self._gpu_available = False
                return
        except Exception:
            pass
        try:
            res = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1.0)
            self._gpu_available = bool(res.returncode == 0 and b"GPU" in (res.stdout or b""))
        except Exception:
            self._gpu_available = bool(self._gpu_available)

    def _mark_state(self, init: bool) -> None:
        safe_profile = {
            str(key): value
            for key, value in dict(self.actuation_profile or {}).items()
            if not callable(value) and not str(key).startswith("_")
        }
        self._store("vhw/gpu/init_ok", bool(init))
        self._store("vhw/gpu/available", bool(self._gpu_available))
        self._store("vhw/gpu/sustain_pct", float(self.sustain_pct))
        self._store("vhw/gpu/actuation_profile", safe_profile)
        self._store("vhw/gpu/photonic_profile", safe_profile)

    def _store(self, key: str, value: Any) -> None:
        try:
            self.vsd.store(key, value)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_instance: Optional[_GPUInitiator] = None


def start_gpu_initiator(
    vsd: Optional[Any] = None,
    sustain_pct: float = 0.05,
    actuation_profile: Optional[Dict[str, Any]] = None,
) -> _GPUInitiator:
    global _instance
    if _instance is None:
        _instance = _GPUInitiator(
            vsd=vsd,
            sustain_pct=sustain_pct,
            actuation_profile=actuation_profile,
        )
        _instance.start()
    return _instance


def stop_gpu_initiator(timeout: float = 2.0) -> None:
    global _instance
    if _instance is not None:
        try:
            _instance.stop(timeout=timeout)
        except Exception:
            pass
        _instance = None
