# ============================================================================
# Quantum Application / VHW
# ASCII-ONLY SOURCE FILE
# File: vqpu.py
# Version: v4.8.7 Hybrid (final-integration) - PATCHED WITH BIOS LOGGER STANDARD
# Jarvis ADA v4.7 Hybrid Ready
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional
import time
import threading
import logging
import math

# ============================================================================
# Safe fallbacks (ASCII-only stubs)
# ============================================================================
from config.manager import ConfigManager  # type: ignore
from bios.event_bus import get_event_bus  # type: ignore
from VHW.vsd_manager import VSDManager  # type: ignore
from VHW.system_utils import (
    ada_effective_constants,
    ada_latency_kernel,
    ada_phase_update,
    device_snapshot,
    system_headroom,
)

# ============================================================================
# BIOS-LOGGER STANDARD
# ============================================================================
def _mk_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        lg.propagate = False
        h = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg

# ============================================================================
# Helpers
# ============================================================================
def _utc_ts_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _latency_load(headroom_snapshot: Dict[str, Any]) -> float:
    global_util = max(
        0.0,
        min(1.0, float(headroom_snapshot.get("global_util", 0.0))),
    )
    gpu_util = max(
        0.0,
        min(1.0, float(headroom_snapshot.get("gpu_util", 0.0))),
    )
    mem_util = max(
        0.0,
        min(1.0, float(headroom_snapshot.get("mem_util", 0.0))),
    )
    cpu_util = max(
        0.0,
        min(1.0, float(headroom_snapshot.get("cpu_util", 0.0))),
    )
    headroom = max(
        0.0,
        min(1.0, float(headroom_snapshot.get("headroom", 0.0))),
    )
    return max(
        0.0,
        min(
            1.0,
            0.34 * gpu_util
            + 0.26 * mem_util
            + 0.18 * global_util
            + 0.10 * cpu_util
            + 0.12 * (1.0 - headroom),
        ),
    )

# ============================================================================
# VQPU
# ============================================================================
class VQPU:
    """
    Virtual QPU executor (BIOS-gated, queue-buffered, telemetry-integrated).
    """

    def __init__(
        self,
        cfg: Optional[ConfigManager] = None,
        bus_factory: Callable[[], Any] = get_event_bus
    ) -> None:

        self._cfg = cfg or ConfigManager()
        self._bus = bus_factory()
        self._vsd: Optional[VSDManager] = None

        # Logger (BIOS standard)
        self.log = _mk_logger("VHW.VQPU")

        # Tunables
        self._batch_size: int = int(self._cfg.get("vqpu.batch_size", 64))
        self._telemetry_interval_s: float = float(
            self._cfg.get("vqpu.telemetry_interval_s", 1.0)
        )
        self._tick_s: float = float(self._cfg.get("vqpu.tick_s", 0.01))

        # Internal state
        self._queue: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._running = False
        self._thr: Optional[threading.Thread] = None
        self._last_pub = 0.0
        self._exec_count = 0
        self._last_err = ""
        self._phase = 0.0
        self._last_microprocess: Dict[str, Any] = {}
        self._last_benchmark: Dict[str, Any] = {}
        self._microprocess_trace_state: Dict[str, Any] = {}

        # Subscribe to boot.complete
        try:
            self._bus.subscribe("boot.complete", self._on_boot_complete)
        except Exception as e:
            self.log.error("failed subscribing to boot.complete", exc_info=True)

        self.log.info("initialized v4.8.7 Hybrid (patched logging)")

    # ----------------------------------------------------------------------
    # Safety / BIOS gate
    # ----------------------------------------------------------------------
    def _bios_ready(self) -> bool:
        try:
            if self._vsd is None:
                return False
            return bool(self._vsd.get("system/bios_boot_ok", False))
        except Exception:
            return False

    # ----------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------
    def start(self, vsd: Optional[VSDManager] = None) -> None:
        """Start worker loop when BIOS is ready."""
        if self._running:
            return

        self._vsd = vsd or self._vsd or VSDManager()

        if not self._bios_ready():
            self.log.warning("start() blocked: BIOS not ready")
            return

        self._running = True
        self._thr = threading.Thread(
            target=self._loop,
            daemon=True,
            name="vqpu_worker"
        )
        self._thr.start()
        self.log.info("started")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop worker safely."""
        if not self._running:
            return
        self._running = False
        try:
            if self._thr:
                self._thr.join(timeout=timeout)
        except Exception as e:
            self.log.error("stop() join error", exc_info=True)
        finally:
            self._thr = None
            self.log.info("stopped")

    def _on_boot_complete(self, event: Optional[Dict[str, Any]] = None) -> None:
        """EventBus: start on boot.complete."""
        try:
            if not self._running:
                self.start(self._vsd or VSDManager())
        except Exception as e:
            self.log.error("boot.complete handler failed", exc_info=True)

    # ----------------------------------------------------------------------
    # Queue operations
    # ----------------------------------------------------------------------
    def enqueue(self, task: Dict[str, Any]) -> None:
        """Buffered enqueue."""
        if not isinstance(task, dict):
            return
        with self._lock:
            self._queue.append(task)

    def execute(self, opcode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one QPU op immediately."""
        try:
            opcode_text = str(opcode or "").strip().lower()
            if opcode_text in ("substrate_microprocess", "gpu_pulse_microprocess"):
                from VHW.gpu_pulse_runtime import run_substrate_microprocess

                res = run_substrate_microprocess(
                    dict(payload or {}),
                    previous_trace_state=dict(self._microprocess_trace_state or {}),
                )
                self._microprocess_trace_state = dict(res.get("trace_state", {}))
                self._last_microprocess = dict(res)
                res["opcode"] = str(opcode)
            elif opcode_text in ("substrate_microprocess_benchmark", "substrate_benchmark", "gpu_pulse_benchmark"):
                from VHW.gpu_pulse_runtime import benchmark_substrate_microprocess

                res = benchmark_substrate_microprocess(
                    dict(payload or {}),
                    previous_trace_state=dict(self._microprocess_trace_state or {}),
                )
                substrate_result = dict(res.get("substrate_result", {}) or {})
                self._microprocess_trace_state = dict(substrate_result.get("trace_state", {}))
                self._last_benchmark = dict(res)
                self._last_microprocess = dict(substrate_result)
                res["opcode"] = str(opcode)
            elif opcode_text in ("telemetry_resonance", "telemetry_noise_resonance"):
                from VHW.gpu_pulse_runtime import run_substrate_telemetry_resonance

                res = run_substrate_telemetry_resonance(
                    dict(payload or {}),
                    previous_trace_state=dict(self._microprocess_trace_state or {}),
                )
                self._microprocess_trace_state = dict(res.get("trace_state", {}))
                self._last_microprocess = dict(res)
                res["opcode"] = str(opcode)
            elif opcode_text in ("telemetry_resonance_benchmark", "telemetry_noise_benchmark"):
                from VHW.gpu_pulse_runtime import benchmark_telemetry_resonance

                res = benchmark_telemetry_resonance(
                    dict(payload or {}),
                    previous_trace_state=dict(self._microprocess_trace_state or {}),
                )
                substrate_result = dict(res.get("substrate_result", {}) or {})
                self._microprocess_trace_state = dict(substrate_result.get("trace_state", {}))
                self._last_benchmark = dict(res)
                self._last_microprocess = dict(substrate_result)
                res["opcode"] = str(opcode)
            else:
                res = {
                    "ok": True,
                    "opcode": str(opcode),
                    "hash": len(str(payload))
                }
            self._exec_count += 1
            return res
        except Exception as exc:
            self._last_err = str(exc)
            return {"ok": False, "error": str(exc)}

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            try:
                batch: List[Dict[str, Any]] = []
                with self._lock:
                    if self._queue:
                        take = min(self._batch_size, len(self._queue))
                        batch = self._queue[:take]
                        del self._queue[:take]

                # process batch
                for task in batch:
                    opcode = str(task.get("op", "noop"))
                    payload = dict(task.get("data", {}))
                    _ = self.execute(opcode, payload)

                # telemetry + ADA latency
                now = time.time()
                if (now - self._last_pub) >= self._telemetry_interval_s:
                    self._publish_telemetry()
                    self._last_pub = now

                sh = system_headroom()
                consts = ada_effective_constants({
                    "global_util": float(sh.get("global_util", 0.0)),
                    "gpu_util": float(sh.get("gpu_util", 0.0)),
                    "mem_bw_util": float(sh.get("mem_util", 0.0)),
                    "cpu_util": float(sh.get("cpu_util", 0.0)),
                })
                latency_load = _latency_load(sh)
                self._phase = ada_phase_update(self._phase, float(sh.get("headroom", 0.12)))
                latency = ada_latency_kernel(float(sh.get("headroom", 0.12)), latency_load, self._phase)

                time.sleep(latency)

            except Exception as exc:
                self._last_err = str(exc)

    # ----------------------------------------------------------------------
    # Telemetry
    # ----------------------------------------------------------------------
    def _publish_telemetry(self) -> None:
        snap = self.stats()
        sh = system_headroom()
        consts = ada_effective_constants({
            "global_util": float(sh.get("global_util", 0.0)),
            "gpu_util": float(sh.get("gpu_util", 0.0)),
            "mem_bw_util": float(sh.get("mem_util", 0.0)),
            "cpu_util": float(sh.get("cpu_util", 0.0)),
        })
        snap["ada_constants"] = consts
        snap["headroom"] = float(sh.get("headroom", 0.0))
        snap["phase"] = float(self._phase)
        snap["latency_load"] = float(_latency_load(sh))
        snap["predicted_kernel_latency_s"] = float(
            ada_latency_kernel(float(sh.get("headroom", 0.12)), float(snap["latency_load"]), float(self._phase))
        )
        try:
            self._bus.publish("telemetry.qpu", snap)
        except Exception as e:
            self.log.error("failed publishing telemetry.qpu", exc_info=True)

        try:
            if self._vsd:
                self._vsd.store("telemetry/qpu/current", snap)
        except Exception:
            self.log.error("vsd store error", exc_info=True)

    # ----------------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        try:
            with self._lock:
                qlen = len(self._queue)
            last_result = dict(dict(self._last_microprocess or {}).get("result", {}) or {})
            latency_calibration = dict(last_result.get("latency_calibration", {}) or {})
            return {
                "ts": _utc_ts_str(),
                "running": bool(self._running),
                "queued": int(qlen),
                "exec_total": int(self._exec_count),
                "last_error": self._last_err,
                "batch_size": int(self._batch_size),
                "tick_s": float(self._tick_s),
                "microprocess_trace_gate": float(dict(self._microprocess_trace_state or {}).get("trace_gate", 0.0)),
                "microprocess_trace_alignment": float(dict(self._microprocess_trace_state or {}).get("trace_alignment", 0.0)),
                "last_microprocess_tag": str(last_result.get("result_tag", "")),
                "last_microprocess_route": str(last_result.get("route_case", "")),
                "last_microprocess_steps": int(last_result.get("step_count", 0)),
                "last_microprocess_predicted_latency_s": float(latency_calibration.get("predicted_latency_s", 0.0)),
                "last_microprocess_kernel_request_s": float(latency_calibration.get("kernel_request_s", 0.0)),
                "last_microprocess_pulse_generation_s": float(latency_calibration.get("pulse_generation_s", 0.0)),
                "last_microprocess_actuation_compensation": float(latency_calibration.get("actuation_compensation", 0.0)),
                "last_microprocess_phase_transport_term": float(latency_calibration.get("phase_transport_term", 0.0)),
                "last_microprocess_flux_transport_term": float(latency_calibration.get("flux_transport_term", 0.0)),
                "last_microprocess_reverse_transport_gate": float(latency_calibration.get("reverse_transport_gate", 0.0)),
                "last_microprocess_observer_damping": float(latency_calibration.get("observer_damping", 0.0)),
                "last_benchmark_speedup_total": float(dict(self._last_benchmark or {}).get("speedup_total", 0.0)),
                "last_benchmark_speedup_exec_only": float(dict(self._last_benchmark or {}).get("speedup_exec_only", 0.0)),
                "last_benchmark_results_match": bool(dict(self._last_benchmark or {}).get("results_match", False)),
            }
        except Exception as exc:
            return {"error": str(exc)}

# ============================================================================
# End of file
# ============================================================================
