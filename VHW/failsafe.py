# ============================================================================
# Virtual Hardware / SAFETY GOVERNOR
# File: VHW/failsafe.py
# Version: v4.8.8 Contextual + Matrix-Aligned
# ============================================================================

import time
import threading
import logging

# ---------------------------------------------------------------------------
# Safe Imports with Fallbacks
# ---------------------------------------------------------------------------

try:
    from bios.event_bus import get_event_bus
except Exception:
    def get_event_bus():
        class _NullBus:
            def publish(self, *_args, **_kwargs):
                pass
        return _NullBus()

try:
    from VHW.vsd_manager import VSDManager
except Exception:
    class VSDManager:
        def get(self, *_args, **_kwargs): return False
        def store(self, *_args, **_kwargs): pass

# ---------------------------------------------------------------------------
# Logger Setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("FailsafeGovernor")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    ))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Contextual Rolling Window
# ---------------------------------------------------------------------------
class ContextualWindow:
    """
    Stores a rolling buffer of frames that triggered alerts.
    Only relevant frames (e.g., high lane utilization) extend the buffer.
    """
    def __init__(self, max_len: int = 200):
        self._buf = []
        self._max_len = max_len
        self._lock = threading.RLock()

    def append_if_alert(self, frame: dict):
        with self._lock:
            if frame.get("alert", False):
                self._buf.append(frame)
                if len(self._buf) > self._max_len:
                    self._buf = self._buf[-self._max_len:]

    def snapshot(self):
        with self._lock:
            return list(self._buf)

# ---------------------------------------------------------------------------
# Class Definition
# ---------------------------------------------------------------------------
class FailsafeGovernor:
    """
    The safety watchdog responsible for detecting unstable lanes and
    deactivating them gracefully. Operates only after BIOS boot flag is set.
    Integrates contextual alert window from telemetry.
    """

    def __init__(self, cfg: dict | None = None, vsd: VSDManager | None = None, bus=None):
        # Config and dependencies
        self.cfg = dict(cfg or {})
        self.vsd = vsd or VSDManager()
        self.bus = bus or get_event_bus()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Configurable parameters (fallback defaults)
        self.cooldown_s = float(self.cfg.get("failsafe.cooldown_s", 3.0))
        self.bias = float(self.cfg.get("failsafe.bias", 0.85))

        # Local state
        self.lane_status = {}
        self._bios_checked = False

        # Contextual alert window
        self.context_window = ContextualWindow(max_len=int(self.cfg.get("failsafe.context_window_len", 200)))

        logger.info("FailsafeGovernor initialized (cooldown=%.2fs, bias=%.2f)",
                    self.cooldown_s, self.bias)

    # -----------------------------------------------------------------------
    # BIOS readiness check
    # -----------------------------------------------------------------------
    def _bios_ready(self):
        # Return True if BIOS has completed boot sequence. #
        try:
            ready = self.vsd.get("system/bios_boot_ok") is True
            if not ready:
                logger.warning("BIOS not ready - failsafe paused.")
            return ready
        except Exception as e:
            logger.error("FailsafeGovernor BIOS readiness check failed: %s", e)
            return False

    # -----------------------------------------------------------------------
    # Core monitoring loop
    # -----------------------------------------------------------------------
    def run(self):
        """Start continuous safety loop once BIOS ready."""
        if not self._bios_ready():
            logger.warning("FailsafeGovernor aborted start: BIOS not ready.")
            return

        logger.info("FailsafeGovernor starting loop.")
        while not self._stop_event.is_set():
            try:
                self._evaluate_lanes()
            except Exception as e:
                logger.exception("Failsafe evaluation error: %s", e)
            time.sleep(self.cooldown_s)
        logger.info("FailsafeGovernor stopped cleanly.")

    # -----------------------------------------------------------------------
    # Lane evaluation logic
    # -----------------------------------------------------------------------
    def _evaluate_lanes(self):
        """Scan all lanes and deactivate unstable ones."""
        with self._lock:
            # Determine unstable lanes based on bias
            unstable = [lane for lane, score in self.lane_status.items()
                        if score < self.bias]

            # Integrate contextual window: mark lanes with historical alerts
            for frame in self.context_window.snapshot():
                lane = frame.get("lane")
                if lane and lane not in unstable:
                    unstable.append(lane)

        if not unstable:
            logger.debug("All lanes stable (%.2f bias).", self.bias)
            return

        for lane in unstable:
            self._deactivate_lane(lane)

    # -----------------------------------------------------------------------
    # Lane deactivation handler
    # -----------------------------------------------------------------------
    def _deactivate_lane(self, lane_id: str):
        """Deactivate a lane and publish failsafe event."""
        try:
            logger.warning("Deactivating unstable lane: %s", lane_id)
            with self._lock:
                self.lane_status[lane_id] = 0.0
            payload = {"lane": lane_id, "ts": time.time()}
            self.bus.publish("failsafe.prune", payload)
            logger.info("Published failsafe.prune for lane %s", lane_id)

            # Also append to contextual window for monitoring
            self.context_window.append_if_alert(payload)
        except Exception as e:
            logger.exception("Failed to deactivate lane %s: %s", lane_id, e)

    # -----------------------------------------------------------------------
    # External health check API (used by MinerEngine)
    # -----------------------------------------------------------------------
    def check_health(self, telemetry: dict | None = None) -> None:
        """
        Update lane health from provided telemetry and apply failsafe rules.

        Expected flexible telemetry formats (all optional):
          - { "lanes": { "laneA": {"score": 0.9, "alert": False}, ... } }
          - { "lane_utilization": { "laneA": 0.92, ... } }  # 0..1
          - { "lane_scores": { "laneA": 0.9, ... } }        # 0..1
          - { "alerts": [ {"lane": "laneA", "level": "crit"}, ... ] }

        Any alert entry will bias the lane below threshold to trigger pruning.
        This method is best-effort and should never raise.
        """
        try:
            if not self._bios_ready():
                return

            telemetry = telemetry or {}

            # Gather candidate lane scores from multiple possible shapes
            lane_updates: dict[str, float] = {}

            lanes = telemetry.get("lanes", {}) or {}
            if isinstance(lanes, dict):
                for lane_id, info in lanes.items():
                    try:
                        score = float(info.get("score",
                                                info.get("util",
                                                         info.get("utilization", 1.0))))
                    except Exception:
                        score = 1.0
                    # clamp 0..1
                    score = 0.0 if score < 0.0 else (1.0 if score > 1.0 else score)
                    # if explicit alert flag, bias below threshold
                    if bool(info.get("alert", False)):
                        score = min(score, max(0.0, self.bias - 0.1))
                    lane_updates[lane_id] = score

            util = telemetry.get("lane_utilization", {}) or {}
            if isinstance(util, dict):
                for lane_id, val in util.items():
                    try:
                        score = float(val)
                    except Exception:
                        score = 1.0
                    score = 0.0 if score < 0.0 else (1.0 if score > 1.0 else score)
                    lane_updates.setdefault(lane_id, score)

            scores = telemetry.get("lane_scores", {}) or {}
            if isinstance(scores, dict):
                for lane_id, val in scores.items():
                    try:
                        score = float(val)
                    except Exception:
                        score = 1.0
                    score = 0.0 if score < 0.0 else (1.0 if score > 1.0 else score)
                    lane_updates.setdefault(lane_id, score)

            alerts = telemetry.get("alerts", []) or []
            if isinstance(alerts, list):
                for entry in alerts:
                    try:
                        lane_id = str(entry.get("lane"))
                    except Exception:
                        lane_id = ""
                    if lane_id:
                        # force below threshold to ensure pruning consideration
                        lane_updates[lane_id] = min(lane_updates.get(lane_id, 1.0), max(0.0, self.bias - 0.2))
                        # record in contextual window
                        self.context_window.append_if_alert({
                            "lane": lane_id,
                            "alert": True,
                            "level": entry.get("level", "warn"),
                            "ts": time.time(),
                        })

            # Apply updates and evaluate
            if lane_updates:
                with self._lock:
                    for lane_id, score in lane_updates.items():
                        self.lane_status[lane_id] = float(score)

            # Evaluate once per call; rely on cooldown inside run() otherwise
            self._evaluate_lanes()

        except Exception as e:
            # Never surface to caller (MinerEngine); log and continue
            logger.error("check_health failed: %s", e, exc_info=True)

    # -----------------------------------------------------------------------
    # Snapshot telemetry for diagnostics
    # -----------------------------------------------------------------------
    def snapshot(self):
        """Return current internal state for telemetry or monitoring."""
        with self._lock:
            return {
                "cooldown_s": self.cooldown_s,
                "bias": self.bias,
                "bios_ready": self._bios_ready(),
                "lanes": dict(self.lane_status),
                "context_window_len": len(self.context_window.snapshot()),
                "timestamp": time.time(),
            }

    # -----------------------------------------------------------------------
    # Stop control
    # -----------------------------------------------------------------------
    def stop(self):
        """Signal failsafe loop to halt."""
        self._stop_event.set()
        logger.info("FailsafeGovernor stop signal issued.")

# ---------------------------------------------------------------------------
# End of Module
# ---------------------------------------------------------------------------
