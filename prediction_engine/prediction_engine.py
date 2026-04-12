# ============================================================================
# Quantum Application / prediction_engine
# ASCII-ONLY SOURCE FILE
# File: prediction_engine.py
# Version: v1.4 Persistent-Aware (StateManager + Health Telemetry + Boot-Autorun)
# ============================================================================
"""
Purpose
-------
Live prediction orchestrator:
  1) discovers assets (liquidity-ranked) from Crypto.com,
  2) fetches market data,
  3) runs clustered predictions,
  4) fuses results,
  5) asks MurphyWatchdog to approve trade,
  6) executes live orders via TradeExecutor,
  7) logs signals and trades to VSD,
  8) persists model and trade state via StateManager.

Notes
-----
- ASCII-only. No Unicode.
- Runs once per call (no background thread here) unless wired with
  register_prediction_autorun(...) which creates a background loop
  after BIOS 'boot.complete'.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import time
import logging
import threading

from .crypto_com_api import CryptoComAPI
from prediction_engine.model_core import ModelCore
from prediction_engine.predictive_cluster import PredictiveClusterManager
from prediction_engine.ensemble_fusion import EnsembleFusion
from prediction_engine.trade_executor import TradeExecutor
from bios.state_manager import StateManager

# BIOS EventBus (optional, for boot wiring)
try:
    from bios.event_bus import get_event_bus  # type: ignore
except Exception as exc:  # noqa: BLE001
    # We log lazily once logger is defined.
    def get_event_bus():  # type: ignore
        return None
    _BOOT_BUS_IMPORT_ERROR = str(exc)
else:
    _BOOT_BUS_IMPORT_ERROR = ""

BOOT_COMPLETE_TOPIC = "boot.complete"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _init_logger() -> logging.Logger:
    logger = logging.getLogger("prediction.engine")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = _init_logger()

if _BOOT_BUS_IMPORT_ERROR:
    logger.warning("bios.event_bus unavailable; prediction autostart wiring disabled: %s",
                   _BOOT_BUS_IMPORT_ERROR)

# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
class _MapVSD:
    def __init__(self) -> None:
        self._m: Dict[str, Any] = {}
    def get(self, key: str, default: Any = None) -> Any:
        return self._m.get(str(key), default)
    def store(self, key: str, value: Any) -> None:
        self._m[str(key)] = value

def _utc_now_str() -> str:
    import time as _t
    return _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())

# ============================================================================
# PredictionEngine
# ============================================================================
class PredictionEngine:
    """
    Trade-aware, persistent-aware prediction orchestrator.
    Exposes:
      - initialize()
      - run_once()
      - run_cycle()  (alias of run_once)
    """

    def __init__(self,
                 watchdog: Any,
                 vsd: Optional[Any] = None,
                 base_quote: str = "USDT",
                 max_assets: int = 12,
                 lanes_per_cluster: int = 6,
                 max_clusters: int = 4,
                 min_confidence: float = 0.99,
                 candle_timeframe: str = "1h",
                 candle_limit: int = 400):
        self.watchdog = watchdog
        self.vsd = vsd if vsd is not None else _MapVSD()
        self.base_quote = str(base_quote).upper()
        self.max_assets = int(max_assets)
        self.lanes_per_cluster = int(lanes_per_cluster)
        self.max_clusters = int(max_clusters)
        self.min_confidence = float(min_confidence)
        self.candle_timeframe = candle_timeframe
        self.candle_limit = int(candle_limit)

        # Core components
        self.api = CryptoComAPI()
        self.model_core = ModelCore()
        self.cluster_mgr = PredictiveClusterManager(
            max_clusters=self.max_clusters,
            lanes_per_cluster=self.lanes_per_cluster
        )
        self.fuser = EnsembleFusion(weight_cap=0.99)
        self.trader = TradeExecutor(min_confidence=self.min_confidence)
        self.state_mgr = StateManager(self.vsd)

    # ---------------------------------------------------------------------
    # Initialization and connectivity checks
    # ---------------------------------------------------------------------
    def initialize(self) -> None:
        """Verify exchange connectivity and record boot status."""
        ok = False
        try:
            ok = self.api.verify_connection()
        except Exception as exc:  # noqa: BLE001
            logger.error("PredictionEngine.initialize verify_connection failed: %s",
                         exc, exc_info=True)
            ok = False
        self.vsd.store("prediction/boot", {
            "ts": _utc_now_str(),
            "api_ok": bool(ok),
            "base_quote": self.base_quote
        })

    # ---------------------------------------------------------------------
    # Asset discovery
    # ---------------------------------------------------------------------
    def discover_assets(self) -> List[str]:
        """Return liquidity-ranked list of instruments, capped by max_assets."""
        try:
            ranked = self.api.get_top_liquid_assets(self.base_quote, top_n=self.max_assets)
            if isinstance(ranked, list):
                out = [r for r in ranked if isinstance(r, str)]
                if out:
                    self.vsd.store("prediction/assets/last", {
                        "ts": _utc_now_str(),
                        "assets": out
                    })
                    return out
        except Exception as exc:  # noqa: BLE001
            logger.error("discover_assets failed: %s", exc, exc_info=True)
        return []

    # ---------------------------------------------------------------------
    # Market data get
    # ---------------------------------------------------------------------
    def fetch_market_data(self, symbols: List[str]) -> Dict[str, List[Dict[str, float]]]:
        """Fetch OHLCV arrays for each symbol."""
        out: Dict[str, List[Dict[str, float]]] = {}
        for sym in symbols:
            try:
                candles = self.api.get_candles(sym, timeframe=self.candle_timeframe,
                                               limit=self.candle_limit)
                if isinstance(candles, list) and candles:
                    out[sym] = candles
            except Exception as exc:  # noqa: BLE001
                logger.error("fetch_market_data failed for %s: %s", sym, exc, exc_info=True)
                continue
        self.vsd.store("prediction/market_data/last",
                       {"ts": _utc_now_str(), "symbols": list(out.keys())})
        return out

    # ---------------------------------------------------------------------
    # Prediction and fusion
    # ---------------------------------------------------------------------
    def _predict_and_fuse(self,
                          symbols: List[str],
                          market: Dict[str, List[Dict[str, float]]]) -> List[Dict[str, Any]]:
        """Allocate clusters, run model lanes, and fuse results."""
        if not symbols:
            return []
        self.cluster_mgr.allocate_clusters(symbols)
        cluster_results = self.cluster_mgr.run_all(market)
        fused = self.fuser.combine(cluster_results)
        self.vsd.store("prediction/signals/fused",
                       {"ts": _utc_now_str(), "count": len(fused)})
        return fused

    # ---------------------------------------------------------------------
    # Trade gate and execution
    # ---------------------------------------------------------------------
    def _maybe_trade(self, fused_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ask MurphyWatchdog to approve trades and execute via TradeExecutor."""
        orders: List[Dict[str, Any]] = []
        for sig in fused_results:
            try:
                conf = float(sig.get("avg_confidence", 0.0))
                if conf < self.min_confidence:
                    orders.append({"ok": False, "reason": "low_confidence", "signal": sig})
                    continue
                if not getattr(self.watchdog, "approve_trade", lambda x: False)(sig):
                    orders.append({"ok": False, "reason": "watchdog_block", "signal": sig})
                    continue

                result = self.trader.execute(sig)
                orders.append(result)

                # Log and persist trade
                try:
                    if hasattr(self.trader, "log_to_vsd"):
                        self.trader.log_to_vsd(self.vsd, result)
                except Exception as exc:  # noqa: BLE001
                    logger.error("TradeExecutor.log_to_vsd failed: %s", exc, exc_info=True)
                try:
                    self.state_mgr.append_trade_log(result)
                except Exception as exc:  # noqa: BLE001
                    logger.error("append_trade_log failed: %s", exc, exc_info=True)

            except Exception as exc:  # noqa: BLE001
                logger.error("PredictionEngine._maybe_trade failure: %s", exc, exc_info=True)
                orders.append({"ok": False,
                               "reason": "exception",
                               "error": str(exc),
                               "signal": sig})

        # Save truncated order history
        try:
            hist = self.vsd.get("trade/orders/history", []) or []
            hist.extend(orders)
            self.vsd.store("trade/orders/history", hist[-500:])
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to archive prediction orders: %s", exc, exc_info=True)
        return orders

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run_once(self) -> Dict[str, Any]:
        """Execute a single end-to-end prediction and optional trade pass."""
        try:
            assets = self.discover_assets()
            market = self.fetch_market_data(assets)
            fused = self._predict_and_fuse(assets, market)
            orders = self._maybe_trade(fused)

            # Expose latest fused signals
            try:
                self.vsd.store("telemetry/predictions/latest", fused)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to save predictions to VSD: %s", exc, exc_info=True)

            # Persist state and heartbeat
            try:
                self.state_mgr.save_models(self.model_core)
                self.vsd.store("prediction/health", {
                    "ts": _utc_now_str(),
                    "assets": len(assets),
                    "signals": len(fused),
                    "orders": len(orders)
                })
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to persist prediction state: %s", exc, exc_info=True)

            report = {
                "ts": _utc_now_str(),
                "assets_considered": len(assets),
                "signals": len(fused),
                "orders": len(orders),
            }
            self.vsd.store("prediction/last_report", report)
            return report
        except Exception as exc:  # noqa: BLE001
            logger.error("PredictionEngine.run_once failure: %s", exc, exc_info=True)
            err = {"ts": _utc_now_str(), "error": str(exc)}
            self.vsd.store("prediction/last_error", err)
            return err

    def run_cycle(self) -> Dict[str, Any]:
        """Backwards-compatible alias for run_once()."""
        return self.run_once()


# ---------------------------------------------------------------------------
# Boot wiring helper
# ---------------------------------------------------------------------------
def register_prediction_autorun(engine: PredictionEngine,
                                interval_s: float = 60.0) -> None:
    """
    Subscribe the given PredictionEngine instance to BIOS 'boot.complete' so that
    it starts a background loop calling run_once() every interval_s seconds.

    This does NOT change run_once; it only adds an optional background loop.

    Usage (e.g. in BIOS or run_app wiring):

        pe = PredictionEngine(watchdog=MurphyWatchdog(), vsd=vsd)
        register_prediction_autorun(pe, interval_s=300.0)
    """
    bus = get_event_bus()
    if bus is None:
        logger.warning("EventBus not available; PredictionEngine autorun disabled")
        return

    def _loop() -> None:
        logger.info("PredictionEngine autorun loop started (interval_s=%.3f)", interval_s)
        error_count = 0
        max_errors = 10
        while True:
            try:
                result = engine.run_once()
                # If run_once returns an error dict, increment error_count
                if isinstance(result, dict) and "error" in result:
                    error_count += 1
                    logger.error("PredictionEngine run_once error (%d/%d): %s", error_count, max_errors, result.get("error"))
                else:
                    error_count = 0  # Reset on success
            except Exception as exc:  # noqa: BLE001
                error_count += 1
                logger.error("PredictionEngine autorun run_once failed (%d/%d): %s", error_count, max_errors, exc, exc_info=True)
            if error_count >= max_errors:
                logger.critical("PredictionEngine encountered too many errors (%d). Stopping autorun loop.", error_count)
                break
            time.sleep(interval_s)

    # PATCHED: handler must accept one argument from EventBus
    def _on_boot_complete(event: Dict[str, Any] | None = None) -> None:
        logger.info("Received boot.complete; starting PredictionEngine autorun")
        try:
            t = threading.Thread(
                target=_loop,
                daemon=True,
                name="prediction_engine_autorun"
            )
            t.start()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to start PredictionEngine autorun thread: %s",
                exc,
                exc_info=True
            )

    try:
        # PATCHED: removed unsupported "name=" argument
        bus.subscribe(
            BOOT_COMPLETE_TOPIC,
            _on_boot_complete,
            once=True,
            priority=0
        )
        logger.info("PredictionEngine autorun wired to '%s'", BOOT_COMPLETE_TOPIC)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to subscribe PredictionEngine to '%s': %s",
            BOOT_COMPLETE_TOPIC,
            exc,
            exc_info=True
        )


# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[PredictionEngine] Self-test starting...")
    dummy_vsd = _MapVSD()

    class DummyWD:
        def approve_trade(self, signal: Dict[str, Any]) -> bool:
            return False

    pe = PredictionEngine(watchdog=DummyWD(), vsd=dummy_vsd, max_assets=4)
    pe.initialize()
    report = pe.run_once()
    print("Report:", report)
