#!/usr/bin/env python3
# ============================================================================
# Quantum Application - Main Entry Point
# ============================================================================
"""
Main entry point for Quantum Application.
Initializes BIOS, starts all subsystems, and runs the application.
"""

import sys
import os
import logging
import signal
import time

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)sZ | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
# Force UTC timestamps
logging.Formatter.converter = time.gmtime
logger = logging.getLogger('QuantumApplication')

# Import autostart helpers
from miner.miner_engine import register_miner_autostart
from prediction_engine.prediction_engine import register_prediction_autorun

# Global context for shutdown
_app_context = None


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Shutdown signal received. Stopping application...")
    if _app_context:
        shutdown(_app_context)
    sys.exit(0)


def startup():
    """Initialize and start all application subsystems."""
    global _app_context

    logger.info("=" * 70)
    logger.info("Quantum Application Starting...")
    logger.info("=" * 70)

    try:
        # Import BIOS main runtime
        logger.info("Importing BIOS runtime...")
        from bios.main_runtime import run, shutdown as bios_shutdown

        # Load configuration
        logger.info("Loading configuration...")
        config = {
            'mode': 'production',
            'debug': False
        }

        # Start BIOS and all subsystems
        logger.info("Starting BIOS and subsystems...")
        _app_context = run(config)

        # ------------------------------------------------------------------
        # NEW: Wire Miner Engine + Prediction Engine to BIOS boot.complete
        # ------------------------------------------------------------------
        miner_engine = _app_context.get("miner_engine")
        prediction_engine = _app_context.get("prediction_engine")

        if miner_engine:
            logger.info("Wiring MinerEngine to boot.complete...")
            register_miner_autostart(miner_engine)
        else:
            logger.critical("Fatal: miner_engine missing in app context (production mode).")
            raise RuntimeError("miner_engine missing")

        if prediction_engine:
            logger.info("Wiring PredictionEngine to boot.complete...")
            register_prediction_autorun(prediction_engine, interval_s=300.0)
        else:
            logger.critical("Fatal: prediction_engine missing in app context (production mode).")
            raise RuntimeError("prediction_engine missing")
        # ------------------------------------------------------------------

        logger.info("=" * 70)
        logger.info("Quantum Application Started Successfully!")
        logger.info("=" * 70)
        logger.info("Started at: %s", _app_context.get('started_at', 'unknown'))
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 70)

        return _app_context

    except ImportError as e:
        logger.error("Failed to import required modules: %s", e)
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        return None
    except Exception as e:
        logger.error("Failed to start application: %s", e)
        import traceback
        traceback.print_exc()
        return None


def shutdown(context):
    """Gracefully shutdown all subsystems."""
    try:
        from bios.main_runtime import shutdown as bios_shutdown
        logger.info("Shutting down application...")
        bios_shutdown(context)
        logger.info("Application stopped successfully")
    except Exception as e:
        logger.error("Error during shutdown: %s", e)


def main():
    """Main application loop."""
    global _app_context

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start application
    _app_context = startup()

    if not _app_context:
        logger.error("Application failed to start. Exiting.")
        sys.exit(1)

    # Keep application running, but exit if both engines are unhealthy
    try:
        unhealthy_count = 0
        max_unhealthy_checks = 5
        critical_failure = False
        while True:
            miner_engine = _app_context.get("miner_engine")
            prediction_engine = _app_context.get("prediction_engine")
            miner_unhealthy = hasattr(miner_engine, "_error_count") and getattr(miner_engine, "_error_count", 0) >= getattr(miner_engine, "_max_errors", 10)
            prediction_unhealthy = False
            # Check for PredictionEngine autorun thread health by inspecting VSD error
            if hasattr(prediction_engine, "vsd"):
                last_error = prediction_engine.vsd.get("prediction/last_error", {})
                if isinstance(last_error, dict) and last_error.get("error"):
                    unhealthy_count += 1
                else:
                    unhealthy_count = 0
            if miner_unhealthy or unhealthy_count >= max_unhealthy_checks:
                logger.critical("Critical subsystem failure detected. Shutting down application.")
                critical_failure = True
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received")
    finally:
        shutdown(_app_context)
        # Exit non-zero on critical failure path only
        try:
            if critical_failure:
                sys.exit(2)
        except Exception:
            pass


if __name__ == "__main__":
    main()
