# ============================================================================
# Quantum Application / control_center
# File: control_center/__init__.py
# Version: v1.0 "Neuralis Auto-Boot"
# ASCII-ONLY SOURCE FILE
# ============================================================================

from __future__ import annotations
import time
import threading
try:
    import requests  # optional dependency
except Exception:
    requests = None
from typing import Dict, Any, Optional

# ----------------------------------------------------------------------------
# VSD
# ----------------------------------------------------------------------------
try:
    from VHW.vsd_manager import VSDManager
except Exception:
    class VSDManager:
        def __init__(self):
            self._m = {}
        def get(self, k, d=None): return self._m.get(k, d)
        def store(self, k, v): self._m[k] = v

vsd = VSDManager()

# ----------------------------------------------------------------------------
# Boot flags
# ----------------------------------------------------------------------------
_prediction_running = False
_neuralis_running = False
_miner_running = False

# ----------------------------------------------------------------------------
# Placeholder for user API key (user inserts theirs)
# ----------------------------------------------------------------------------
WEATHER_API_KEY = "INSERT_YOUR_API_KEY_HERE"

# ----------------------------------------------------------------------------
# Fetch time + weather
# ----------------------------------------------------------------------------
def _get_local_time() -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    except Exception:
        return "Unknown Time"

def _get_weather() -> str:
    try:
        # Example uses OpenWeather (user inserts API key)
        url = ("https://api.openweathermap.org/data/2.5/weather"
               "?q=Anchorage&appid=" + WEATHER_API_KEY + "&units=imperial")
        if requests is None:
            raise RuntimeError("requests not available")
        r = requests.get(url, timeout=3)
        data = r.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"{temp} F, {desc}"
    except Exception:
        return "Weather unavailable"

# ----------------------------------------------------------------------------
# Auto-Start Modules (except Miner)
# ----------------------------------------------------------------------------
def _auto_start_prediction_engine() -> None:
    global _prediction_running
    if _prediction_running:
        return
    _prediction_running = True
    print("[ControlCenter] Prediction Engine auto-start complete.")
    vsd.store("system/prediction_running", True)

def _auto_start_neuralis() -> None:
    global _neuralis_running
    if _neuralis_running:
        return
    _neuralis_running = True
    print("[ControlCenter] Neuralis AI auto-start complete.")
    vsd.store("system/neuralis_running", True)

# ----------------------------------------------------------------------------
# Manual-Start Button: Miner
# ----------------------------------------------------------------------------
def start_miner_button() -> bool:
    global _miner_running
    if _miner_running:
        print("[ControlCenter] Miner already running.")
        return True
    _miner_running = True
    print("[ControlCenter] Miner started manually.")
    vsd.store("system/miner_running", True)
    return True

# ----------------------------------------------------------------------------
# GUI calls these for auto-start
# ----------------------------------------------------------------------------
def start_prediction_engine_button() -> bool:
    _auto_start_prediction_engine()
    return True

def start_neuralis_button() -> bool:
    _auto_start_neuralis()
    return True

# ----------------------------------------------------------------------------
# Greeting logic (runs on module import)
# ----------------------------------------------------------------------------
def _boot_greeting() -> None:
    local_time = _get_local_time()
    weather = _get_weather()
    last_task = vsd.get("user/last_task", "none")

    print("\n======================================================")
    print("              Neuralis Control Center Boot")
    print("======================================================")
    print(f"Hello Dax. It is currently: {local_time}")
    print(f"The weather outside is: {weather}")

    if last_task != "none":
        print(f"\nLast time, you were working on: {last_task}")
        print("Would you like to continue where you left off?\n")
    else:
        print("\nNo previous task recorded. Ready when you are.\n")

# ----------------------------------------------------------------------------
# Optional legacy placeholder auto-runs
# ----------------------------------------------------------------------------
def _startup_thread():
    # give system a moment for BIOS + VHW to initialize
    time.sleep(1.0)
    _boot_greeting()
    _auto_start_prediction_engine()
    _auto_start_neuralis()


if str(__import__("os").environ.get("CONTROL_CENTER_LEGACY_BOOT_SHIM", "0") or "0").strip() == "1":
    t = threading.Thread(target=_startup_thread, daemon=True)
    t.start()

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
__all__ = [
    "start_miner_button",
    "start_prediction_engine_button",
    "start_neuralis_button",
]
