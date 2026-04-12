# ============================================================================
# Quantum Application / Control Center
# ASCII-ONLY SOURCE FILE
# File: Control_Center/control_center_app.py
# Version: v1.0 TUI App
# ----------------------------------------------------------------------------
# Purpose
# -------
# ASCII-only curses-based Control Center UI. Provides a collapsible sidebar,
# status bar with mic toggle, and three main panels: Miner, Prediction,
# Neuralis. Panels read from VSD and subscribe to EventBus updates.
#
# Usage
# -----
#   python -m Control_Center.control_center_app
# ============================================================================

from __future__ import annotations
from typing import Dict, Any
import time

try:
    import curses
except Exception:  # pragma: no cover - runtime-only
    curses = None  # type: ignore

from .theme import Theme
from .sidebar import Sidebar
from .ui_layout import UiLayout
from .panel_miner import MinerPanel
from .panel_prediction import PredictionPanel
from .panel_neuralis import NeuralisPanel
from .panel_system import SystemPanel
from .panel_ada import AdaPanel
from .panel_settings import SettingsPanel
from .command_registry import CommandRegistry, Command


def _get_bus_and_vsd() -> tuple[Any, Any]:
    """Resolve EventBus and a shared VSD instance.

    Order:
      1) BIOS EventBus via bios.event_bus.get_event_bus()
      2) Shared VSD via bios.main_runtime._vsd
      3) Fallback to VHW.vsd_manager.VSDManager()
      4) Final fallback to in-memory map with get/store
    """
    # EventBus
    try:
        from bios.event_bus import get_event_bus as _get_bios_bus  # type: ignore
        bus = _get_bios_bus()
    except Exception:
        bus = None

    # Try shared VSD from BIOS
    try:
        from bios.main_runtime import _vsd as shared_vsd  # type: ignore
        vsd = shared_vsd
        return (bus, vsd)
    except Exception:
        pass

    # Try constructing a VSDManager
    try:
        from VHW.vsd_manager import VSDManager  # type: ignore
        vsd = VSDManager()
        return (bus, vsd)
    except Exception:
        pass

    # In-memory minimal VSD
    class _MapVSD:
        def __init__(self) -> None:
            self._m: Dict[str, Any] = {}
        def get(self, key: str, default: Any = None) -> Any:
            return self._m.get(str(key), default)
        def store(self, key: str, value: Any) -> None:
            self._m[str(key)] = value

    return (bus, _MapVSD())


class ControlCenterApp:
    def __init__(self) -> None:
        self.bus, self.vsd = _get_bus_and_vsd()
        self.sidebar = Sidebar()
        self.layout = None
        self.panels: Dict[str, Any] = {}
        self.active_key = "miner"
        self._last_note = ""
        self._voice_ts = 0.0
        self._input_buf: str = ""
        self._shutdown_requested: bool = False
        # Try to start Neuralis CommandRouter so voice input is handled
        try:
            from Neuralis_AI.ai_core import start_command_router  # type: ignore
            start_command_router()
        except Exception:
            pass

        # Respect persisted autostart flags and trigger miner/prediction
        # startup via Control Center commands instead of direct imports.
        try:
            auto_miner = bool(self.vsd.get("control_center/autostart/miner", False))
            auto_pred = bool(self.vsd.get("control_center/autostart/prediction", False))
            if auto_miner and self.bus is not None:
                self.bus.publish("control_center.ui.action", {"type": "start_miner"})
            if auto_pred and self.bus is not None:
                self.bus.publish("control_center.ui.action", {"type": "start_prediction"})
        except Exception:
            pass

    def _init_panels(self) -> None:
        self.panels = {
            # Miner subsystem
            "miner": MinerPanel(self.vsd),
            # PredictionEngine subsystem
            "prediction": PredictionPanel(self.vsd),
            # Neuralis subsystem (state + commands reuse)
            "neuralis": NeuralisPanel(self.vsd, self.bus),
            "commands": NeuralisPanel(self.vsd, self.bus),
            # VHW / ADA telemetry
            "ada": AdaPanel(self.vsd),
            # System + settings
            "system": SystemPanel(self.vsd),
            "settings": SettingsPanel(self.vsd, self._on_setting_changed),
        }
        # Note: Commands panel reuses NeuralisPanel renderer for chat/commands log tail

    def _subscribe_events(self) -> None:
        # Subscribe to VSD write notifications to hint quick refresh
        try:
            if self.bus is None:
                return
            def _on_vsd_write(_payload: Dict[str, Any]) -> None:
                # Lightweight hint; actual render loop will pick up data
                self._last_note = "VSD updated"
            self.bus.subscribe("vsd.write", _on_vsd_write, priority=0)

            # UI routing subscriptions
            def _route(topic: str, panel_key: str):
                def _handler(payload: Dict[str, Any]) -> None:
                    pnl = self.panels.get(panel_key)
                    if pnl and hasattr(pnl, "on_message"):
                        try:
                            pnl.on_message(topic, payload)
                        except Exception:
                            pass
                return _handler

            self.bus.subscribe("control_center.telemetry.miner", _route("control_center.telemetry.miner", "miner"))
            self.bus.subscribe("control_center.telemetry.prediction", _route("control_center.telemetry.prediction", "prediction"))
            self.bus.subscribe("control_center.telemetry.neuralis", _route("control_center.telemetry.neuralis", "neuralis"))
            self.bus.subscribe("control_center.telemetry.system", _route("control_center.telemetry.system", "system"))

            # React to sidebar panel switch events
            def _on_panel_switch(payload: Dict[str, Any]) -> None:
                key = str((payload or {}).get("panel", ""))
                if key:
                    self.active_key = key
                    if self.layout:
                        self.layout.show_panel(key)
            self.bus.subscribe("panel.switch", _on_panel_switch, priority=0)

            # UI actions from Neuralis CommandRouter or others
            def _on_ui_action(payload: Dict[str, Any]) -> None:
                try:
                    t = str((payload or {}).get("type", ""))
                    if t == "switch_panel":
                        p = str((payload or {}).get("panel", ""))
                        if p:
                            self.active_key = p
                            if self.layout:
                                self.layout.show_panel(p)
                    elif t == "sidebar":
                        mode = str((payload or {}).get("mode", "toggle"))
                        if mode == "toggle":
                            self.sidebar.toggle()
                        elif mode == "collapse":
                            self.sidebar.collapsed = True
                        elif mode == "expand":
                            self.sidebar.collapsed = False
                        if self.layout:
                            self.layout.resize()
                    elif t == "clear_log":
                        # clear specific panel or current
                        target = str((payload or {}).get("panel", ""))
                        k = target if target in self.panels else self.active_key
                        pnl = self.panels.get(k)
                        if pnl and hasattr(pnl, "clear_log"):
                            pnl.clear_log()  # type: ignore
                    elif t == "toggle_mic":
                        self.layout.mic_enabled = not self.layout.mic_enabled
                        self._voice_ts = time.time()
                        if self.bus is not None:
                            self.bus.publish("neuralis.voice_mode", {
                                "ts": self._voice_ts,
                                "enabled": self.layout.mic_enabled
                            })
                        if not self.layout.mic_enabled:
                            self._input_buf = ""
                    elif t == "shutdown":
                        self._shutdown_requested = True
                    # Notify if message provided
                    note = str((payload or {}).get("note", ""))
                    if note and self.layout:
                        self.layout.notify(note)
                except Exception:
                    pass
            self.bus.subscribe("control_center.ui.action", _on_ui_action, priority=0)

            # Voice/text to command fallback: if Neuralis Router is unavailable,
            # handle plain text via CommandRegistry.
            try:
                self._cmdreg = CommandRegistry(self.bus, self.vsd)  # type: ignore[attr-defined]
            except Exception:
                self._cmdreg = None  # type: ignore[attr-defined]

            def _on_voice_input(payload: Dict[str, Any]) -> None:
                try:
                    if self._cmdreg is None:
                        return
                    cmd_name = str((payload or {}).get("command_id", "")).strip()
                    if not cmd_name:
                        return
                    try:
                        cmd = Command[cmd_name]
                    except KeyError:
                        return
                    args = dict((payload or {}).get("args", {}))
                    self._cmdreg.run_enum(cmd, args)
                    # Route output to Commands panel visually
                    self.active_key = "commands"
                    if self.layout:
                        self.layout.show_panel("commands")
                except Exception:
                    pass
            self.bus.subscribe("neuralis.voice.input", _on_voice_input, priority=0)
        except Exception:
            pass

    # Settings callback from SettingsPanel
    def _on_setting_changed(self, key: str, value: Any) -> None:  # type: ignore[override]
        try:
            if key == "theme":
                from .theme import Theme
                Theme.set_mode("light" if value == "light" else "dark")
                Theme.init_colors()
                if self.layout:
                    self.layout.notify("Theme: %s" % value)
            elif key == "timestamp_precision":
                if self.layout:
                    self.layout.notify("TS precision: %s" % ("high" if value else "low"))
            elif key == "verbosity":
                if self.layout:
                    self.layout.notify("Verbosity: %s" % str(value))
            elif key == "autostart:miner":
                # Persist to VSD so next app boot can auto-start miner.
                try:
                    self.vsd.store("control_center/autostart/miner", bool(value))
                except Exception:
                    pass
                if self.layout:
                    self.layout.notify("Miner autostart: %s" % ("ON" if value else "OFF"))
            elif key == "autostart:prediction":
                try:
                    self.vsd.store("control_center/autostart/prediction", bool(value))
                except Exception:
                    pass
                if self.layout:
                    self.layout.notify("Prediction autostart: %s" % ("ON" if value else "OFF"))
            elif key.startswith("autoscroll:"):
                if self.layout:
                    panel = key.split(":", 1)[1]
                    self.layout._auto_scroll[panel] = bool(value)
                    self.layout.notify("Autoscroll %s: %s" % (panel, "on" if value else "off"))
        except Exception:
            pass

    def run(self, stdscr) -> None:
        if curses is None:
            return
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        Theme.init_colors()

        self.layout = UiLayout(stdscr, self.sidebar)
        self._init_panels()
        self._subscribe_events()

        last_resize = (0, 0)
        last_render = 0.0
        try:
            while True:
                if self._shutdown_requested:
                    break
                # Handle resize
                h, w = stdscr.getmaxyx()
                if (h, w) != last_resize:
                    last_resize = (h, w)
                    self.layout.resize()

                # Key handling (non-blocking)
                try:
                    ch = stdscr.getch()
                except Exception:
                    ch = -1
                if ch != -1:
                    # Hotkeys map (Ctrl+N fallback to digits/letters)
                    try:
                        import curses.ascii as _ca  # type: ignore
                        CTRL_L = _ca.ctrl('l')
                        CTRL_S = _ca.ctrl('s')
                        CTRL_M = _ca.ctrl('m')
                        CTRL_Q = _ca.ctrl('q')
                    except Exception:
                        CTRL_L = CTRL_S = CTRL_M = CTRL_Q = -9999  # type: ignore

                    if ch in (ord('q'), ord('Q'), CTRL_Q):
                        break
                    # Sidebar collapse/expand
                    elif ch in (ord('s'), ord('S'), CTRL_S):
                        self.sidebar.toggle()
                        self.layout.resize()
                    # Log clear: delegate to active panel
                    elif ch in (ord('l'), ord('L'), CTRL_L):
                        panel = self.panels.get(self.active_key)
                        if panel and hasattr(panel, "clear_log"):
                            try:
                                panel.clear_log()
                            except Exception:
                                pass
                    # Voice mode toggle
                    elif ch in (ord('m'), ord('M'), CTRL_M):
                        self.layout.mic_enabled = not self.layout.mic_enabled
                        self._voice_ts = time.time()
                        try:
                            if self.bus is not None:
                                self.bus.publish("neuralis.voice_mode", {
                                    "ts": self._voice_ts,
                                    "enabled": self.layout.mic_enabled
                                })
                        except Exception:
                            pass
                        if self.layout:
                            self.layout.notify("Voice: %s" % ("ON" if self.layout.mic_enabled else "OFF"))
                    # Navigate sidebar
                    elif ch in (curses.KEY_DOWN, ord('j')):
                        self.sidebar.next()
                        self.active_key = self.sidebar.current_key()
                        self.sidebar.click(self.bus)
                    elif ch in (curses.KEY_UP, ord('k')):
                        self.sidebar.prev()
                        self.active_key = self.sidebar.current_key()
                        self.sidebar.click(self.bus)
                    # Activate selected sidebar entry
                    elif ch in (curses.KEY_ENTER, 10, 13):
                        self.sidebar.click(self.bus)
                        self.active_key = self.sidebar.current_key()
                    # Numeric hotkeys (Ctrl+1..5 fallback to '1'..'5')
                    elif ch in (ord('1'),):
                        self.active_key = "miner"
                    elif ch in (ord('2'),):
                        self.active_key = "prediction"
                    elif ch in (ord('3'),):
                        self.active_key = "neuralis"
                    elif ch in (ord('4'),):
                        self.active_key = "system"
                    elif ch in (ord('5'),):
                        self.active_key = "settings"
                    # Pass other keys to active panel for per-panel actions
                    else:
                        # Voice input capture when mic enabled
                        if self.layout.mic_enabled:
                            try:
                                if ch in (curses.KEY_ENTER, 10, 13):
                                    text = self._input_buf.strip()
                                    if text:
                                        if self.bus is not None:
                                            self.bus.publish("neuralis.voice.input", {
                                                "ts": time.time(),
                                                "text": text
                                            })
                                        # local note and clear
                                        if self.layout:
                                            self.layout.notify("You: " + text)
                                        self._input_buf = ""
                                elif ch in (curses.KEY_BACKSPACE, 127):
                                    self._input_buf = self._input_buf[:-1]
                                elif 32 <= ch <= 126:
                                    self._input_buf += chr(ch)
                            except Exception:
                                pass
                        else:
                            panel = self.panels.get(self.active_key)
                            if panel and hasattr(panel, "handle_key"):
                                try:
                                    panel.handle_key(ch)
                                except Exception:
                                    pass

                # Ensure layout selected panel reflects active_key
                if self.layout and self.layout.active_tab != self.active_key:
                    self.layout.show_panel(self.active_key)

                # Periodic render
                now = time.time()
                if now - last_render >= 0.1:
                    # Pass input buffer for status rendering
                    self.layout.render_status(self._last_note, input_buf=self._input_buf)
                    self.layout.render_sidebar()
                    panel = self.panels.get(self.active_key) or self.panels.get("miner")
                    # Allow panels to do fast refresh probes
                    if hasattr(panel, "refresh_fast"):
                        try:
                            panel.refresh_fast()  # type: ignore
                        except Exception:
                            pass
                    self.layout.render_main(panel)
                    self._last_note = ""
                    last_render = now

                time.sleep(0.01)
        except KeyboardInterrupt:
            pass


def main() -> None:
    if curses is None:
        print("curses not available")
        return
    app = ControlCenterApp()
    curses.wrapper(app.run)


if __name__ == "__main__":
    main()
