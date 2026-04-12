# ============================================================================
# Quantum Application / config
# ASCII-ONLY SOURCE FILE
# File: config/manager.py
# Version: v1.0 Minimal
# ----------------------------------------------------------------------------
"""
Minimal ConfigManager for environments without external config package.
Provides class-level get/set with in-memory storage and sane defaults.
"""
from __future__ import annotations
from typing import Any, Dict
import json
import os
from pathlib import Path


class ConfigManager:
    _store: Dict[str, Any] = {}
    _loaded: bool = False

    @classmethod
    def _settings_path(cls) -> Path:
        override = str(os.environ.get("QM_USER_SETTINGS_PATH", "") or "").strip()
        if override:
            return Path(override)
        return Path(__file__).resolve().with_name("user_settings.json")

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        path = cls._settings_path()
        try:
            if not path.exists():
                cls._store = {}
                return
            raw = json.loads(path.read_text(encoding="ascii"))
            cls._store = dict(raw) if isinstance(raw, dict) else {}
        except Exception:
            cls._store = {}

    @classmethod
    def _save(cls) -> None:
        path = cls._settings_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cls._store, indent=2, sort_keys=True), encoding="ascii")
        except Exception:
            pass

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        cls._ensure_loaded()
        try:
            return cls._store.get(str(key), default)
        except Exception:
            return default

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        cls._ensure_loaded()
        try:
            cls._store[str(key)] = value
            cls._save()
        except Exception:
            pass

    @classmethod
    def path(cls) -> str:
        return str(cls._settings_path())

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        cls._ensure_loaded()
        return dict(cls._store)
