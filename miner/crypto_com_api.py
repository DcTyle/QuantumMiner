# ASCII-ONLY SOURCE FILE
# Path: miner/crypto_com_api.py
# Purpose: Integration wrapper that re-exports the Crypto.com API client
#          from prediction_engine. Centralizes the import path
#          (e.g., bios.scheduler) without duplicating logic.

from __future__ import annotations

# Strict import: no fallback stubs permitted
from prediction_engine.crypto_com_api import CryptoComAPI  # type: ignore

__all__ = ["CryptoComAPI"]
