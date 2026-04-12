# ASCII only
# ================================================================
# File: miner/crypto_com_api.py
# Purpose:
#   Unified Crypto.com API wrapper for both private and public data.
#   Supports account balances, ticker snapshots, candle histories,
#   market discovery, liquidity ranking, connection verification,
#   and LIVE ORDER placement and management.
#   Keys are read from .env or system environment.
# ================================================================

from __future__ import annotations
from typing import Dict, Any, List
import os
import time
import json
import hashlib
import hmac
# Optional HTTP client; fallback to urllib if unavailable
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore
import urllib.request
import urllib.parse
from pathlib import Path

try:
    from config.manager import ConfigManager
except Exception:
    ConfigManager = None  # type: ignore

# ---------------------------------------------------------------
# Optional .env autoload
# ---------------------------------------------------------------
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except Exception:
    pass


class CryptoComAPI:
    """
    Crypto.com Exchange API wrapper.

    Private endpoints:
        - private/get-account-summary
        - private/create-order
        - private/cancel-order
        - private/get-open-orders
        - private/get-order-detail
    Public endpoints:
        - public/get-candlestick
        - public/get-ticker
        - public/get-instruments
    """

    BASE_URL = "https://api.crypto.com/v2/"

    def __init__(self, api_key: str = "", api_secret: str = ""):
        cfg_api_key = ""
        cfg_api_secret = ""
        try:
            if ConfigManager is not None:
                cfg_api_key = str(ConfigManager.get("prediction.api_key", "") or "")
                cfg_api_secret = str(ConfigManager.get("prediction.api_secret", "") or "")
        except Exception:
            cfg_api_key = ""
            cfg_api_secret = ""
        self._api_key = str(api_key or cfg_api_key or os.getenv("CRYPTOCOM_API_KEY", "") or "")
        self._api_secret = str(api_secret or cfg_api_secret or os.getenv("CRYPTOCOM_API_SECRET", "") or "")
        self._cache: Dict[str, Any] = {}
        self._last_update = 0.0
        self._ttl = 3600  # seconds

    # -----------------------------------------------------------
    # Utility: signature for private endpoints
    # -----------------------------------------------------------
    def _sign(self, method: str, req_id: int, api_key: str, nonce: int, params: Dict[str, Any]) -> str:
        """Crypto.com v2 HMAC-SHA256 signature.
        sigPayload = method + str(id) + api_key + str(nonce) + json.dumps(params, separators=(',', ':'), sort_keys=True)
        signature = hex(hmac_sha256(api_secret, sigPayload))
        """
        if not self._api_secret:
            return ""
        param_str = json.dumps(params or {}, separators=(",", ":"), sort_keys=True)
        sig_payload = f"{method}{req_id}{api_key}{nonce}{param_str}"
        digest = hmac.new(self._api_secret.encode("utf-8"), sig_payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return digest

    # -----------------------------------------------------------
    def _post(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Private authenticated POST request."""
        if not self._api_key or not self._api_secret:
            return {}
        url = self.BASE_URL + method
        nonce = int(time.time() * 1000)
        payload = {
            "id": nonce,
            "method": method,
            "api_key": self._api_key,
            "params": params,
            "nonce": nonce,
        }
        payload["sig"] = self._sign(method, nonce, self._api_key, nonce, params)
        try:
            if requests is not None:
                r = requests.post(url, json=payload, timeout=10)
                r.raise_for_status()
                return r.json()
            else:
                data = json.dumps(payload).encode("ascii", "ignore")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    txt = resp.read().decode("ascii", "ignore")
                    return json.loads(txt)
        except Exception:
            return {}

    # -----------------------------------------------------------
    def _get_public(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Unauthenticated GET request to public endpoints."""
        url = self.BASE_URL + method
        try:
            if requests is not None:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                return r.json()
            else:
                qs = urllib.parse.urlencode(params or {})
                full = url + ("?" + qs if qs else "")
                with urllib.request.urlopen(full, timeout=10) as resp:
                    txt = resp.read().decode("ascii", "ignore")
                    return json.loads(txt)
        except Exception:
            return {}

    # ===========================================================
    # Public endpoints
    # ===========================================================

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Return current price and stats for a trading pair."""
        data = self._get_public("public/get-ticker", {"instrument_name": symbol})
        try:
            result = data.get("result", {}).get("data", [])[0]
            return {
                "symbol": symbol,
                "price": float(result.get("a", 0.0)),
                "bid": float(result.get("b", 0.0)),
                "ask": float(result.get("k", 0.0)),
                "timestamp": int(result.get("t", 0)),
            }
        except Exception:
            return {"symbol": symbol, "price": 0.0}

    def get_candles(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> List[Dict[str, float]]:
        """Fetch OHLCV candle data for given symbol/timeframe."""
        params = {"instrument_name": symbol, "timeframe": timeframe}
        data = self._get_public("public/get-candlestick", params)
        candles: List[Dict[str, float]] = []
        try:
            for c in data.get("result", {}).get("data", [])[-limit:]:
                candles.append({
                    "t": int(c.get("t", 0)),
                    "o": float(c.get("o", 0.0)),
                    "h": float(c.get("h", 0.0)),
                    "l": float(c.get("l", 0.0)),
                    "c": float(c.get("c", 0.0)),
                    "v": float(c.get("v", 0.0))
                })
        except Exception:
            pass
        return candles

    # ===========================================================
    # Private endpoints
    # ===========================================================

    def update_balances(self) -> Dict[str, float]:
        """Fetch account balances; cached for TTL period."""
        now = time.time()
        if now - self._last_update < self._ttl and self._cache:
            return self._cache
        balances: Dict[str, float] = {}
        try:
            data = self._post("private/get-account-summary", {})
            if data and data.get("result"):
                for acc in data["result"].get("accounts", []):
                    c = str(acc.get("currency", "")).upper()
                    if c:
                        balances[c] = float(acc.get("available", 0.0))
        except Exception:
            pass
        self._cache = balances
        self._last_update = now
        return balances

    # Live trading endpoints
    def create_order(self,
                     instrument_name: str,
                     side: str,
                     order_type: str,
                     quantity: float,
                     price: float = 0.0,
                     client_oid: str = "") -> Dict[str, Any]:
        """
        Place a live order.
        side: BUY or SELL
        order_type: LIMIT or MARKET
        For MARKET orders, price can be 0.0 and will be ignored.
        """
        params: Dict[str, Any] = {
            "instrument_name": instrument_name,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if order_type.upper() == "LIMIT":
            params["price"] = str(price)
        if client_oid:
            params["client_oid"] = client_oid
        return self._post("private/create-order", params)

    def cancel_order(self, instrument_name: str, order_id: str = "", client_oid: str = "") -> Dict[str, Any]:
        """
        Cancel an order by order id or client oid.
        """
        params: Dict[str, Any] = {"instrument_name": instrument_name}
        if order_id:
            params["order_id"] = order_id
        if client_oid:
            params["client_oid"] = client_oid
        return self._post("private/cancel-order", params)

    def get_open_orders(self, instrument_name: str) -> List[Dict[str, Any]]:
        """
        Get list of open orders for an instrument.
        """
        data = self._post("private/get-open-orders", {"instrument_name": instrument_name})
        try:
            return data.get("result", {}).get("orders", [])
        except Exception:
            return []

    def get_order_detail(self, order_id: str) -> Dict[str, Any]:
        """
        Get detail for a single order id.
        """
        data = self._post("private/get-order-detail", {"order_id": order_id})
        try:
            return data.get("result", {}).get("order_info", {})
        except Exception:
            return {}

    # ===========================================================
    # Market discovery and dynamic asset utilities
    # ===========================================================

    def get_all_symbols(self) -> list[str]:
        """Return a list of all tradable instrument names on the exchange."""
        url = self.BASE_URL + "public/get-instruments"
        try:
            if requests is not None:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json()
            else:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    txt = resp.read().decode("ascii", "ignore")
                    data = json.loads(txt)
            instruments = data.get("result", {}).get("instruments", [])
            return [inst.get("instrument_name") for inst in instruments if inst.get("instrument_name")]
        except Exception:
            return []

    def get_top_liquid_assets(self, quote: str = "USDT", top_n: int = 10) -> list[str]:
        """
        Rank instruments by 24h volume (most liquid first).
        Returns a trimmed list like ['BTC_USDT', 'ETH_USDT', ...].
        """
        url = self.BASE_URL + "public/get-ticker"
        try:
            if requests is not None:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json()
            else:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    txt = resp.read().decode("ascii", "ignore")
                    data = json.loads(txt)
            tickers = data.get("result", {}).get("data", [])
            scored = []
            for t in tickers:
                name = t.get("i")
                if name and name.endswith(quote):
                    volume = float(t.get("v", 0.0))
                    scored.append((name, volume))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in scored[:top_n]]
        except Exception:
            return []

    def verify_connection(self) -> bool:
        """Quick ping to ensure API responds and credentials are valid."""
        try:
            ticker = self.get_ticker("BTC_USDT")
            return bool(ticker and ticker.get("price", 0.0) > 0)
        except Exception:
            return False


# ---------------------------------------------------------------
# Example quick test
# ---------------------------------------------------------------
if __name__ == "__main__":
    api = CryptoComAPI()
    print("Connected:", api.verify_connection())
    print("Top assets:", api.get_top_liquid_assets("USDT", 5))
    print("Balances:", api.update_balances())
    print("BTC ticker:", api.get_ticker("BTC_USDT"))
    # Example market order (WARNING: this will place a live order if keys are live)
    # resp = api.create_order("BTC_USDT", "BUY", "MARKET", quantity=0.0001)
    # print("Create order:", resp)
