# ================================================================
# Quantum Application / miner
# File: pool_stats.py
# Version: v4.8.3
# ----------------------------------------------------------------
# Purpose:
#   Live pool stats and USD pricing via 2Miners and Coinbase.
#   Normalizes all hashrate units to canonical H/s using core.utils.
#   Provides readable display strings for telemetry and profit math.
# ================================================================

from __future__ import annotations
from typing import Dict, Any, Optional
import time
import json
try:
    import requests
except Exception:
    requests = None
from urllib.request import Request, urlopen
# ---------------------------------------------------------------------------
# Layered imports
# ---------------------------------------------------------------------------
from core.utils import to_hs, auto_format_hs


# ----------------------------------------------------------------
# Mappings for supported networks
# ----------------------------------------------------------------
COIN_TO_2MINERS = {
    "BTC": "btc",
    "ETC": "etc",
    "RVN": "rvn",
    "LTC": "ltc",
    "ERG": "erg",
}

COIN_TO_COINBASE = {
    "BTC": "BTC-USD",
    "ETC": "ETC-USD",
    "RVN": "RVN-USD",
    "LTC": "LTC-USD",
    "ERG": "ERG-USD",
}

# ----------------------------------------------------------------
# Class: PoolStats
# ----------------------------------------------------------------
class PoolStats:
    def __init__(self, ttl_s: int = 60):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_update = 0.0
        self._ttl = int(ttl_s)

    # ------------------------------------------------------------
    # Fetch network statistics from 2Miners
    # ------------------------------------------------------------
    def _fetch_json(self, url: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        if requests is not None:
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code != 200:
                    return None
                return r.json() or {}
            except Exception:
                return None
        try:
            req = Request(url, headers={"User-Agent": "QuantumMiner/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", 200))
                if status != 200:
                    return None
                raw = resp.read().decode("utf-8", "ignore")
            data = json.loads(raw or "{}")
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _fetch_2miners(self, coin: str) -> Optional[Dict[str, Any]]:
        sym = coin.upper()
        short = COIN_TO_2MINERS.get(sym)
        if not short:
            return None
        url = f"https://{short}.2miners.com/api/stats"
        try:
            data = self._fetch_json(url, timeout=10)
            if not data:
                return None
            node = {}
            try:
                nodes = list(data.get("nodes", []))
                if nodes:
                    node = dict(nodes[0] or {})
            except Exception:
                node = {}

            # Prefer total network hashrate; top-level "hashrate" is pool-side.
            raw_hash = node.get("networkhashps", data.get("network_hashrate_hs", 0.0))
            block_time = float(node.get("avgBlockTime", data.get("blockTime", 0.0)))
            reward_coin = float(data.get("blockReward", 0.0))
            difficulty = float(node.get("difficulty", data.get("difficulty", 0.0)))

            # normalize hashrate via core.utils (in case API changes unit)
            try:
                h_val = to_hs(raw_hash)
                h_disp = auto_format_hs(h_val)
            except Exception:
                h_val = float(raw_hash)
                h_disp = f"{h_val} H/s"

            res = {
                "network_hashrate_hs": h_val,
                "network_hashrate_display": h_disp,
                "block_time_s": block_time,
                "block_reward_coin": reward_coin,
                "difficulty": difficulty,
            }
            return res
        except Exception:
            return None

    # ------------------------------------------------------------
    # Fetch spot USD price from Coinbase
    # ------------------------------------------------------------
    def _fetch_price(self, coin: str) -> float:
        sym = coin.upper()
        pair = COIN_TO_COINBASE.get(sym)
        if not pair:
            return 0.0
        url = f"https://api.coinbase.com/v2/prices/{pair}/spot"
        try:
            data = self._fetch_json(url, timeout=10) or {}
            return float(data.get("data", {}).get("amount", 0.0))
        except Exception:
            return 0.0

    # ------------------------------------------------------------
    # Refresh all network data (respecting TTL)
    # ------------------------------------------------------------
    def _refresh_all(self, coins_cfg: Dict[str, Any]) -> None:
        now = time.time()
        if now - self._last_update < self._ttl:
            return
        out: Dict[str, Dict[str, Any]] = {}
        for sym in coins_cfg.keys():
            coin_u = sym.upper()
            net = self._fetch_2miners(coin_u) or {}
            price = self._fetch_price(coin_u)
            if net:
                net["price_usd"] = price
                net["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                # safety: ensure H/s field present
                if "network_hashrate_hs" not in net and "network_hashrate" in net:
                    try:
                        h_val = to_hs(net["network_hashrate"])
                        net["network_hashrate_hs"] = h_val
                        net["network_hashrate_display"] = auto_format_hs(h_val)
                    except Exception:
                        pass
                out[coin_u] = net
        self._cache = out
        self._last_update = now

    # ------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------
    def get_all(self, coins_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve normalized stats for each coin in coins_cfg.
        Returns:
            { COIN: { network_hashrate_hs, block_time_s,
                      block_reward_coin, price_usd, ts } }
        """
        self._refresh_all(coins_cfg)
        return dict(self._cache)

# ----------------------------------------------------------------
# Manual test
# ----------------------------------------------------------------
if __name__ == "__main__":
    ps = PoolStats(ttl_s=5)
    result = ps.get_all({"ETC": {}, "RVN": {}})
    for coin, snap in result.items():
        hs_disp = snap.get("network_hashrate_display", "?")
        hs_val = snap.get("network_hashrate_hs", 0.0)
        print(f"{coin}: {hs_disp} (canonical {hs_val} H/s)  Price: ${snap.get('price_usd', 0.0)}")
