# ================================================================
# Path: miner/market_data.py
# Project: Virtual Quantum Miner
# Version: v4.8.3 (ProfitCycle15s)
# ================================================================
# Description:
#   Fetches live market and network data for supported coins.
#   Queries CoinGecko API for USD price
#   Queries 2Miners public API for difficulty & hashrate
#   Computes derived "efficiency_factor" for adaptive tuning
#   Adds live profit computation every 15 s using core.utils
# ================================================================

from __future__ import annotations
from typing import Dict, Any
import time
import requests

# ---------------------------------------------------------------------------
# Layered imports
# ---------------------------------------------------------------------------
from core.utils import to_hs, hs_to_best_unit


# ----------------------------------------------------------------
# Supported coins and API mapping
# ----------------------------------------------------------------
COINS = {
    "ETC": {
        "coingecko_id": "ethereum-classic",
        "pool_api": "https://etc.2miners.com/api/stats"
    },
    "RVN": {
        "coingecko_id": "ravencoin",
        "pool_api": "https://rvn.2miners.com/api/stats"
    },
    "FLUX": {
        "coingecko_id": "zelcash",
        "pool_api": "https://flux.2miners.com/api/stats"
    },
    "ERG": {
        "coingecko_id": "ergo",
        "pool_api": "https://erg.2miners.com/api/stats"
    }
}

# ----------------------------------------------------------------
# MarketData
# ----------------------------------------------------------------
class MarketData:
    def __init__(self, vsd: Any = None):
        self.vsd = vsd
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.last_update = 0.0
        self.cache_lifetime = 60  # seconds
        self._profit_thread = None
        self._stop_flag = False
        self._profit_interval = 15.0

    # ------------------------------------------------------------
    def get_all_prices(self) -> Dict[str, Dict[str, Any]]:
        """Return dict of {coin: {price_usd, difficulty, efficiency_factor}}."""
        now = time.time()
        if now - self.last_update < self.cache_lifetime and self.cache:
            return self.cache

        data: Dict[str, Dict[str, Any]] = {}
        for symbol, meta in COINS.items():
            try:
                price = self._get_price_usd(meta["coingecko_id"])
                diff, net_hashrate = self._get_pool_stats(meta["pool_api"])
                net_hashrate_hs = utils.to_hs(net_hashrate)
                eff = self._calc_efficiency(symbol, diff, net_hashrate_hs)
                data[symbol] = {
                    "price_usd": price,
                    "difficulty": diff,
                    "network_hashrate_hs": net_hashrate_hs,
                    "network_hashrate_display": utils.auto_format_hs(net_hashrate_hs),
                    "efficiency_factor": eff
                }
            except Exception as e:
                prev = self.cache.get(symbol, {})
                data[symbol] = {
                    "price_usd": prev.get("price_usd", 0.0),
                    "difficulty": prev.get("difficulty", 0.0),
                    "network_hashrate_hs": prev.get("network_hashrate_hs", 0.0),
                    "network_hashrate_display": prev.get("network_hashrate_display", "0 H/s"),
                    "efficiency_factor": prev.get("efficiency_factor", 1.0),
                    "error": str(e)
                }

        self.cache = data
        self.last_update = now
        return data

    # ------------------------------------------------------------
    def _get_price_usd(self, coingecko_id: str) -> float:
        """Fetch coin price from CoinGecko."""
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        js = r.json()
        return float(js[coingecko_id]["usd"])

    # ------------------------------------------------------------
    def _get_pool_stats(self, pool_api: str) -> tuple[float, float]:
        """Fetch network difficulty and hashrate from 2Miners public API."""
        r = requests.get(pool_api, timeout=10)
        r.raise_for_status()
        js = r.json()
        network = js.get("nodes", [{}])[0].get("network", {})
        diff = float(network.get("difficulty", 0.0))
        hashrate = float(network.get("hashrate", 0.0))
        return diff, hashrate

    # ------------------------------------------------------------
    def _calc_efficiency(self, symbol: str, difficulty: float, hashrate_hs: float) -> float:
        """Compute a simple performance scaling factor."""
        price = self.cache.get(symbol, {}).get("price_usd", 1.0)
        if difficulty <= 0 or hashrate_hs <= 0:
            return 1.0
        eff = (price / difficulty) * (1e12 / hashrate_hs)
        return max(0.1, min(10.0, eff))

    # ------------------------------------------------------------
    # Profit computation cycle
    # ------------------------------------------------------------
    def _profit_loop(self) -> None:
        """Background thread: compute profit for each coin every 15 s."""
        while not self._stop_flag:
            try:
                snapshots = self.get_all_prices()
                for coin, snap in snapshots.items():
                    # get our accepted hashrate from VSD (if available)
                    our_hs = 0.0
                    if self.vsd:
                        our_hs = float(self.vsd.get(f"telemetry/mine/{coin}/accepted_hashrate_hs", 0.0))

                    net_hs = float(snap.get("network_hashrate_hs", 0.0))
                    block_time_s = 60.0  # default guess if not available
                    reward_coin = 1.0    # unknown block reward placeholder
                    price_usd = float(snap.get("price_usd", 0.0))

                    result = utils.compute_profit(
                        h_local_hs=our_hs,
                        h_net_hs=net_hs,
                        block_time_s=block_time_s,
                        block_reward_coin=reward_coin,
                        price_usd=price_usd
                    )

                    if result.get("ok"):
                        snap["profit_usd_day"] = result["usd_day"]
                        snap["profit_usd_hour"] = result["usd_hour"]
                        snap["profit_usd_min"] = result["usd_min"]
                        snap["profit_usd_sec"] = result["usd_sec"]

                        if self.vsd:
                            path = f"telemetry/profit/{coin}/current"
                            entry = {
                                "usd_day": result["usd_day"],
                                "usd_hour": result["usd_hour"],
                                "usd_min": result["usd_min"],
                                "usd_sec": result["usd_sec"],
                                "share_ratio": result["share_ratio"],
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                            }
                            self.vsd.store(path, entry)
            except Exception:
                pass
            time.sleep(self._profit_interval)

    def start_profit_cycle(self) -> None:
        """Launch 15 s profit-computation background thread."""
        if self._profit_thread and self._profit_thread.is_alive():
            return
        self._stop_flag = False
        self._profit_thread = threading.Thread(target=self._profit_loop, daemon=True, name="profit_cycle_15s")
        self._profit_thread.start()

    def stop_profit_cycle(self, timeout: float = 2.0) -> None:
        """Stop profit background thread."""
        self._stop_flag = True
        if self._profit_thread:
            self._profit_thread.join(timeout=timeout)
            self._profit_thread = None

# ------------------------------------------------------------
# Standalone diagnostic
# ------------------------------------------------------------
if __name__ == "__main__":
    from VHW.writer import BufferWriter
    vsd = BufferWriter(root="VHW/VD", base="telemetry")
    md = MarketData(vsd)
    md.start_profit_cycle()
    print("=== Live Market & Profit Snapshot (15 s cycle) ===")
    try:
        while True:
            time.sleep(15)
            cache = md.cache
            for c, d in cache.items():
                p = d.get("profit_usd_day", 0.0)
                print(f"{c}: ${d['price_usd']:.4f}, hash={d['network_hashrate_display']}, profit_day=${p:.6f}")
    except KeyboardInterrupt:
        md.stop_profit_cycle()
