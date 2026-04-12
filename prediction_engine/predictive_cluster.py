# ================================================================
# File: prediction_engine/predictive_cluster.py
# Purpose:
#   Manage parallel prediction clusters and aggregate lane outputs.
#   Each cluster handles multiple assets using threads.
# ================================================================

from __future__ import annotations
from typing import List, Dict, Any
import threading
import concurrent.futures
import time
import random

from prediction_engine.model_core import ModelCore

# ================================================================
# PredictiveCluster
# ================================================================

class PredictiveCluster:
    """
    One cluster represents a threaded group of lanes.
    Each lane executes one model prediction for a symbol.
    """

    def __init__(self, cluster_id: int, symbols: List[str], model_core: ModelCore):
        self.cluster_id = cluster_id
        self.symbols = symbols
        self.model_core = model_core
        self.results: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    # ------------------------------------------------------------
    def run_cluster(self, market_data: Dict[str, List[Dict[str, float]]]) -> List[Dict[str, Any]]:
        """
        Run predictions for all assigned symbols.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.symbols)) as executor:
            futures = []
            for sym in self.symbols:
                candles = market_data.get(sym, [])
                futures.append(executor.submit(self._lane_predict, sym, candles))
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    with self.lock:
                        self.results.append(res)
                except Exception as exc:
                    print(f"[Cluster {self.cluster_id}] Lane failed: {exc}")
        return self.results

    # ------------------------------------------------------------
    def _lane_predict(self, symbol: str, candles: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Worker for one lane. Executes model_core training and prediction.
        """
        if not candles:
            return {"symbol": symbol, "predicted_change": 0.0, "confidence": 0.0}
        # Train if not available
        if symbol not in self.model_core.models:
            self.model_core.train(symbol, candles)
        norm = self.model_core.preprocessor.normalize(candles)
        vec = self.model_core.preprocessor.generate_state_vector(norm)
        result = self.model_core.predict(symbol, vec)
        result["cluster_id"] = self.cluster_id
        return result


# ================================================================
# PredictiveClusterManager
# ================================================================

class PredictiveClusterManager:
    """
    Coordinates multiple predictive clusters.
    Handles allocation and execution of lane groups.
    """

    def __init__(self, max_clusters: int = 8, lanes_per_cluster: int = 8):
        self.max_clusters = int(max_clusters)
        self.lanes_per_cluster = int(lanes_per_cluster)
        self.clusters: List[PredictiveCluster] = []
        self.model_core = ModelCore()

    # ------------------------------------------------------------
    def allocate_clusters(self, asset_list: List[str]) -> None:
        """
        Split the asset list among available clusters.
        """
        self.clusters = []
        if not asset_list:
            return
        total = len(asset_list)
        count = min(self.max_clusters, max(1, (total // self.lanes_per_cluster) + 1))
        per_cluster = max(1, total // count)
        for i in range(count):
            start = i * per_cluster
            end = start + per_cluster
            subset = asset_list[start:end]
            cluster = PredictiveCluster(i, subset, self.model_core)
            self.clusters.append(cluster)
        print(f"[ClusterManager] Allocated {len(self.clusters)} clusters for {len(asset_list)} assets.")

    # ------------------------------------------------------------
    def run_all(self, market_data: Dict[str, List[Dict[str, float]]]) -> List[Dict[str, Any]]:
        """
        Execute all clusters concurrently and aggregate results.
        """
        all_results: List[Dict[str, Any]] = []
        if not self.clusters:
            return all_results
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clusters)) as executor:
            futures = [executor.submit(c.run_cluster, market_data) for c in self.clusters]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    all_results.extend(fut.result())
                except Exception as exc:
                    print(f"[ClusterManager] Cluster run error: {exc}")
        return all_results

# ---------------------------------------------------------------
# Example standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    from random import random
    # mock candle data for several symbols
    symbols = ["BTC_USDT", "ETH_USDT", "LTC_USDT", "CRO_USDT"]
    mock_data = {}
    for s in symbols:
        mock_data[s] = [{"t": i, "o": 1 + random(), "h": 1.2 + random(),
                         "l": 0.8 + random(), "c": 1 + random(), "v": 1000 + random() * 10}
                        for i in range(200)]
    mgr = PredictiveClusterManager(max_clusters=2, lanes_per_cluster=2)
    mgr.allocate_clusters(symbols)
    results = mgr.run_all(mock_data)
    print("Cluster results:")
    for r in results:
        print(r)
