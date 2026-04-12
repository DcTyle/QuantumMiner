# ================================================================
# File: prediction_engine/ensemble_fusion.py
# Purpose:
#   Combine multiple cluster outputs into a single fused result.
#   Performs confidence-weighted averaging and ranking.
# ================================================================

from __future__ import annotations
from typing import List, Dict, Any

# ================================================================
# EnsembleFusion
# ================================================================

class EnsembleFusion:
    """
    Merges results from multiple predictive clusters.
    Combines per-asset predictions using confidence-weighted
    averaging. Produces a sorted list of fused results.
    """

    def __init__(self, weight_cap: float = 0.99):
        self.weight_cap = float(weight_cap)

    # ------------------------------------------------------------
    def combine(self, cluster_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fuse all cluster outputs into one list of averaged predictions.
        """
        if not cluster_outputs:
            return []

        fusion_map: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, int] = {}

        for res in cluster_outputs:
            symbol = str(res.get("symbol", "")).upper()
            if not symbol:
                continue
            fusion_map.setdefault(symbol, {"predicted_change": 0.0, "confidence": 0.0})
            fusion_map[symbol]["predicted_change"] += float(res.get("predicted_change", 0.0))
            fusion_map[symbol]["confidence"] += float(res.get("confidence", 0.0))
            counts[symbol] = counts.get(symbol, 0) + 1

        fused_results: List[Dict[str, Any]] = []
        for symbol, vals in fusion_map.items():
            n = counts.get(symbol, 1)
            avg_pred = vals["predicted_change"] / n
            avg_conf = min(vals["confidence"] / n, self.weight_cap)
            fused_results.append({
                "symbol": symbol,
                "avg_predicted_change": avg_pred,
                "avg_confidence": avg_conf,
            })

        fused_results.sort(key=lambda x: x["avg_confidence"], reverse=True)
        return fused_results


# ---------------------------------------------------------------
# Example standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    sample = [
        {"symbol": "BTC_USDT", "predicted_change": 0.02, "confidence": 0.96},
        {"symbol": "BTC_USDT", "predicted_change": 0.03, "confidence": 0.98},
        {"symbol": "ETH_USDT", "predicted_change": -0.01, "confidence": 0.90},
        {"symbol": "ETH_USDT", "predicted_change": -0.02, "confidence": 0.92},
    ]
    fusion = EnsembleFusion(weight_cap=0.99)
    fused = fusion.combine(sample)
    print("Fused results:")
    for r in fused:
        print(r)
