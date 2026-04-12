# ================================================================
# File: prediction_engine/data_preprocessor.py
# Purpose:
#   Normalize and transform OHLCV market data into
#   model-ready matrices and compact state vectors.
#   Used by model_core, predictive_cluster, and state_manager.
# ================================================================

from __future__ import annotations
from typing import List, Dict, Tuple
import math
import statistics

# ================================================================
# DataPreprocessor
# ================================================================

class DataPreprocessor:
    """
    Converts raw candlestick data (list of dicts with
    keys: t, o, h, l, c, v) into normalized numeric forms.

    Public interface:
        normalize()            -> per-feature normalization
        build_feature_matrix() -> model-ready 2D list
        split_train_test()     -> train/test partition
        generate_state_vector() -> flattened latest vector
    """

    def __init__(self, smoothing_window: int = 10):
        self.smoothing_window = max(1, int(smoothing_window))

    # ------------------------------------------------------------
    def normalize(self, ohlcv: List[Dict[str, float]]) -> Dict[str, List[float]]:
        """
        Normalize OHLCV values using relative changes and
        min-max scaling to [0, 1]. Returns dict of lists.
        """
        if not ohlcv:
            return {"open": [], "high": [], "low": [], "close": [], "volume": []}

        closes = [float(c["c"]) for c in ohlcv]
        opens = [float(c["o"]) for c in ohlcv]
        highs = [float(c["h"]) for c in ohlcv]
        lows = [float(c["l"]) for c in ohlcv]
        vols = [float(c["v"]) for c in ohlcv]

        # Normalization helpers
        def minmax(values: List[float]) -> List[float]:
            mn, mx = min(values), max(values)
            if mx == mn:
                return [0.5] * len(values)
            return [(v - mn) / (mx - mn) for v in values]

        norm = {
            "open": minmax(opens),
            "high": minmax(highs),
            "low": minmax(lows),
            "close": minmax(closes),
            "volume": minmax(vols),
        }

        # Derived indicators
        norm["sma"] = self._moving_average(closes, self.smoothing_window)
        norm["ema"] = self._ema(closes, self.smoothing_window)
        norm["roc"] = self._rate_of_change(closes)
        norm["volatility"] = self._rolling_volatility(closes, self.smoothing_window)

        return norm

    # ------------------------------------------------------------
    def _moving_average(self, seq: List[float], n: int) -> List[float]:
        out: List[float] = []
        for i in range(len(seq)):
            window = seq[max(0, i - n + 1): i + 1]
            out.append(sum(window) / len(window))
        return out

    def _ema(self, seq: List[float], n: int) -> List[float]:
        if not seq:
            return []
        k = 2 / (n + 1)
        ema_values = [seq[0]]
        for price in seq[1:]:
            ema_values.append(price * k + ema_values[-1] * (1 - k))
        return ema_values

    def _rate_of_change(self, seq: List[float]) -> List[float]:
        if len(seq) < 2:
            return [0.0] * len(seq)
        return [0.0] + [(seq[i] - seq[i - 1]) / seq[i - 1] if seq[i - 1] != 0 else 0.0 for i in range(1, len(seq))]

    def _rolling_volatility(self, seq: List[float], n: int) -> List[float]:
        if not seq:
            return []
        out: List[float] = []
        for i in range(len(seq)):
            window = seq[max(0, i - n + 1): i + 1]
            if len(window) > 1:
                out.append(statistics.pstdev(window))
            else:
                out.append(0.0)
        return out

    # ------------------------------------------------------------
    def build_feature_matrix(self, normalized: Dict[str, List[float]]) -> List[List[float]]:
        """
        Assemble normalized columns into a feature matrix.
        Each row = one time step; columns = features.
        """
        keys = [k for k in normalized.keys() if normalized[k]]
        if not keys:
            return []
        length = len(normalized[keys[0]])
        matrix: List[List[float]] = []
        for i in range(length):
            row = [normalized[k][i] for k in keys]
            matrix.append(row)
        return matrix

    # ------------------------------------------------------------
    def split_train_test(self,
                         matrix: List[List[float]],
                         ratio: float = 0.8) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Split matrix into train/test sets.
        Ratio defines portion of training data.
        """
        if not matrix:
            return [], []
        cutoff = int(len(matrix) * ratio)
        train = matrix[:cutoff]
        test = matrix[cutoff:]
        return train, test

    # ------------------------------------------------------------
    def generate_state_vector(self, normalized: Dict[str, List[float]]) -> List[float]:
        """
        Flatten the last N points of normalized data into a single
        feature vector for live prediction.
        """
        if not normalized:
            return []
        latest_values: List[float] = []
        for k, v in normalized.items():
            if not v:
                continue
            latest_values.append(v[-1])
        return latest_values


# ---------------------------------------------------------------
# Example usage (debug)
# ---------------------------------------------------------------
if __name__ == "__main__":
    from random import random
    # mock OHLCV data
    candles = [{"t": i, "o": 1 + random(), "h": 1.2 + random(),
                "l": 0.8 + random(), "c": 1 + random(), "v": 1000 + random() * 10}
               for i in range(100)]
    dp = DataPreprocessor()
    norm = dp.normalize(candles)
    matrix = dp.build_feature_matrix(norm)
    train, test = dp.split_train_test(matrix)
    vector = dp.generate_state_vector(norm)
    print("Matrix rows:", len(matrix))
    print("Train/Test sizes:", len(train), len(test))
    print("Latest state vector length:", len(vector))
