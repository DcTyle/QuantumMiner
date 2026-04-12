# ================================================================
# File: prediction_engine/trade_executor.py
# Purpose:
#   Execute LIVE trades on Crypto.com Exchange based on fused signals.
#   No simulation paths included.
# ================================================================

from __future__ import annotations
from typing import Dict, Any
import time

from prediction_engine.crypto_com_api import CryptoComAPI

class TradeExecutor:
    """
    Executes live trades using CryptoComAPI.
    Expects signals with:
        symbol: str (e.g., "BTC_USDT")
        avg_predicted_change: float
        avg_confidence: float in [0, 1]
    """

    def __init__(self,
                 min_confidence: float = 0.99,
                 max_slippage: float = 0.002,
                 base_order_usd: float = 25.0):
        self.api = CryptoComAPI()
        self.min_confidence = float(min_confidence)
        self.max_slippage = float(max_slippage)
        self.base_order_usd = float(base_order_usd)

    # ------------------------------------------------------------
    def execute(self, fused_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place a live MARKET order when confidence threshold is met.
        Side is chosen by the sign of avg_predicted_change.
        Quantity is computed from base_order_usd and latest price.
        """
        symbol = str(fused_signal.get("symbol") or fused_signal.get("asset") or "")
        if not symbol:
            return {"ok": False, "reason": "missing_symbol"}

        conf = float(fused_signal.get("avg_confidence", 0.0))
        change = float(fused_signal.get("avg_predicted_change", 0.0))

        if conf < self.min_confidence:
            return {"ok": False, "reason": "low_confidence", "confidence": conf}

        side = "BUY" if change >= 0 else "SELL"
        ticker = self.api.get_ticker(symbol)
        price = float(ticker.get("price", 0.0))
        if price <= 0.0:
            return {"ok": False, "reason": "bad_price"}

        qty = self._compute_quantity(price)
        if qty <= 0.0:
            return {"ok": False, "reason": "bad_quantity"}

        # Market order placement
        client_oid = f"vm_{symbol}_{int(time.time()*1000)}"
        resp = self.api.create_order(
            instrument_name=symbol,
            side=side,
            order_type="MARKET",
            quantity=qty,
            price=0.0,
            client_oid=client_oid
        )

        out = {
            "ok": bool(resp),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price_snapshot": price,
            "confidence": conf,
            "predicted_change": change,
            "response": resp
        }
        return out

    # ------------------------------------------------------------
    def _compute_quantity(self, price: float) -> float:
        """
        Compute order quantity from base USD size and price.
        Round down to 8 decimals to avoid precision issues.
        """
        if price <= 0:
            return 0.0
        qty = self.base_order_usd / price
        # floor to 8 dp
        factor = 10_000_000
        qty = int(qty * factor) / float(factor)
        return max(qty, 0.0)
