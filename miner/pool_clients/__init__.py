# ASCII-ONLY FILE
# NetSpec base interfaces and registry
from __future__ import annotations
from typing import Any, Dict
from miner.common_types import QMJob, qmjob_from_dict

class NetID:
    BTC = 1
    LTC = 2
    ETC = 3
    RVN = 4

class NetSpecBase:
    network: str = ""
    enum_id: int = 0

    # Must return a dict with keys: powhash (hex string), mixhash (hex string or ""), header (hex string)
    def hash_fn(self, job: Any, nonce: int) -> Dict[str, str]:
        raise NotImplementedError()

    # Must return True if powhash meets target implied by job
    def target_fn(self, job: Any, powhash_hex: str) -> bool:
        raise NotImplementedError()

    # Convert raw pool job -> normalized QMJob dict
    def network_to_system_fn(self, raw: Dict[str, Any]) -> QMJob:
        raise NotImplementedError()

    # Convert QMShare -> pool submission payload dict
    def system_to_network_fn(self, qmshare: Any) -> Dict[str, Any]:
        raise NotImplementedError()

    # Determine batch size for this network and job
    def compute_batch_size(self, job: Dict[str, Any]) -> int:
        return 64

# Registry

def get_netspec(network: str) -> NetSpecBase:
    net = str(network).upper()
    if net == "BTC":
        from .btc_client import NetSpec_BTC
        return NetSpec_BTC()
    if net == "LTC":
        from .ltc_client import NetSpec_LTC
        return NetSpec_LTC()
    if net == "ETC":
        from .etc_client import NetSpec_ETC
        return NetSpec_ETC()
    if net == "RVN":
        from .rvn_client import NetSpec_RVN
        return NetSpec_RVN()
    raise ValueError("unsupported network: %s" % net)
