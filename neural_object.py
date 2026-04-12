# ASCII-ONLY FILE
# Universal Compute Unification Layer
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
import hashlib
import time
import math


class ComputeNetwork(Enum):
    UNKNOWN = 0
    BTC = auto()
    LTC = auto()
    ETC = auto()
    RVN = auto()


class variable_format_enum(Enum):
    BTC_BlockTemplate = auto()
    LTC_ScryptTemplate = auto()
    ETC_EtchashTemplate = auto()
    RVN_KawpowJob = auto()
    Miner_DerivativeNonce = auto()
    Neuralis_TimeSeries = auto()
    Neuralis_ContextPacket = auto()
    Prediction_CorrelationPacket = auto()
    Prediction_AssetVector = auto()
    VSD_StateSnapshot = auto()
    BIOS_TelemetryFrame = auto()


@dataclass
class neural_objectPacket:
    packet_type: variable_format_enum
    network: ComputeNetwork
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    system_payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    derived_state: Dict[str, Any] = field(default_factory=dict)


class neural_object:
    def __init__(self, lane_id: str, dmt_env: Optional[Dict[str, float]] = None) -> None:
        self.lane_id = str(lane_id)
        self.state: Dict[str, Any] = {
            "base_nonce": int(hashlib.sha256(self.lane_id.encode("ascii")).hexdigest()[:8], 16),
            "psi": 0.0,
            "flux": 0.0,
            "harmonic": 0.0,
            "phase": 0.314159265,
            "d1": 0,
        }
        self.dmt_env = dict(dmt_env or {})

    def get_state(self) -> Dict[str, Any]:
        return dict(self.state)

    def save_state(self, st: Dict[str, Any]) -> None:
        self.state = dict(st)

    def _lane_seed(self) -> int:
        h = 2166136261
        for ch in self.lane_id.encode("ascii", errors="ignore"):
            h ^= ch
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def _dmt_update(self, state: Dict[str, Any], env: Dict[str, float]) -> None:
        gu = float(env.get("global_util", self.dmt_env.get("global_util", 0.5)))
        gpu = float(env.get("gpu_util", self.dmt_env.get("gpu_util", 0.5)))
        mbw = float(env.get("mem_bw_util", self.dmt_env.get("mem_bw_util", 0.5)))
        cpu = float(env.get("cpu_util", self.dmt_env.get("cpu_util", 0.5)))
        a_flux = float(env.get("alpha_flux", self.dmt_env.get("alpha_flux", 0.15)))
        f_coeff = float(env.get("flux_coeff", self.dmt_env.get("flux_coeff", 0.20)))
        d_coeff = float(env.get("drift_coeff", self.dmt_env.get("drift_coeff", 0.35)))
        p_step = float(env.get("phase_step", self.dmt_env.get("phase_step", 0.07)))

        psi = float(state.get("psi", 0.0))
        phase = float(state.get("phase", 0.0))

        psi_next = psi + a_flux * (gu - gpu)
        flux = f_coeff * (mbw - cpu)
        harmonic = d_coeff * math.sin(2.0 * math.pi * phase)
        phase_next = (phase + p_step) % 1.0

        state["psi"] = psi_next
        state["flux"] = flux
        state["harmonic"] = harmonic
        state["phase"] = phase_next

    def _derive_nonce(self, state: Dict[str, Any], lane_seed: int) -> int:
        base_nonce = int(state.get("base_nonce", 0)) & 0xFFFFFFFF
        d1 = int(state.get("d1", 0)) & 0xFFFFFFFF
        psi = float(state.get("psi", 0.0))
        flux = float(state.get("flux", 0.0))
        harmonic = float(state.get("harmonic", 0.0))
        phase = float(state.get("phase", 0.0))

        psi_flux_term = int((psi + flux) * (1 << 16)) & 0xFFFFFFFF
        lane_phase_shift = int((phase * 977) + (lane_seed % 4093)) & 0xFFFFFFFF
        harmonic_drift = int(harmonic * (1 << 18)) & 0xFFFFFFFF
        nonce_next = (base_nonce + d1 + psi_flux_term + harmonic_drift + lane_phase_shift) & 0xFFFFFFFF
        state["base_nonce"] = (base_nonce * 1664525 + 1013904223) & 0xFFFFFFFF
        state["d1"] = (d1 + 1) & 0xFFFFFFFF
        return nonce_next

    def evolve(self, packet: neural_objectPacket) -> neural_objectPacket:
        # Generic evolution step for non-mining packets; performs one DMT update
        st = self.get_state()
        env = dict(packet.system_payload or {})
        self._dmt_update(st, env)
        st["last_evolve_ts"] = time.time()
        self.save_state(st)
        out = neural_objectPacket(
            packet_type=packet.packet_type,
            network=packet.network,
            raw_payload=dict(packet.raw_payload),
            system_payload=dict(packet.system_payload),
            metadata=dict(packet.metadata),
            derived_state=st,
        )
        return out


def _safe_hex(s: str) -> str:
    if not isinstance(s, str):
        return ""
    if s.startswith("0x"):
        return s[2:]
    return s


def _btc_double_sha256(header_hex: str, nonce_int: int) -> str:
    hhex = _safe_hex(header_hex)
    try:
        header = bytes.fromhex(hhex) if hhex else b""
    except Exception:
        header = b""
    hdr = bytearray(header[:80].ljust(80, b"\x00"))
    nonce_bytes = int(nonce_int & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
    hdr[76:80] = nonce_bytes
    return hashlib.sha256(hashlib.sha256(bytes(hdr)).digest()).hexdigest()


def _cmp_pow_leq_target(pow_hex: str, target_hex: str) -> bool:
    try:
        p = int(_safe_hex(pow_hex) or "0", 16)
        t = int(_safe_hex(target_hex) or "0", 16)
        return p <= t
    except Exception:
        return False


def _preferred_job_target(job: Dict[str, Any]) -> str:
    if not isinstance(job, dict):
        return ""
    share_target = str(job.get("share_target", "") or "")
    if share_target:
        return share_target
    active_target = str(job.get("active_target", "") or "")
    if active_target:
        return active_target
    return str(job.get("target", "") or "")


def _schema_convert_generic(raw: Dict[str, Any], network_name: str) -> Dict[str, Any]:
    out = dict(raw or {})
    out["network"] = network_name
    return {
        "raw_payload": out,
        "system_payload": {},
        "metadata": {},
        "derived_state": {},
    }


def _normalized_job_payload(raw: Dict[str, Any], header_keys: List[str], required_keys: List[str]) -> bool:
    if not isinstance(raw, dict):
        return False
    if not any(str(raw.get(key, "")) for key in header_keys):
        return False
    for key in required_keys:
        if str(raw.get(key, "")) == "":
            return False
    return True


# Schema function implementations per packet type
def _btc_convert_incoming(raw: Dict[str, Any]) -> Dict[str, Any]:
    if _normalized_job_payload(raw, ["header_hex", "header"], ["target"]):
        return {
            "raw_payload": dict(raw or {}),
            "system_payload": {},
            "metadata": {},
            "derived_state": {},
        }
    try:
        from miner.stratum_adapters import BTCAdapter
        job = BTCAdapter().convert_job(raw)
    except Exception:
        job = {}
    return {
        "raw_payload": job,
        "system_payload": {},
        "metadata": {},
        "derived_state": {},
    }


def _btc_convert_outgoing(packet: neural_objectPacket) -> Dict[str, Any]:
    # Expect nonce and job in system_payload
    sp = packet.system_payload or {}
    job = sp.get("job", {}) if isinstance(sp, dict) else {}
    nonce = sp.get("nonce")
    extranonce2 = job.get("extranonce2")
    ntime = job.get("ntime")
    job_id = job.get("job_id")
    return {
        "job_id": job_id,
        "extranonce2": extranonce2,
        "ntime": ntime,
        "nonce": f"{int(nonce):08x}" if nonce is not None else "00000000",
        "lane": sp.get("lane"),
        "received_at": sp.get("received_at", 0.0),
        "bucket_id": sp.get("bucket_id", ""),
        "worker_bucket": sp.get("worker_bucket", ""),
        "bucket_priority": float(sp.get("bucket_priority", 0.0) or 0.0),
        "target_interval": int(sp.get("target_interval", 0) or 0),
        "sequence_index": int(sp.get("sequence_index", 0) or 0),
        "decoded_data": dict(sp.get("decoded_data", {}) or {}),
        "pow_test": dict(sp.get("pow_test", {}) or {}),
    }


def _btc_pack_header(packet_norm: Dict[str, Any]) -> str:
    job = packet_norm.get("raw_payload", {})
    return str(job.get("header_hex", job.get("header", "")))


def _btc_hash_function(packet_norm: Dict[str, Any], nonce: int) -> str:
    try:
        from miner.algos.sha256d_pow import sha256d_compute_share

        share = sha256d_compute_share(packet_norm.get("raw_payload", {}), int(nonce))
        return str(share.get("hash_hex", "")).replace("0x", "")
    except Exception:
        hdr = _btc_pack_header(packet_norm)
        return _btc_double_sha256(hdr, nonce)


def _btc_verify_target(packet_norm: Dict[str, Any], pow_hex: str) -> bool:
    job = packet_norm.get("raw_payload", {})
    return _cmp_pow_leq_target(pow_hex, _preferred_job_target(job))


def _ltc_convert_incoming(raw: Dict[str, Any]) -> Dict[str, Any]:
    if _normalized_job_payload(raw, ["header_hex", "header"], ["target"]):
        return {"raw_payload": dict(raw or {}), "system_payload": {}, "metadata": {}, "derived_state": {}}
    try:
        from miner.stratum_adapters import LTCAdapter
        job = LTCAdapter().convert_job(raw)
    except Exception:
        job = {}
    return {"raw_payload": job, "system_payload": {}, "metadata": {}, "derived_state": {}}


def _ltc_convert_outgoing(packet: neural_objectPacket) -> Dict[str, Any]:
    sp = packet.system_payload or {}
    job = sp.get("job", {}) if isinstance(sp, dict) else {}
    nonce = sp.get("nonce")
    return {
        "job_id": job.get("job_id"),
        "extranonce2": job.get("extranonce2"),
        "ntime": job.get("ntime"),
        "nonce": f"{int(nonce):08x}" if nonce is not None else "00000000",
        "lane": sp.get("lane"),
        "received_at": sp.get("received_at", 0.0),
        "bucket_id": sp.get("bucket_id", ""),
        "worker_bucket": sp.get("worker_bucket", ""),
        "bucket_priority": float(sp.get("bucket_priority", 0.0) or 0.0),
        "target_interval": int(sp.get("target_interval", 0) or 0),
        "sequence_index": int(sp.get("sequence_index", 0) or 0),
    }


def _ltc_hash_function(packet_norm: Dict[str, Any], nonce: int) -> str:
    try:
        from miner.algos.scrypt_pow import scrypt_compute_share
        extra = scrypt_compute_share(packet_norm.get("raw_payload", {}), nonce)
        return str(extra.get("hash_hex", "")).replace("0x", "")
    except Exception:
        return ""


def _ltc_verify_target(packet_norm: Dict[str, Any], pow_hex: str) -> bool:
    job = packet_norm.get("raw_payload", {})
    return _cmp_pow_leq_target(pow_hex, _preferred_job_target(job))


def _etc_convert_incoming(raw: Dict[str, Any]) -> Dict[str, Any]:
    if _normalized_job_payload(raw, ["header_hash", "header_hex", "header"], ["target"]):
        return {"raw_payload": dict(raw or {}), "system_payload": {}, "metadata": {}, "derived_state": {}}
    try:
        from miner.stratum_adapters.adapter_etc import ETCAdapter  # type: ignore
        job = ETCAdapter().convert_job(raw)  # adapter_etc follows base interface
    except Exception:
        # Some repos expose only through __init__ import
        try:
            from miner.stratum_adapters import ETCAdapter  # type: ignore
            job = ETCAdapter().convert_job(raw)
        except Exception:
            job = {}
    return {"raw_payload": job, "system_payload": {}, "metadata": {}, "derived_state": {}}


def _etc_convert_outgoing(packet: neural_objectPacket) -> Dict[str, Any]:
    sp = packet.system_payload or {}
    job = sp.get("job", {}) if isinstance(sp, dict) else {}
    nonce = sp.get("nonce")
    return {
        "job_id": job.get("job_id"),
        "nonce": f"0x{int(nonce)&0xFFFFFFFF:08x}" if nonce is not None else "0x00000000",
        "header_hash": sp.get("header_hash", job.get("header_hash", job.get("header_hex", job.get("header", "")))),
        "mix_hash": sp.get("mix_hash", job.get("mix_hash", "")),
        "lane": sp.get("lane"),
        "received_at": sp.get("received_at", 0.0),
        "bucket_id": sp.get("bucket_id", ""),
        "worker_bucket": sp.get("worker_bucket", ""),
        "bucket_priority": float(sp.get("bucket_priority", 0.0) or 0.0),
        "target_interval": int(sp.get("target_interval", 0) or 0),
        "sequence_index": int(sp.get("sequence_index", 0) or 0),
    }


def _etc_hash_function(packet_norm: Dict[str, Any], nonce: int) -> Dict[str, Any]:
    try:
        from miner.algos.etchash import etchash_compute_share
        return dict(etchash_compute_share(packet_norm.get("raw_payload", {}), nonce) or {})
    except Exception:
        return {}


def _etc_verify_target(packet_norm: Dict[str, Any], pow_hex: str) -> bool:
    job = packet_norm.get("raw_payload", {})
    return _cmp_pow_leq_target(pow_hex, _preferred_job_target(job))


def _rvn_convert_incoming(raw: Dict[str, Any]) -> Dict[str, Any]:
    if _normalized_job_payload(raw, ["header_hash", "header_hex", "header"], ["target"]):
        return {"raw_payload": dict(raw or {}), "system_payload": {}, "metadata": {}, "derived_state": {}}
    try:
        from miner.stratum_adapters import RVNAdapter
        job = RVNAdapter().convert_job(raw)
    except Exception:
        job = {}
    return {"raw_payload": job, "system_payload": {}, "metadata": {}, "derived_state": {}}


def _rvn_convert_outgoing(packet: neural_objectPacket) -> Dict[str, Any]:
    sp = packet.system_payload or {}
    job = sp.get("job", {}) if isinstance(sp, dict) else {}
    nonce = sp.get("nonce")
    return {
        "job_id": job.get("job_id"),
        "nonce": f"0x{int(nonce)&0xFFFFFFFF:08x}" if nonce is not None else "0x00000000",
        "header": sp.get("header_hash", job.get("header_hash", job.get("header_hex", job.get("header", "")))),
        "mix_hash": sp.get("mix_hash", job.get("mix_hash", "")),
        "target": job.get("target"),
        "lane": sp.get("lane"),
        "received_at": sp.get("received_at", 0.0),
        "bucket_id": sp.get("bucket_id", ""),
        "worker_bucket": sp.get("worker_bucket", ""),
        "bucket_priority": float(sp.get("bucket_priority", 0.0) or 0.0),
        "target_interval": int(sp.get("target_interval", 0) or 0),
        "sequence_index": int(sp.get("sequence_index", 0) or 0),
    }


def _rvn_hash_function(packet_norm: Dict[str, Any], nonce: int) -> Dict[str, Any]:
    try:
        from miner.algos.kawpow import kawpow_compute_share
        return dict(kawpow_compute_share(packet_norm.get("raw_payload", {}), nonce) or {})
    except Exception:
        return {}


def _rvn_verify_target(packet_norm: Dict[str, Any], pow_hex: str) -> bool:
    job = packet_norm.get("raw_payload", {})
    return _cmp_pow_leq_target(pow_hex, _preferred_job_target(job))




def _simple_batch_size(_packet_norm: Dict[str, Any]) -> int:
    return 64


neural_objectSchema: Dict[variable_format_enum, Dict[str, Any]] = {
    variable_format_enum.BTC_BlockTemplate: {
        "convert_incoming": _btc_convert_incoming,
        "convert_outgoing": _btc_convert_outgoing,
        "pack_header": _btc_pack_header,
        "verify_target": _btc_verify_target,
        "derive_batch_size": _simple_batch_size,
        "hash_function": _btc_hash_function,
        "required_fields": ["job_id", "header_hex", "target"],
    },
    variable_format_enum.LTC_ScryptTemplate: {
        "convert_incoming": _ltc_convert_incoming,
        "convert_outgoing": _ltc_convert_outgoing,
        "pack_header": lambda n: str(n.get("raw_payload", {}).get("header_hex", "")),
        "verify_target": _ltc_verify_target,
        "derive_batch_size": _simple_batch_size,
        "hash_function": _ltc_hash_function,
        "required_fields": ["job_id", "header_hex", "target"],
    },
    variable_format_enum.ETC_EtchashTemplate: {
        "convert_incoming": _etc_convert_incoming,
        "convert_outgoing": _etc_convert_outgoing,
        "pack_header": lambda n: str(n.get("raw_payload", {}).get("header_hash", "")),
        "verify_target": _etc_verify_target,
        "derive_batch_size": _simple_batch_size,
        "hash_function": _etc_hash_function,
        "required_fields": ["job_id", "header_hash", "seed_hash", "target"],
    },
    variable_format_enum.RVN_KawpowJob: {
        "convert_incoming": _rvn_convert_incoming,
        "convert_outgoing": _rvn_convert_outgoing,
        "pack_header": lambda n: str(n.get("raw_payload", {}).get("header_hash", "")),
        "verify_target": _rvn_verify_target,
        "derive_batch_size": _simple_batch_size,
        "hash_function": _rvn_hash_function,
        "required_fields": ["job_id", "header_hash", "target"],
    },
    # Non-mining packets: use identity transforms
    variable_format_enum.Miner_DerivativeNonce: {
        "convert_incoming": lambda raw: {"raw_payload": dict(raw or {}), "system_payload": {}, "metadata": {}, "derived_state": {}},
        "convert_outgoing": lambda pkt: dict(pkt.raw_payload or {}),
        "pack_header": lambda _n: "",
        "verify_target": lambda _n, _p: True,
        "derive_batch_size": lambda _n: 1,
        "hash_function": lambda _n, _nonce: "",
        "required_fields": [],
    },
    variable_format_enum.Neuralis_TimeSeries: {
        "convert_incoming": _schema_convert_generic,
        "convert_outgoing": lambda pkt: dict(pkt.raw_payload or {}),
        "pack_header": lambda _n: "",
        "verify_target": lambda _n, _p: True,
        "derive_batch_size": lambda _n: 1,
        "hash_function": lambda _n, _nonce: "",
        "required_fields": [],
    },
    variable_format_enum.Neuralis_ContextPacket: {
        "convert_incoming": _schema_convert_generic,
        "convert_outgoing": lambda pkt: dict(pkt.raw_payload or {}),
        "pack_header": lambda _n: "",
        "verify_target": lambda _n, _p: True,
        "derive_batch_size": lambda _n: 1,
        "hash_function": lambda _n, _nonce: "",
        "required_fields": [],
    },
    variable_format_enum.Prediction_CorrelationPacket: {
        "convert_incoming": _schema_convert_generic,
        "convert_outgoing": lambda pkt: dict(pkt.raw_payload or {}),
        "pack_header": lambda _n: "",
        "verify_target": lambda _n, _p: True,
        "derive_batch_size": lambda _n: 1,
        "hash_function": lambda _n, _nonce: "",
        "required_fields": [],
    },
    variable_format_enum.Prediction_AssetVector: {
        "convert_incoming": _schema_convert_generic,
        "convert_outgoing": lambda pkt: dict(pkt.raw_payload or {}),
        "pack_header": lambda _n: "",
        "verify_target": lambda _n, _p: True,
        "derive_batch_size": lambda _n: 1,
        "hash_function": lambda _n, _nonce: "",
        "required_fields": [],
    },
    variable_format_enum.VSD_StateSnapshot: {
        "convert_incoming": _schema_convert_generic,
        "convert_outgoing": lambda pkt: dict(pkt.raw_payload or {}),
        "pack_header": lambda _n: "",
        "verify_target": lambda _n, _p: True,
        "derive_batch_size": lambda _n: 1,
        "hash_function": lambda _n, _nonce: "",
        "required_fields": [],
    },
    variable_format_enum.BIOS_TelemetryFrame: {
        "convert_incoming": _schema_convert_generic,
        "convert_outgoing": lambda pkt: dict(pkt.raw_payload or {}),
        "pack_header": lambda _n: "",
        "verify_target": lambda _n, _p: True,
        "derive_batch_size": lambda _n: 1,
        "hash_function": lambda _n, _nonce: "",
        "required_fields": [],
    },
}


def network_to_packet_type(net: str) -> variable_format_enum:
    n = (net or "").upper()
    if n == "BTC":
        return variable_format_enum.BTC_BlockTemplate
    if n == "LTC":
        return variable_format_enum.LTC_ScryptTemplate
    if n == "ETC":
        return variable_format_enum.ETC_EtchashTemplate
    if n == "RVN":
        return variable_format_enum.RVN_KawpowJob
    return variable_format_enum.Miner_DerivativeNonce


def name_to_network(net: str) -> ComputeNetwork:
    n = (net or "").upper()
    if n == "BTC":
        return ComputeNetwork.BTC
    if n == "LTC":
        return ComputeNetwork.LTC
    if n == "ETC":
        return ComputeNetwork.ETC
    if n == "RVN":
        return ComputeNetwork.RVN
    return ComputeNetwork.UNKNOWN
