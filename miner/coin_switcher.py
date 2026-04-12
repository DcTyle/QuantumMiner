from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple


class SupportedCoin(str, Enum):
    BTC = "BTC"
    LTC = "LTC"
    ETC = "ETC"
    RVN = "RVN"


@dataclass(frozen=True)
class CoinProfile:
    coin: SupportedCoin
    mode: str
    algorithm: str
    wallet: str
    worker_index: int
    worker_name: str
    username: str
    pool_url: str
    host: str
    port: int
    endpoint: str
    use_tls: bool
    subscribe_method: str
    subscribe_params: Tuple[Any, ...]
    authorize_method: str
    authorize_with_password: bool
    password: str = "x"


def coin_config_entry(coins: Dict[str, Any], network: str) -> Dict[str, Any]:
    network_u = str(network).upper()
    direct = dict(coins or {}).get(network_u)
    if isinstance(direct, dict):
        return dict(direct)

    entry: Dict[str, Any] = {}
    for key, value in dict(coins or {}).items():
        if isinstance(value, dict) and network_u in value:
            entry[str(key)] = value.get(network_u)
    return entry


def configured_coins(coins: Dict[str, Any]) -> List[str]:
    discovered: List[str] = []
    for key, value in dict(coins or {}).items():
        key_u = str(key).upper()
        if isinstance(value, dict) and key_u in SupportedCoin.__members__ and key_u not in discovered:
            discovered.append(key_u)
    for coin in SupportedCoin:
        if coin.value not in discovered and coin_config_entry(coins, coin.value):
            discovered.append(coin.value)
    return discovered


def _parse_pool_url(pool_url: str, fallback_port: int) -> Tuple[str, str, int, bool]:
    raw = str(pool_url or "").strip()
    scheme = "stratum+tcp"
    host = ""
    port = int(fallback_port or 0)
    use_tls = False
    if "://" in raw:
        scheme, host_port = raw.split("://", 1)
        use_tls = "ssl" in scheme.lower() or "tls" in scheme.lower()
        try:
            host, port_str = host_port.rsplit(":", 1)
            port = int(port_str)
        except Exception:
            host = host_port
    elif raw:
        host = raw
    return scheme, host, max(0, int(port or 0)), use_tls


def _normalize_worker_index(worker_index: Any) -> int:
    try:
        value = int(worker_index)
    except Exception:
        value = 1
    return max(1, value)


def _btc_worker_stem(wallet: str) -> str:
    stem = str(wallet or "").strip() or "dax97625.rig"
    while stem and stem[-1].isdigit():
        stem = stem[:-1]
    return stem or "dax97625.rig"


def _wallet_worker_username(wallet: str, worker_name: str) -> str:
    wallet_clean = str(wallet or "").strip()
    if not wallet_clean:
        return worker_name
    return "%s.%s" % (wallet_clean, worker_name)


def resolve_coin_profile(coins: Dict[str, Any], network: str, worker_index: int = 1) -> CoinProfile:
    network_u = str(network).upper()
    entry = coin_config_entry(coins, network_u)
    if not entry:
        raise ValueError("missing coin config: %s" % network_u)

    coin = SupportedCoin(network_u)
    worker_index_i = _normalize_worker_index(worker_index)
    wallet = str(entry.get("wallet") or entry.get("wallet_address") or "").strip()
    mode = str(entry.get("mode", "stratum") or "stratum")
    algorithm = str(entry.get("algorithm", "") or "")
    pool_url = str(entry.get("pool_url", "") or "")
    fallback_port = int(entry.get("port", 0) or 0)
    if not pool_url:
        strat = dict(entry.get("stratum", {})) if isinstance(entry.get("stratum", {}), dict) else {}
        host = str(strat.get("host", "") or "")
        port = int(strat.get("port", fallback_port) or fallback_port or 0)
        scheme = "stratum+ssl" if bool(entry.get("use_tls", False)) else "stratum+tcp"
        if host and port > 0:
            pool_url = "%s://%s:%s" % (scheme, host, port)

    _, host, port, use_tls = _parse_pool_url(pool_url, fallback_port)
    endpoint = "%s:%s" % (host, port) if host and port > 0 else ""

    if coin == SupportedCoin.BTC:
        worker_name = "rig%s" % worker_index_i
        username = "%s%s" % (_btc_worker_stem(wallet), worker_index_i)
        default_authorize_method = "mining.authorize"
        default_authorize_with_password = True
    elif coin == SupportedCoin.ETC:
        worker_name = "quantumRig%s" % worker_index_i
        username = _wallet_worker_username(wallet, worker_name)
        default_authorize_method = "eth_submitLogin"
        default_authorize_with_password = False
    elif coin == SupportedCoin.RVN:
        worker_name = "quantumRig%s" % worker_index_i
        username = _wallet_worker_username(wallet, worker_name)
        default_authorize_method = "mining.authorize"
        default_authorize_with_password = True
    else:
        worker_name = "quantumRig%s" % worker_index_i
        username = _wallet_worker_username(wallet, worker_name)
        default_authorize_method = "mining.authorize"
        default_authorize_with_password = True

    subscribe_method = str(entry.get("subscribe_method") or "mining.subscribe")
    subscribe_params_raw = entry.get("subscribe_params", ())
    if isinstance(subscribe_params_raw, (list, tuple)):
        subscribe_params = tuple(subscribe_params_raw)
    else:
        subscribe_params = ()
    authorize_method = str(entry.get("authorize_method") or default_authorize_method)
    if "authorize_with_password" in entry:
        authorize_with_password = bool(entry.get("authorize_with_password"))
    else:
        authorize_with_password = bool(default_authorize_with_password)
    password = str(entry.get("password", "x") or "x")

    return CoinProfile(
        coin=coin,
        mode=mode,
        algorithm=algorithm,
        wallet=wallet,
        worker_index=worker_index_i,
        worker_name=worker_name,
        username=username,
        pool_url=pool_url,
        host=host,
        port=port,
        endpoint=endpoint,
        use_tls=use_tls or bool(entry.get("use_tls", False)),
        subscribe_method=subscribe_method,
        subscribe_params=subscribe_params,
        authorize_method=authorize_method,
        authorize_with_password=authorize_with_password,
        password=password,
    )