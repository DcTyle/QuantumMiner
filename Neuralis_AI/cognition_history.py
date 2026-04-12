from typing import Any, Dict, List, Optional

HISTORY_ROOT = "neuralis/packets/cognition_capabilities/v1/history"


def _path(domain: str) -> str:
    return f"{HISTORY_ROOT}/{domain}"


def append_history(vsd: Any, domain: str, entry: Dict[str, Any], maxlen: int = 64) -> List[Dict[str, Any]]:
    try:
        path = _path(domain)
        items = vsd.load(path) or []
        if not isinstance(items, list):
            items = []
        items.append(dict(entry))
        if len(items) > maxlen:
            items = items[-maxlen:]
        vsd.store(path, items)
        return items
    except Exception:
        return []


def get_history(vsd: Any, domain: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        items = vsd.load(_path(domain)) or []
        if not isinstance(items, list):
            return []
        if isinstance(limit, int) and limit >= 0:
            return items[-limit:] if limit > 0 else []
        return items
    except Exception:
        return []
