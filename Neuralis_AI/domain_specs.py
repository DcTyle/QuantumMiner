from typing import Any, Dict, Callable, Optional, Tuple


class DomainSpec:
    def __init__(self,
                 *,
                 defaults: Dict[str, Any],
                 allowed: Dict[str, str],
                 validators: Optional[Dict[str, Callable[[Any], bool]]] = None) -> None:
        self.defaults = dict(defaults)
        self.allowed = dict(allowed)
        self.validators = dict(validators or {})

    def normalize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(self.defaults)
        for k, t in self.allowed.items():
            if k not in payload:
                continue
            v = payload.get(k)
            try:
                if t == "str":
                    v = str(v)
                elif t == "int":
                    v = int(v)
                elif t == "float":
                    v = float(v)
                elif t == "bool":
                    v = bool(v)
                else:
                    v = str(v)
            except Exception:
                continue
            ok = True
            if k in self.validators:
                try:
                    ok = bool(self.validators[k](v))
                except Exception:
                    ok = False
            if ok:
                out[k] = v
        return out


def _in_set(options: Tuple[str, ...]) -> Callable[[Any], bool]:
    def _f(x: Any) -> bool:
        try:
            return str(x) in options
        except Exception:
            return False
    return _f


def _ge_int(n: int) -> Callable[[Any], bool]:
    def _f(x: Any) -> bool:
        try:
            return int(x) >= n
        except Exception:
            return False
    return _f


DOMAIN_REGISTRY: Dict[str, DomainSpec] = {
    "systemcognition": DomainSpec(
        defaults={
            "last_tick_ts": 0.0,
            "tick_source": "bios.scheduler",
            "notes": "system scope tick",
        },
        allowed={
            "task": "str",
            "phase": "str",
            "seq": "int",
        },
        validators={
            "phase": _in_set(("boot", "tick", "shutdown", "idle")),
            "seq": _ge_int(0),
        },
    ),
    "neural network": DomainSpec(
        defaults={
            "last_tick_ts": 0.0,
            "tick_source": "bios.scheduler",
            "notes": "neuralis cognition tick",
        },
        allowed={
            "task": "str",
            "phase": "str",
            "seq": "int",
        },
        validators={
            "phase": _in_set(("tick", "learn", "sync")),
            "seq": _ge_int(0),
        },
    ),
    "boundary cognition": DomainSpec(
        defaults={
            "last_tick_ts": 0.0,
            "tick_source": "bios.scheduler",
            "notes": "boundary evaluation tick",
        },
        allowed={
            "task": "str",
            "phase": "str",
            "seq": "int",
            "policy": "str",
        },
        validators={
            "phase": _in_set(("tick", "audit")),
            "seq": _ge_int(0),
            "policy": _in_set(("strict", "advisory")),
        },
    ),
}


def get_domain_spec(name: str) -> Optional[DomainSpec]:
    return DOMAIN_REGISTRY.get(name)
