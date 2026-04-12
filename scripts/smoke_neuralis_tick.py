import time
from typing import Any, Dict

try:
    from bios.event_bus import get_event_bus  # preferred if available
except Exception:
    get_event_bus = None  # type: ignore

try:
    from VHW.vsd_manager import VSD as _GLOBAL_VSD
    from VHW.vsd_manager import VSDManager
except Exception:
    _GLOBAL_VSD = None  # type: ignore
    class VSDManager:  # type: ignore
        def __init__(self): self._m: Dict[str, Any] = {}
        def get(self, k: str, d: Any=None): return self._m.get(str(k), d)
        def store(self, k: str, v: Any): self._m[str(k)] = v
        def load(self, k: str): return self._m.get(str(k))

from Neuralis_AI.packet_cognition_capabilities import read_packet, write_packet

TOPIC = "scheduler.task.rollup"


def _get_bus():
    if get_event_bus is not None:
        try:
            return get_event_bus()
        except Exception:
            return None
    return None


def _publish(bus: Any, topic: str, payload: Dict[str, Any]) -> None:
    if bus is None:
        print("[warn] no event bus available; skipping publish")
        return
    try:
        bus.publish(topic, payload)
    except Exception as e:
        print(f"[warn] bus publish failed: {e}")


def main() -> None:
    vsd = _GLOBAL_VSD if _GLOBAL_VSD is not None else VSDManager()
    pkt = read_packet(vsd)
    if not pkt:
        pkt = write_packet(vsd)

    bus = _get_bus()
    seq = int(time.time()) % 100000
    payload = {"task": "heartbeat", "phase": "tick", "seq": seq}
    print(f"Publishing to {TOPIC}: {payload}")
    _publish(bus, TOPIC, payload)

    time.sleep(0.2)

    pkt2 = read_packet(vsd)
    dm = (((pkt2 or {}).get("meta", {}) or {}).get("domain_meta", {}) or {})
    print("domain_meta:", dm)


if __name__ == "__main__":
    main()
