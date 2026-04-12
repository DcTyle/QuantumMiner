from typing import Any, Dict, List

from VHW.vsd_manager import VSDManager

CONTEXT_GRAPH_PATH = "neuralis/packets/context_graph/v1/current"


def build_context_graph() -> Dict[str, Any]:
    """Return a static context graph for Neuralis subsystems.

    This is a read-only description of how Neuralis relates to the
    major folders. It does not scan the filesystem dynamically.
    """
    nodes: List[Dict[str, Any]] = [
        {"id": "Neuralis_AI", "label": "Neuralis_AI", "coherence": 1.0},
        {"id": "VHW", "label": "VHW", "coherence": 0.9},
        {"id": "bios", "label": "bios", "coherence": 0.7},
        {"id": "Control_Center", "label": "Control_Center", "coherence": 0.6},
        {"id": "core", "label": "core", "coherence": 0.5},
        {"id": "prediction_engine", "label": "prediction_engine", "coherence": 0.4},
        {"id": "miner", "label": "miner", "coherence": 0.3},
    ]
    edges: List[Dict[str, Any]] = [
        {"source": "Neuralis_AI", "target": "VHW", "kind": "vsd"},
        {"source": "bios", "target": "Neuralis_AI", "kind": "scheduler"},
        {"source": "Control_Center", "target": "Neuralis_AI", "kind": "ui"},
        {"source": "prediction_engine", "target": "VHW", "kind": "vsd"},
        {"source": "miner", "target": "VHW", "kind": "vsd"},
    ]
    return {"nodes": nodes, "edges": edges}


def store_context_graph(vsd: VSDManager | None = None) -> Dict[str, Any]:
    if vsd is None:
        try:
            vsd = VSDManager.global_instance()  # type: ignore[attr-defined]
        except Exception:
            vsd = None
    graph = build_context_graph()
    if vsd is not None:
        try:
            vsd.store(CONTEXT_GRAPH_PATH, graph)
        except Exception:
            pass
    return graph
