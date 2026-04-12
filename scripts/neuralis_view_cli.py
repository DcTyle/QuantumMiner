#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from typing import Any

from VHW.vsd_manager import VSDManager
from frontend.neuralis_view import (
    get_context_graph,
    get_cognition_capabilities,
    get_cognition_summary,
    get_all_domain_history,
)


def _get_vsd() -> Any:
    try:
        return VSDManager.global_instance()  # type: ignore[attr-defined]
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuralis view CLI (virtual documents from VSD)")
    parser.add_argument(
        "what",
        choices=["graph", "capabilities", "summary", "history"],
        help="Which virtual document to print",
    )
    parser.add_argument("--limit", type=int, default=16, help="History limit per domain (for 'history')")
    args = parser.parse_args()

    vsd = _get_vsd()

    if args.what == "graph":
        obj = get_context_graph(vsd)
    elif args.what == "capabilities":
        obj = get_cognition_capabilities(vsd)
    elif args.what == "summary":
        obj = get_cognition_summary(vsd)
    else:
        obj = get_all_domain_history(vsd, limit=args.limit)

    try:
        print(json.dumps(obj, indent=2, sort_keys=True))
    except Exception:
        print(str(obj))


if __name__ == "__main__":
    main()
