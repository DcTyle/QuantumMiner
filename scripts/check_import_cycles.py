# ASCII-ONLY
"""
Lightweight import/dependency checker for internal modules.

Usage:
    python3 scripts/check_import_cycles.py [--strict] [--focus PKG[,PKG2,...]] \
                                                                                 [--check-import NAME ...] [--forbid-import NAME ...]

Default: prints detected cycles (if any) and exits 0.
--strict: exits 1 when cycles are detected or when a forbidden import is found.
--focus: only report cycles that touch any of the listed top-level packages.
--check-import: print modules that import NAME (multiple allowed).
--forbid-import: fail if any module imports NAME (multiple allowed).
"""
from __future__ import annotations

import os
import sys
import ast
from typing import Dict, Set, List, Tuple
import argparse

ROOTS = {"bios", "VHW", "miner", "core", "Neuralis_AI", "prediction_engine", "Control_Center"}


def is_py(path: str) -> bool:
    return path.endswith(".py") and not os.path.basename(path).startswith("__pycache__")


def module_name_from_path(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    # drop empty segments
    parts = [p for p in parts if p and p != "__pycache__"]
    return ".".join(parts)


def collect_files(root: str) -> List[str]:
    out: List[str] = []
    for d, _sub, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            if is_py(p):
                out.append(p)
    return out


def parse_imports(py_path: str) -> Set[str]:
    try:
        with open(py_path, "r", encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
        tree = ast.parse(src, filename=py_path)
    except Exception:
        return set()
    refs: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                name = (n.name or "").split(".")[0]
                if name in ROOTS:
                    refs.add(name)
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in ROOTS:
                refs.add(mod)
    return refs


def build_graph(workspace: str) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    files = collect_files(workspace)
    modmap: Dict[str, str] = {}
    graph: Dict[str, Set[str]] = {}
    for p in files:
        mod = module_name_from_path(workspace, p)
        modmap[mod] = p
        graph.setdefault(mod, set())
    for mod, p in modmap.items():
        refs = parse_imports(p)
        for r in refs:
            # record edge to top-level package if referenced
            graph[mod].add(r)
    return graph, modmap


def find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    seen: Set[str] = set()
    stack: Set[str] = set()
    cycles: List[List[str]] = []

    def dfs(u: str, path: List[str]) -> None:
        if u in stack:
            try:
                i = path.index(u)
                cycles.append(path[i:] + [u])
            except Exception:
                pass
            return
        if u in seen:
            return
        seen.add(u)
        stack.add(u)
        for v in graph.get(u, ()):  # type: ignore
            # only follow edges to internal packages or concrete modules we have
            if v in ROOTS or v in graph:
                dfs(v, path + [u])
        stack.remove(u)

    for m in list(graph.keys()):
        if m not in seen:
            dfs(m, [])
    # de-duplicate cycle permutations
    norm: List[List[str]] = []
    seen_sig: Set[str] = set()
    for cyc in cycles:
        sig = "->".join(cyc)
        if sig not in seen_sig:
            seen_sig.add(sig)
            norm.append(cyc)
    return norm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--focus", type=str, default="")
    p.add_argument("--check-import", dest="check", action="append", default=[])
    p.add_argument("--forbid-import", dest="forbid", action="append", default=[])
    p.add_argument("--help", action="help")
    return p.parse_args()


def _scan_import_usage(workspace: str, names: List[str]) -> Dict[str, List[str]]:
    files = collect_files(workspace)
    hits: Dict[str, List[str]] = {n: [] for n in names}
    for p in files:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                src = fh.read()
        except Exception:
            continue
        try:
            tree = ast.parse(src, filename=p)
        except Exception:
            continue
        imported: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    base = (n.name or "").split(".")[0]
                    imported.add(base)
            elif isinstance(node, ast.ImportFrom):
                base = (node.module or "").split(".")[0]
                if base:
                    imported.add(base)
        for name in names:
            if name in imported:
                hits[name].append(p)
    return hits


def main() -> int:
    args = _parse_args()
    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    graph, _ = build_graph(workspace)

    # 1) Cycle detection
    cycles = find_cycles(graph)
    if args.focus:
        focus = set([s.strip() for s in args.focus.split(",") if s.strip()])
        if focus:
            filtered: List[List[str]] = []
            for cyc in cycles:
                if any((node.split(".")[0] in focus) for node in cyc):
                    filtered.append(cyc)
            cycles = filtered
    if cycles:
        print("[import-cycles] Detected cycles:")
        for c in cycles:
            print("  - ", " -> ".join(c))
    else:
        print("[import-cycles] No cycles detected")

    # 2) Import usage checks
    rc = 0
    if args.check or args.forbid:
        names = list(set((args.check or []) + (args.forbid or [])))
        hits = _scan_import_usage(workspace, names)
        for name in names:
            files = hits.get(name, [])
            if files:
                print(f"[import-usage] '{name}' imported by:")
                for f in files:
                    print("  -", os.path.relpath(f, workspace))
            else:
                print(f"[import-usage] '{name}' not imported")
        # forbids
        for name in args.forbid or []:
            if hits.get(name) and args.strict:
                rc = 1

    if args.strict and cycles:
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
