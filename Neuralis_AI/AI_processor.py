# ============================================================================
# Quantum Application / Neuralis_Ai
# ASCII-ONLY SOURCE FILE
# File: AI_processor.py
# Version: v4.9.1 (Delta-Fusion Super-Compression + Legacy VectorCodec)
# Jarvis ADA v4.7 Hybrid Ready
# ============================================================================
"""
Purpose
-------
Central pack / rehydrate engine for QuantumApplication.

Responsibilities
----------------
1) Enumerate project files and build a deterministic "fullstate" JSON manifest.
2) Optionally generate a single fused super-vector text file using:
       - ascii_floatmap_v1
       - delta-fusion on int-domain vectors
       - zlib level 9 + base85 for ASCII-only compression
3) Provide a rehydration path that:
       - Reads the fullstate manifest.
       - Optionally reconstructs all files from the fused vector.

Compatibility
-------------
- Preserves the legacy VectorEncoder / VectorDecoder interface so that
  core.init and other modules can safely import these symbols:
      from core.AI_processor import VectorEncoder, VectorDecoder
  These are implemented as thin wrappers around the Dehydrator / Rehydrator.

- Emits events on the BIOS EventBus:
      "ai_processor.pack_complete"
      "ai_processor.rehydrate_complete"

- Uses VSDManager for "last pack/load path" hints, but degrades gracefully
  to a no-op stub if VSDManager is unavailable.

CLI
---
python -m core.AI_processor pack  <root> <out_json> [--mem "<text>"] [--super]
python -m core.AI_processor load  <fullstate.json>
python -m core.AI_processor rehydrate-super <vector.txt> <manifest.json> [out_dir]

Super Mode Artifacts
--------------------
- Manifest JSON:     <base>.json  (header + list of files + vlen per file)
- Fused vector txt:  <base>_super.txt  (ASCII-only compressed content)
"""

from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple, Optional, List
import os
import io
import sys
import json
import time
import logging
import hashlib
import zlib
import base64

# ----------------------------------------------------------------------------
# Structured logging
# ----------------------------------------------------------------------------
logger = logging.getLogger("core.AI_processor")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _fmt = logging.Formatter(
        fmt="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    _handler.setFormatter(_fmt)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# Safe imports (guarded) for VSD, config, EventBus, and rate limiting
# ----------------------------------------------------------------------------
try:
    from VHW.vsd_manager import VSDManager
except Exception:
    class VSDManager:  # type: ignore
        def get(self, key: str, default: Any = None) -> Any:
            return default
        def store(self, key: str, value: Any) -> None:
            return

try:
    from bios.event_bus import get_event_bus
except Exception:
    def get_event_bus():
        class _NoBus:
            def publish(self, event: str, data: Dict[str, Any]) -> None:
                return
        return _NoBus()

try:
    from VHW.rate_limiter import RateLimiter
    _api_limiter = RateLimiter(max_calls=100, window_seconds=60)
except Exception:
    class _DummyLimiter:
        def allow(self) -> bool:
            return True
    _api_limiter = _DummyLimiter()

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------
def _now_utc_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _file_iter(root: str) -> Iterable[Tuple[str, int, str, str]]:
    """
    Yield (rel_path, size_bytes, sha256_hex, abs_path) for files under root.
    Skips .git, __pycache__, and IDE transient folders.
    """
    root = os.path.abspath(root)
    skip_dirs = {".git", "__pycache__", ".vs", ".idea", ".vscode"}
    for dirpath, dnames, fnames in os.walk(root):
        dnames[:] = [d for d in dnames if d not in skip_dirs]
        for fn in fnames:
            abs_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(abs_path, root)
            rel_norm = rel.replace("\\", "/")
            try:
                st = os.stat(abs_path)
                size = int(st.st_size)
                h = hashlib.sha256()
                with open(abs_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        if not chunk:
                            break
                        h.update(chunk)
                sha_hex = h.hexdigest()
                yield (rel_norm, size, sha_hex, abs_path)
            except Exception:
                logger.exception("Cannot stat/hash file: %s", abs_path)

def _stream_json(fp: io.TextIOBase, obj: Dict[str, Any]) -> None:
    """
    Streaming JSON writer to keep memory usage bounded.
    Uses compact separators but ensures ASCII-only output.
    """
    encoder = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"), indent=None)
    for chunk in encoder.iterencode(obj):
        fp.write(chunk)

# ----------------------------------------------------------------------------
# ascii_floatmap_v1 encode / decode helpers
# (byte stream -> float[-1,1] -> ASCII vector 32..126 -> back)
# ----------------------------------------------------------------------------
ASCII_MIN = 32
ASCII_MAX = 126
ASCII_RANGE = ASCII_MAX - ASCII_MIN + 1  # should be 95

def _bytes_to_floats(b: bytes) -> List[float]:
    # map [0..255] -> [-1.0..1.0]
    return [(x / 127.5) - 1.0 for x in b]

def _floats_to_ascii(vals: List[float]) -> str:
    # map float [-1,1] -> ascii 32..126
    out: List[str] = []
    for v in vals:
        if v < -1.0:
            v = -1.0
        if v > 1.0:
            v = 1.0
        code = int((v + 1.0) * 47.0) + ASCII_MIN
        if code < ASCII_MIN:
            code = ASCII_MIN
        if code > ASCII_MAX:
            code = ASCII_MAX
        out.append(chr(code))
    return "".join(out)

def _ascii_to_floats(s: str) -> List[float]:
    # map ascii 32..126 -> float [-1,1]
    vals: List[float] = []
    for ch in s:
        code = ord(ch)
        if code < ASCII_MIN:
            code = ASCII_MIN
        if code > ASCII_MAX:
            code = ASCII_MAX
        vals.append(((code - ASCII_MIN) / 47.0) - 1.0)
    return vals

def _floats_to_bytes(vals: List[float]) -> bytes:
    # map float [-1,1] -> [0..255]
    buf: List[int] = []
    for v in vals:
        if v < -1.0:
            v = -1.0
        if v > 1.0:
            v = 1.0
        q = int((v + 1.0) * 127.5)
        if q < 0:
            q = 0
        if q > 255:
            q = 255
        buf.append(q)
    return bytes(buf)

def _vector_to_ints(vec: str) -> List[int]:
    # ascii 32..126 -> [0..94]
    return [ord(ch) - ASCII_MIN for ch in vec]

def _ints_to_vector(vals: List[int]) -> str:
    # [0..94] -> ascii 32..126
    chars: List[str] = []
    for v in vals:
        if v < 0:
            v = 0
        if v >= ASCII_RANGE:
            v = ASCII_RANGE - 1
        chars.append(chr(v + ASCII_MIN))
    return "".join(chars)

# ----------------------------------------------------------------------------
# Delta fusion helpers
# ----------------------------------------------------------------------------
def _delta_encode(prev: List[int], curr: List[int]) -> List[int]:
    """
    Encode curr relative to prev into 0..94 delta codes.

    For i < len(prev):
        delta[i] = (curr[i] - prev[i]) mod 95

    For tail:
        remaining curr values are appended as-is.
    """
    limit = min(len(prev), len(curr))
    out: List[int] = []
    for i in range(limit):
        d = (curr[i] - prev[i]) % ASCII_RANGE
        out.append(d)
    out.extend(curr[limit:])
    return out

def _delta_decode(prev: List[int], delta: List[int]) -> List[int]:
    """
    Invert _delta_encode.

    For i < len(prev):
        curr[i] = (prev[i] + delta[i]) mod 95

    For tail:
        curr[i] = delta[i].
    """
    limit = min(len(prev), len(delta))
    out: List[int] = []
    for i in range(limit):
        c = (prev[i] + delta[i]) % ASCII_RANGE
        out.append(c)
    out.extend(delta[limit:])
    return out

# ----------------------------------------------------------------------------
# Compression helpers (zlib + base85 for ASCII-only vector text)
# ----------------------------------------------------------------------------
def _compress_ascii(s: str) -> str:
    raw = s.encode("ascii", errors="strict")
    comp = zlib.compress(raw, 9)
    return base64.b85encode(comp).decode("ascii")

def _decompress_ascii(s: str) -> str:
    comp = base64.b85decode(s.encode("ascii"))
    raw = zlib.decompress(comp)
    return raw.decode("ascii")

def _encode_file_super(abs_path: str) -> Tuple[str, str, int]:
    """
    Encode a file into ascii_floatmap_v1 space.

    Returns:
        sha256_raw_bytes (hex string)
        ascii_vector     (str)
        vlen_ints        (int) length of int-domain vector
    """
    with open(abs_path, "rb") as f:
        b = f.read()
    sha_raw = hashlib.sha256(b).hexdigest()
    floats = _bytes_to_floats(b)
    vec = _floats_to_ascii(floats)
    ints = _vector_to_ints(vec)
    return sha_raw, vec, len(ints)

# ----------------------------------------------------------------------------
# Dehydrator: pack project -> JSON fullstate (+ optional super vector)
# ----------------------------------------------------------------------------
class Dehydrator:
    """
    Dehydrator
    ----------
    Scans a root directory, builds a deterministic manifest, and optionally
    generates a fused vector text file using delta fusion.

    The main entry point is:

        pack(root_path, out_json_path, memory_text="", super_compress=False)
    """

    def __init__(self) -> None:
        self.vsd = VSDManager()
        self.bus = get_event_bus()
        self.config = _cfg

    def pack(
        self,
        root_path: str,
        out_json_path: str,
        memory_text: str = "",
        super_compress: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.time()
        root_abs = os.path.abspath(root_path)

        header: Dict[str, Any] = {
            "format": "json_statevector_v2",
            "encoding": "ascii_floatmap_v1",
            "timestamp": _now_utc_str(),
            "root": root_abs.replace("\\", "/"),
        }

        files_meta: List[Dict[str, Any]] = []
        total_bytes = 0
        total_files = 0

        # For super mode
        prev_ints: List[int] = []
        fused_ints: List[int] = []

        for rel, size, sha, abs_path in _file_iter(root_abs):
            total_files += 1
            total_bytes += size
            entry: Dict[str, Any] = {
                "path": rel,
                "bytes": size,
                "sha256": sha,
            }

            if super_compress:
                try:
                    sha_raw, vec, vlen = _encode_file_super(abs_path)
                    curr_ints = _vector_to_ints(vec)
                    delta = _delta_encode(prev_ints, curr_ints)
                    fused_ints.extend(delta)
                    prev_ints = curr_ints

                    entry["sha256_content"] = sha_raw
                    entry["vlen"] = vlen
                except Exception:
                    logger.warning("Super encode failed for %s", abs_path)

            files_meta.append(entry)

        container: Dict[str, Any] = {
            "header": dict(header, files_count=total_files, bytes_total=total_bytes),
            "memory_snapshot": memory_text or "",
            "files": files_meta,
        }

        out_abs = os.path.abspath(out_json_path)
        out_dir = os.path.dirname(out_abs)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        vector_txt_path = ""
        if super_compress:
            fused_vec = _ints_to_vector(fused_ints)
            compressed_vec = _compress_ascii(fused_vec)
            base = os.path.splitext(out_abs)[0]
            vector_txt_path = base + "_super.txt"
            try:
                with open(vector_txt_path, "w", encoding="ascii") as vf:
                    vf.write(compressed_vec)
            except Exception:
                logger.exception("Failed to write fused vector: %s", vector_txt_path)
                return {
                    "ok": False,
                    "error": "vector_write_failed",
                    "path": vector_txt_path,
                }

            # Upgrade header for super mode
            container["header"]["format"] = "json_statevector_v3"
            container["header"]["encoding"] = "ascii_floatmap_v1+delta+b85_zlib"
            container["header"]["vector_file"] = os.path.basename(vector_txt_path)
            container["header"]["fused_vector_len"] = len(fused_vec)
            container["header"]["compressed_len"] = len(compressed_vec)
            container["header"]["vector_sha256"] = hashlib.sha256(
                compressed_vec.encode("ascii")
            ).hexdigest()

        # Write JSON manifest
        try:
            with open(out_abs, "w", encoding="utf-8") as fp:
                _stream_json(fp, container)
                fp.write("\n")
        except Exception:
            logger.exception("Failed to write JSON fullstate: %s", out_abs)
            return {
                "ok": False,
                "error": "write_failed",
                "path": out_abs,
            }

        dt = time.time() - t0
        logger.info(
            "Dehydrator.pack wrote %d files (%.2f MB) to %s in %.3fs",
            total_files,
            total_bytes / (1024.0 * 1024.0),
            out_abs,
            dt,
        )

        # EventBus notification
        try:
            payload: Dict[str, Any] = {
                "ts": time.time(),
                "files_count": total_files,
                "bytes_total": total_bytes,
                "path": out_abs,
            }
            if vector_txt_path:
                payload["vector"] = vector_txt_path
            self.bus.publish("ai_processor.pack_complete", payload)
        except Exception:
            logger.warning("EventBus publish failed for pack_complete")

        # Persist VSD hint
        try:
            self.vsd.store("ai_processor/last_pack_path", out_abs)
        except Exception as e:
            logger.warning(f"Failed to store pack path in VSD: {e}")

        result: Dict[str, Any] = {
            "ok": True,
            "path": out_abs,
            "files_count": total_files,
            "bytes_total": total_bytes,
            "dt_s": dt,
        }
        if vector_txt_path:
            result["vector"] = vector_txt_path
        return result

# ----------------------------------------------------------------------------
# Rehydrator: read fullstate manifest, reconstruct files from vector
# ----------------------------------------------------------------------------
class Rehydrator:
    """
    Rehydrator
    ----------
    Reads a fullstate JSON manifest and provides:
      - load(fullstate_path) -> metadata
      - rehydrate_super(vector_path, manifest_json_path, out_root)
    """

    def __init__(self) -> None:
        self.vsd = VSDManager()
        self.bus = get_event_bus()

    def load(self, fullstate_path: str) -> Dict[str, Any]:
        t0 = time.time()
        p = os.path.abspath(fullstate_path)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            logger.exception("Failed to read fullstate JSON: %s", p)
            return {
                "ok": False,
                "error": "read_failed",
                "path": p,
            }

        header = data.get("header", {})
        files_meta = data.get("files", [])
        mem_snapshot = data.get("memory_snapshot", "")

        files_count = int(header.get("files_count", len(files_meta)))
        bytes_total = int(header.get("bytes_total", 0))

        container_sha256 = ""
        try:
            h = hashlib.sha256()
            with open(p, "rb") as rf:
                for chunk in iter(lambda: rf.read(1024 * 1024), b""):
                    if not chunk:
                        break
                    h.update(chunk)
            container_sha256 = h.hexdigest()
        except Exception:
            logger.warning("Could not compute container sha256 for %s", p)

        dt = time.time() - t0
        logger.info(
            "Rehydrator.load read %d files (%.2f MB) from %s in %.3fs",
            files_count,
            bytes_total / (1024.0 * 1024.0),
            p,
            dt,
        )

        # EventBus notification
        try:
            self.bus.publish(
                "ai_processor.rehydrate_complete",
                {
                    "ts": time.time(),
                    "files_count": files_count,
                    "bytes_total": bytes_total,
                    "path": p,
                },
            )
        except Exception:
            logger.warning("EventBus publish failed for rehydrate_complete")

        # VSD hint
        try:
            self.vsd.store("ai_processor/last_load_path", p)
        except Exception as e:
            logger.warning(f"Failed to store load path in VSD: {e}")

        return {
            "ok": True,
            "path": p,
            "header": header,
            "files_count": files_count,
            "bytes_total": bytes_total,
            "memory_present": bool(mem_snapshot),
            "container_sha256": container_sha256,
            "dt_s": dt,
        }

    def rehydrate_super(
        self,
        vector_path: str,
        manifest_json_path: str,
        out_root: str,
    ) -> Dict[str, Any]:
        """
        Rebuild files from a single fused vector using per-file vlen slices.

        vector_path         : path to <base>_super.txt (ascii b85+zlib)
        manifest_json_path  : path to fullstate manifest json
        out_root            : root directory to write restored files
        """
        t0 = time.time()
        vector_abs = os.path.abspath(vector_path)
        manifest_abs = os.path.abspath(manifest_json_path)
        out_root_abs = os.path.abspath(out_root)

        # Read and decompress fused vector
        try:
            with open(vector_abs, "r", encoding="ascii") as vf:
                compressed = vf.read()
            fused_vec = _decompress_ascii(compressed)
            fused_ints = _vector_to_ints(fused_vec)
        except Exception:
            logger.exception("Failed to read or decompress fused vector: %s", vector_abs)
            return {
                "ok": False,
                "error": "vector_read_failed",
                "vector": vector_abs,
            }

        # Read manifest json
        try:
            with open(manifest_abs, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
        except Exception:
            logger.exception("Failed to read manifest JSON: %s", manifest_abs)
            return {
                "ok": False,
                "error": "manifest_read_failed",
                "manifest": manifest_abs,
            }

        files_meta: List[Dict[str, Any]] = manifest.get("files", [])
        if not os.path.exists(out_root_abs):
            os.makedirs(out_root_abs, exist_ok=True)

        prev: List[int] = []
        pos = 0
        restored = 0

        for m in files_meta:
            rel = str(m.get("path", ""))
            vlen = int(m.get("vlen", 0))
            if vlen <= 0:
                logger.warning("Missing vlen for file; cannot restore reliably: %s", rel)
                continue

            if pos + vlen > len(fused_ints):
                logger.warning(
                    "Fused vector exhausted while restoring %s; pos=%d vlen=%d total=%d",
                    rel,
                    pos,
                    vlen,
                    len(fused_ints),
                )
                break

            delta_slice = fused_ints[pos:pos + vlen]
            pos += vlen

            curr = _delta_decode(prev, delta_slice)
            prev = curr

            vec = _ints_to_vector(curr)
            floats = _ascii_to_floats(vec)
            raw = _floats_to_bytes(floats)

            abs_out = os.path.join(out_root_abs, rel)
            out_dir = os.path.dirname(abs_out)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            try:
                with open(abs_out, "wb") as wf:
                    wf.write(raw)
                restored += 1
            except Exception:
                logger.exception("Failed writing restored file: %s", abs_out)

        dt = time.time() - t0
        logger.info(
            "Rehydrated %d/%d files into %s in %.3fs",
            restored,
            len(files_meta),
            out_root_abs,
            dt,
        )

        return {
            "ok": True,
            "restored": restored,
            "total": len(files_meta),
            "out_root": out_root_abs,
            "dt_s": dt,
        }

# ----------------------------------------------------------------------------
# Legacy VectorCodec interface (compat shim)
# ----------------------------------------------------------------------------
class VectorEncoder:
    """
    Legacy compatibility shim for older code expecting VectorEncoder.

    Under the hood, this delegates to Dehydrator.pack.
    Only a minimal method set is provided so core.init and any legacy
    callers can import this symbol safely.
    """

    def __init__(self) -> None:
        self._dehydrator = Dehydrator()

    def pack_project(
        self,
        root_path: str,
        out_json_path: str,
        memory_text: str = "",
        super_compress: bool = False,
    ) -> Dict[str, Any]:
        """
        Thin wrapper around Dehydrator.pack.
        """
        return self._dehydrator.pack(
            root_path=root_path,
            out_json_path=out_json_path,
            memory_text=memory_text,
            super_compress=super_compress,
        )

class VectorDecoder:
    """
    Legacy compatibility shim for older code expecting VectorDecoder.

    Delegates to Rehydrator.load and rehydrator.rehydrate_super.
    """

    def __init__(self) -> None:
        self._rehydrator = Rehydrator()

    def load_manifest(self, fullstate_path: str) -> Dict[str, Any]:
        return self._rehydrator.load(fullstate_path)

    def rehydrate_super(
        self,
        vector_path: str,
        manifest_json_path: str,
        out_root: str,
    ) -> Dict[str, Any]:
        return self._rehydrator.rehydrate_super(
            vector_path=vector_path,
            manifest_json_path=manifest_json_path,
            out_root=out_root,
        )

# ----------------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------------
def _cli_pack(argv: List[str]) -> int:
    """
    pack <root> <out_json> [--mem "<text>"] [--super]
    """
    if len(argv) < 2:
        print("Usage: python -m core.AI_processor pack <root> <out_json> [--mem \"<text>\"] [--super]")
        return 1

    root = argv[0]
    out_json = argv[1]
    mem_text = ""
    super_mode = False

    i = 2
    while i < len(argv):
        arg = argv[i]
        if arg == "--mem" and i + 1 < len(argv):
            mem_text = argv[i + 1]
            i += 2
            continue
        if arg == "--super":
            super_mode = True
            i += 1
            continue
        i += 1

    d = Dehydrator()
    res = d.pack(root, out_json, mem_text, super_mode)
    print(json.dumps(res, ensure_ascii=True))
    return 0 if res.get("ok") else 3

def _cli_load(argv: List[str]) -> int:
    """
    load <fullstate.json>
    """
    if len(argv) < 1:
        print("Usage: python -m core.AI_processor load <fullstate.json>")
        return 1

    p = argv[0]
    r = Rehydrator()
    res = r.load(p)
    print(json.dumps(res, ensure_ascii=True))
    return 0 if res.get("ok") else 4

def _cli_rehydrate_super(argv: List[str]) -> int:
    """
    rehydrate-super <vector.txt> <manifest.json> [out_dir]
    """
    if len(argv) < 2:
        print("Usage: python -m core.AI_processor rehydrate-super <vector.txt> <manifest.json> [out_dir]")
        return 1

    vector = argv[0]
    manifest = argv[1]
    out_dir = argv[2] if len(argv) > 2 else "./rehydrated"

    r = Rehydrator()
    res = r.rehydrate_super(vector, manifest, out_dir)
    print(json.dumps(res, ensure_ascii=True))
    return 0 if res.get("ok") else 5

# ----------------------------------------------------------------------------
# Module entrypoint
# ----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for module execution.

    Commands:
        pack             -> build manifest (and optional super vector)
        load             -> read manifest and print metadata
        rehydrate-super  -> rebuild files from fused vector + manifest
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Commands: pack, load, rehydrate-super")
        return 1

    cmd = args[0].strip().lower()
    rest = args[1:]

    try:
        if cmd == "pack":
            return _cli_pack(rest)
        if cmd == "load":
            return _cli_load(rest)
        if cmd in {"rehydrate-super", "rehydrate"}:
            return _cli_rehydrate_super(rest)
        print("Unknown command:", cmd)
        return 1
    except SystemExit:
        raise
    except Exception:
        logger.exception("AI_processor CLI failure")
        return 10

if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
__all__ = [
    "Dehydrator",
    "Rehydrator",
    "VectorEncoder",
    "VectorDecoder",
    "main",
]
