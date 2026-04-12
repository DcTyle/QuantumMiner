from __future__ import annotations

from pathlib import Path
import argparse
import json

from photonic_identity_backbone import (
    RESEARCH_ROOT,
    analyze_photonic_identity_backbone,
    load_default_inputs,
    write_run45_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Run45 photonic identity research artifacts.")
    parser.add_argument(
        "--output-dir",
        default=str(RESEARCH_ROOT / "Run45"),
        help="Output directory for Run45 artifact files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_default_inputs()
    analysis = analyze_photonic_identity_backbone(
        frames=list(bundle.get("frames", []) or []),
        temporal_schema=dict(bundle.get("temporal_schema", {}) or {}),
        process_schema=dict(bundle.get("process_schema", {}) or {}),
        live_ledger=dict(bundle.get("live_ledger", {}) or {}),
        nist_reference=dict(bundle.get("nist_reference", {}) or {}),
    )
    paths = write_run45_artifacts(Path(args.output_dir), analysis)
    payload = {
        "ok": True,
        "output_dir": str(Path(args.output_dir)),
        "artifacts": paths,
        "summary": dict(analysis.get("summary", {}) or {}),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())