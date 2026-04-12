from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_ROOT.parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from gpu_pulse_axis_dynamics import (
    build_live_telemetry_payload,
    build_surface_telemetry,
    choose_surface_quartet,
    predict_full_spectrum_calibration,
)


DEFAULT_SURFACE_PATH = RESEARCH_ROOT / "gpu_kernel_interference_prediction_surface.json"
DEFAULT_OUTPUT_PATH = RESEARCH_ROOT / "prototyping" / "output" / "live_telemetry_transport_prediction.json"
DEFAULT_FULL_SPECTRUM_OUTPUT_PATH = RESEARCH_ROOT / "prototyping" / "output" / "full_spectrum_calibration_prediction.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-only live telemetry transport predictor."
    )
    parser.add_argument(
        "--surface",
        default=str(DEFAULT_SURFACE_PATH),
        help="Path to a research prediction surface JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write the predicted telemetry payload JSON.",
    )
    parser.add_argument(
        "--full-spectrum-output",
        default=str(DEFAULT_FULL_SPECTRUM_OUTPUT_PATH),
        help="Path to write the directional full-spectrum calibration JSON.",
    )
    parser.add_argument(
        "--phase-turns",
        type=float,
        default=None,
        help="Optional current phase turns override.",
    )
    parser.add_argument(
        "--previous-phase-turns",
        type=float,
        default=None,
        help="Optional previous phase turns override.",
    )
    parser.add_argument(
        "--interval-count",
        type=int,
        default=6,
        help="Number of predictive firing intervals per kernel coordinate.",
    )
    parser.add_argument(
        "--kernel-grid-width",
        type=int,
        default=None,
        help="Optional kernel grid width override.",
    )
    parser.add_argument(
        "--kernel-grid-height",
        type=int,
        default=None,
        help="Optional kernel grid height override.",
    )
    parser.add_argument(
        "--kernel-interval-ms",
        type=float,
        default=None,
        help="Optional kernel interval override in milliseconds.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_phase_turns(surface: dict, args: argparse.Namespace) -> tuple[float, float]:
    best_prediction = dict(surface.get("best_prediction", {}) or {})
    predicted_phase = float(best_prediction.get("prediction_confidence", 0.5))
    current_phase = predicted_phase if args.phase_turns is None else float(args.phase_turns)
    previous_phase = (current_phase - 0.04) if args.previous_phase_turns is None else float(args.previous_phase_turns)
    current_phase = current_phase % 1.0
    previous_phase = previous_phase % 1.0
    return current_phase, previous_phase
def main() -> None:
    args = parse_args()
    surface_path = Path(args.surface).resolve()
    output_path = Path(args.output).resolve()
    full_spectrum_output_path = Path(args.full_spectrum_output).resolve()
    surface = load_json(surface_path)
    quartet = choose_surface_quartet(surface)
    phase_turns, previous_phase_turns = choose_phase_turns(surface, args)
    telemetry = build_surface_telemetry(surface)
    payload = build_live_telemetry_payload(
        quartet=quartet,
        phase_turns=phase_turns,
        previous_phase_turns=previous_phase_turns,
        telemetry=telemetry,
    )
    payload["source_surface"] = str(surface_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    full_spectrum_payload = predict_full_spectrum_calibration(
        surface=surface,
        phase_turns=phase_turns,
        previous_phase_turns=previous_phase_turns,
        interval_count=args.interval_count,
        kernel_grid_width=args.kernel_grid_width,
        kernel_grid_height=args.kernel_grid_height,
        kernel_interval_ms=args.kernel_interval_ms,
    )
    full_spectrum_payload["source_surface"] = str(surface_path)
    full_spectrum_output_path.parent.mkdir(parents=True, exist_ok=True)
    with full_spectrum_output_path.open("w", encoding="utf-8") as handle:
        json.dump(full_spectrum_payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {output_path}")
    print(f"trajectory_spectral_id_u64={payload['photonic_identity']['trajectory_spectral_id_u64']}")
    print(f"wrote {full_spectrum_output_path}")
    print(f"feedback_gate={full_spectrum_payload['feedback_gate']['state']}")


if __name__ == "__main__":
    main()
