# Run 044 Notes

Purpose:
- test whether cross-file, object-like runtime processes can be encoded into substrate vector space and decoded back to the same outputs faster than the classical baseline.
- test whether noise-aware startup telemetry, including latency and actuation prediction, can drive the same substrate path while preserving classical and trace-state equivalence.
- test whether a real Vulkan compute dispatch can actuate the live telemetry capture path strongly enough to form an actuation-adjusted telemetry plane for the calculus.

What was run:
- substrate microprocess benchmark using mixed module names, enum cases, strings, structures, and calculation words.
- synthetic startup telemetry resonance benchmark with latency-aware actuation calibration.
- live startup telemetry resonance benchmark using a real Vulkan actuation backend, frozen actuation-adjusted `device_snapshot()` frames, and substrate/classical decode on the same captured window.

Primary outputs:
- [run_044_research_summary.json](./run_044_research_summary.json)
- [run_044_experiment_notes.md](./run_044_experiment_notes.md)
- [run_044_lab_notes.md](./run_044_lab_notes.md)
- [run_044_equation_stack.md](./run_044_equation_stack.md)
- [run_044_formal_review.md](./run_044_formal_review.md)
- [run_044_benchmark_summary.csv](./run_044_benchmark_summary.csv)
- [run_044_forecast_preview.csv](./run_044_forecast_preview.csv)
- [run_044_dominant_nodes.csv](./run_044_dominant_nodes.csv)
- [run_044_live_startup_frames.csv](./run_044_live_startup_frames.csv)

Current headline result:
- the live actuation run used `VHW.gpu_pulse_runtime._vulkan_calibration_actuation` to build and dispatch the Vulkan calibration kernel against the `NVIDIA GeForce RTX 2060`.
- the live actuation summary reported `applied=true`, `dispatch_elapsed_ms_mean=26.7235`, and `load_hint_mean=0.264657`.
- the frozen live frame window then produced `results_match=true` and `trace_state_match=true` between the substrate and classical decode paths.
- raw OS telemetry counters remained near zero during the short burst, so this run must be interpreted through the actuation-adjusted telemetry plane, not as proof that coarse OS counters alone carry the full calculus.
