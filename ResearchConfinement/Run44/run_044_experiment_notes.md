# Run 044 Experiment Notes

## Purpose
Test whether the runtime can treat the substrate pulse trace as a microprocessor for engine-style logic, then test whether startup telemetry can drive that same substrate while accounting for noise, resonant nodes, latency, and a real kernel actuation point.

## Method
1. Encoded cross-file runtime artifacts into substrate words:
   - module names
   - enum-like route cases
   - strings
   - structures
   - calculation words
2. Ran the encoded artifacts through the substrate trace loop and decoded the resulting words back into deterministic process outputs.
3. Benchmarked the substrate path against the classical baseline using the same process workload.
4. Generated a synthetic startup telemetry cohort, detected resonant nodes, forecast near-term conditions, predicted kernel latency and actuation delay, then ran substrate and classical decoding against the same frozen frame set.
5. Captured a live startup telemetry window from `device_snapshot()` while calling the Vulkan calibration backend on every frame:
   - `ResearchConfinement/Calibrationkernals/vulkan_gpu_calibration_kernels/src/main.cpp`
   - `VHW.gpu_pulse_runtime._vulkan_calibration_actuation`
6. Froze the actuation-adjusted live frame set and reran the same substrate/classical comparison so both paths consumed the exact same live frames.

## Results
1. The substrate microprocess benchmark reproduced the same decoded process result as the classical baseline and ran faster:
   - results match: `true`
   - speedup total: `1.6829795435216928x`
   - speedup exec only: `1.8485025376505608x`
   - route case: `string_fold`
   - result tag: `6e472963-b461dde1-7958a892`
2. The synthetic telemetry benchmark reproduced both the decoded result and the trace-state output:
   - results match: `true`
   - trace-state match: `true`
   - speedup total: `3.500636302412586x`
   - speedup exec only: `5.290160073244487x`
   - predicted latency: `0.015418717524415069 s`
   - actuation compensation: `0.5855153453099242`
3. The live startup telemetry benchmark also reproduced both the decoded result and the trace-state output when evaluated against the same frozen live frame set:
   - actuation backend: `vulkan_calibration`
   - device tag: `NVIDIA GeForce RTX 2060`
   - actuation applied: `true`
   - actuation calls: `8`
   - mean dispatch time: `26.7235 ms`
   - mean actuation load hint: `0.26465699277`
   - results match: `true`
   - trace-state match: `true`
   - speedup total: `2.90718339924873x`
   - speedup exec only: `5.486867172616259x`
   - predicted latency: `0.008410595782358508 s`
   - route case: `enum_switch`
   - result tag: `06d46fc9-9efbd8ab-6ce03820`
4. The runtime therefore passed the current equivalence test:
   - live Vulkan actuation -> actuation-adjusted telemetry frames -> pulse trace substrate -> decoded values
   - same frozen actuation-adjusted frames -> classical path -> decoded values
   - both produced the same result and trace for the same captured live window.

## Deviations / Risk Notes
1. The live telemetry source is still aggregate `device_snapshot()` telemetry, not raw per-kernel driver counters or launch timestamps.
2. On this machine, the short Vulkan bursts did not move the coarse OS `raw_gpu_util` field above zero, so the live frame plane had to carry both:
   - raw counters
   - actuation-adjusted counters derived from the real kernel return values
3. The benchmarked speedups are runtime-path speedups from one-time encoding/profile work plus branch-reduced substrate execution. They are not yet proof that the full production engine has been moved into native GPU kernels.
4. The live equivalence claim depends on frozen-frame comparison. It does not yet prove equivalence under continuously changing live telemetry during long-running execution.

## Interpretation
This run is stronger than the earlier passive live-telemetry version because the kernel actuation point was real. The most accurate interpretation is:

- a real Vulkan compute dispatch produced actuation metadata,
- that metadata was projected into the live telemetry plane,
- the substrate and classical paths then decoded the same values from the same frozen frame set.

What the run still does not claim is that coarse OS telemetry by itself contains the whole calculus. The short-burst actuation had to be represented through the returned kernel summary so the runtime could see the pulse state in time to act on it.
