# QuantumApplication 
## Quick Start (Install)
- Recommended Python: 3.10.x (Windows)
- Install dependencies:
	```bash
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	```
- For Windows-specific guidance and version pins (numpy/numba/llvmlite), see [README_INSTALL.txt](README_INSTALL.txt).


Jarvis, begin initialization scan.  
Neuralis, perform a full repository analysis and report:

1. Architecture map
2. Module boundaries
3. Missing imports
4. Circular dependencies
5. ASCII compliance violations
6. Silent-error risks
7. Prediction engine -> Neuralis interface check
8. Miner engine -> VHW -> BIOS routing check
9. Neuralis_AI module hydration check
10. VSD and EventBus integration

Do not modify any code yet.  
Return a complete diagnostic report only.
return a diagnostic log for every error you find while system is running or while analyzing the project 

## Legacy Policy
- Primary compute type: `neural_objectPacket` only.
- No legacy model support: do not wrap, emulate, or reintroduce legacy paths.
- Compatibility shims are deprecated and slated for removal; favor native packet-based flows.
- Pool clients and adapters must operate on packetized data without hasattr-based fallbacks.
- Any module updates must overwrite legacy logic entirely unless explicitly instructed otherwise.

## Verification & CI
All changes are validated by guards locally and in CI to ensure architectural boundaries and import-graph health.

- Local quick checks:
	- `make verify` — runs guard and tests.
	- `make verify-strict` — enables stricter boundary checks (`STRICT_BOUNDARIES=1`).
	- `make guard` — import/cycle guard only.
	- `make test` — unit tests.

- Optional ASCII-only enforcement (text files):
	- `STRICT_ASCII=1 make verify`

- Repo-scoped Git hooks (recommended):
	- Enable pre-push hook once per clone:
		```bash
		git config core.hooksPath .githooks
		```
	- The hook runs `./scripts/verify.sh` before every push.

- CI workflow:
	- `.github/workflows/verify.yml` runs guard + tests on all pushes and pull requests (any actor: human, bot, or AI agent).
	- Protect branches (e.g., `master`) should require the “Verify” check to pass before merge.

## Virtual Hardware: GPU-Initiated Flow (No GPU Hashing)
- Purpose: The GPU anchors and initiates the Virtual Hardware (VHW) layer; it does not perform hashing. Mining remains in the virtual hardware pipeline.
- Module: `VHW/gpu_initiator.py` probes GPU presence, marks residency in VSD, and emits a light heartbeat to “sustain a small percentage” of coordinator activity before handing off to VHW.
- BIOS wiring: `bios/main_runtime.py` starts the GPU initiator at boot and stops it on shutdown.
- Configuration:
	- Env var: `VHW_GPU_SUSTAIN_PCT` controls the tiny coordinator share (default `0.05`).
	- Example:
		```bash
		export VHW_GPU_SUSTAIN_PCT=0.05
		```
- Verification (VSD keys):
	- `vhw/gpu/init_ok` -> `true` when initiator is running
	- `vhw/gpu/available` -> GPU detected flag
	- `vhw/gpu/heartbeat` -> periodic record with `ts`, `gpu_available`, `sustain_pct`
- Guarantees:
	- No GPU hashing is performed.
	- Virtual hardware runs as the compute substrate; GPU acts as initiator/coordinator only.