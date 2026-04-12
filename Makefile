# QuantumMiner Makefile (ASCII-only)

.PHONY: verify verify-strict guard test

# Fast local verification (non-strict miner->Neuralis_AI with allowlist)
verify:
	./scripts/verify.sh

# Strict verification with miner->Neuralis_AI enforcement (except allowlist path)
verify-strict:
	STRICT_BOUNDARIES=1 ./scripts/verify.sh

guard:
	python3 scripts/check_import_cycles.py --strict --focus bios,VHW,miner,core

test:
	python -m unittest -q
