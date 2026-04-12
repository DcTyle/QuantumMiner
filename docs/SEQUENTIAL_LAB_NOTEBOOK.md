# Sequential Lab Notebook

This document is the repo-level reading order for notebook material.
It does not replace the append-only notebooks under ResearchConfinement.
It tells the reader which notebook or ledger to open first, what each one
answers, and how the operational and research records connect.

## Primary Sequence

1. [docs/CONTEXT_CLASSIFICATION_INDEX.md](CONTEXT_CLASSIFICATION_INDEX.md)
2. [docs/GENESIS_APPEND_ONLY_NOTEBOOK_LOGGING.md](GENESIS_APPEND_ONLY_NOTEBOOK_LOGGING.md)
3. [ResearchConfinement/docs_v4/ENGINE_BUILD_RUNTIME_LAB_NOTEBOOK.md](../ResearchConfinement/docs_v4/ENGINE_BUILD_RUNTIME_LAB_NOTEBOOK.md)
4. [ResearchConfinement/docs_v4/LAB_NOTEBOOK_AND_LOG_SEQUENCE_V4.md](../ResearchConfinement/docs_v4/LAB_NOTEBOOK_AND_LOG_SEQUENCE_V4.md)
5. [ResearchConfinement/docs_v4/README_CONTINUATION_V4.md](../ResearchConfinement/docs_v4/README_CONTINUATION_V4.md)
6. [ResearchConfinement/OPEN_ME_FIRST.txt](../ResearchConfinement/OPEN_ME_FIRST.txt)

## What Each Record Owns

### Context Routing

- [docs/CONTEXT_CLASSIFICATION_INDEX.md](CONTEXT_CLASSIFICATION_INDEX.md)
- Use this first when the question is "which part of the repository explains this?"
- It routes readers by working context: research continuation, canonical engine specs, physics framing, logs, and lab notes.

### Notebook Generation Rules

- [docs/GENESIS_APPEND_ONLY_NOTEBOOK_LOGGING.md](GENESIS_APPEND_ONLY_NOTEBOOK_LOGGING.md)
- Use this when the question is "how are build, install, package, and runtime notes appended?"
- It defines the logging pipeline, provider selection, and append-only constraints for generated notebook entries.

### Engine Build And Runtime Chronology

- [ResearchConfinement/docs_v4/ENGINE_BUILD_RUNTIME_LAB_NOTEBOOK.md](../ResearchConfinement/docs_v4/ENGINE_BUILD_RUNTIME_LAB_NOTEBOOK.md)
- Use this when the question is "what happened during build, install, package, editor, or runtime execution?"
- This is the operational notebook generated from deterministic log sources and shutdown hooks.

Supporting raw logs:
- [GenesisEngineState/Logs](../GenesisEngineState/Logs)
- [build_config_log.txt](../build_config_log.txt)
- [build_config_log_capture.txt](../build_config_log_capture.txt)

### Research Run Chronology

- [ResearchConfinement/docs_v4/LAB_NOTEBOOK_AND_LOG_SEQUENCE_V4.md](../ResearchConfinement/docs_v4/LAB_NOTEBOOK_AND_LOG_SEQUENCE_V4.md)
- Use this when the question is "what happened across the accessible V4 experiment runs, in order?"
- This is the sequential research notebook that stitches the durable run artifacts into one chronology.

### Accessible Research Package Summary

- [ResearchConfinement/docs_v4/README_CONTINUATION_V4.md](../ResearchConfinement/docs_v4/README_CONTINUATION_V4.md)
- Use this when the question is "what was supposed to be in the V4 handoff package, and which parts are actually on disk?"
- This README is the accessible package summary for the current checkout.

### Research Framing

- [ResearchConfinement/OPEN_ME_FIRST.txt](../ResearchConfinement/OPEN_ME_FIRST.txt)
- [ResearchConfinement/photon_native_dmt_sim_spec.md](../ResearchConfinement/photon_native_dmt_sim_spec.md)
- Use these after the sequential notebook when the question is "what is the continuation framing and theoretical setup for the accessible research package?"

### Checkout Note

- The current checkout does not include standalone `RUN_LEDGER_V4.csv`, `EXPERIMENT_ARCHIVE_V4.md`, `MATH_STACK_AND_RESULTS_V4.md`, `FILE_MANIFEST_V4.txt`, or `CONTINUATION_PROMPT_V4.txt` files under `ResearchConfinement/docs_v4`.
- When older notes mention those artifacts, use [ResearchConfinement/docs_v4/README_CONTINUATION_V4.md](../ResearchConfinement/docs_v4/README_CONTINUATION_V4.md) plus [ResearchConfinement/docs_v4/LAB_NOTEBOOK_AND_LOG_SEQUENCE_V4.md](../ResearchConfinement/docs_v4/LAB_NOTEBOOK_AND_LOG_SEQUENCE_V4.md) as the accessible continuation route on this tree.

## Recommended Read Paths

### Build Or Packaging Failure

1. [docs/GENESIS_APPEND_ONLY_NOTEBOOK_LOGGING.md](GENESIS_APPEND_ONLY_NOTEBOOK_LOGGING.md)
2. [ResearchConfinement/docs_v4/ENGINE_BUILD_RUNTIME_LAB_NOTEBOOK.md](../ResearchConfinement/docs_v4/ENGINE_BUILD_RUNTIME_LAB_NOTEBOOK.md)
3. [GenesisEngineState/Logs](../GenesisEngineState/Logs)

### Runtime Or Editor Regression

1. [ResearchConfinement/docs_v4/ENGINE_BUILD_RUNTIME_LAB_NOTEBOOK.md](../ResearchConfinement/docs_v4/ENGINE_BUILD_RUNTIME_LAB_NOTEBOOK.md)
2. [docs/CONTEXT_CLASSIFICATION_INDEX.md](CONTEXT_CLASSIFICATION_INDEX.md)
3. [docs/spec_uploads/GenesisengineSpec.md](spec_uploads/GenesisengineSpec.md)

### Research Continuation

1. [ResearchConfinement/OPEN_ME_FIRST.txt](../ResearchConfinement/OPEN_ME_FIRST.txt)
2. [ResearchConfinement/docs_v4/README_CONTINUATION_V4.md](../ResearchConfinement/docs_v4/README_CONTINUATION_V4.md)
3. [ResearchConfinement/docs_v4/LAB_NOTEBOOK_AND_LOG_SEQUENCE_V4.md](../ResearchConfinement/docs_v4/LAB_NOTEBOOK_AND_LOG_SEQUENCE_V4.md)
4. [ResearchConfinement/photon_native_dmt_sim_spec.md](../ResearchConfinement/photon_native_dmt_sim_spec.md)

## Working Rule

- Operational notebook entries explain what the engine build and runtime actually did.
- Research notebook entries explain what each experiment run produced in chronological order.
- The ledger fixes ordering; the archive fixes interpretation.
- When a question crosses engineering and research boundaries, start from this file and then branch into the appropriate notebook lane.