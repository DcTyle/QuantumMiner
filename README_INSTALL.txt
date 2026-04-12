Quantum Application - FINAL FULL BUILD v3
========================================

This package contains the full Quantum Application source tree,
plus installer scripts, setup wizard, and utility glue files.

Folders
-------
bios/               BIOS boot, scheduler, self-check, event bus
VHW/                Virtual Hardware (lanes, tiers, VQPU, VQGPU, VSD)
core/               Physics and DMT equations
miner/              Mining engine, pool client, runtime config
prediction_engine/  Trading prediction engine and telemetry
Neuralis_AI/        Neuralis AI contextual engine
control_center/     Top-level controller / dashboards
setup_wizard/       Installer/setup wizard support code
installer/          Inno Setup script for Windows installer

Key Files
---------
run_app.py                  Main entry point (used by PyInstaller)
build_installer.bat         Windows build script (PyInstaller + Inno Setup)
quantum_application.spec    PyInstaller spec file
gpu_detect.bat              Simple GPU detection helper
README_INSTALL.txt          This file

How to build the Windows installer
----------------------------------
1. Install Python 3.10+ and open CMD in this folder.
   - pip install pyinstaller

2. Install Inno Setup 6 (default path is fine).

3. In this folder, run:
   build_installer.bat

   This will:
   - Run PyInstaller using quantum_application.spec
   - Produce dist/QuantumApplication/QuantumApplication.exe
   - Call the Inno Setup compiler to build:
     Quantum_Application_Setup.exe

4. Distribute Quantum_Application_Setup.exe to install on other systems.

Portable (no installer) usage
-----------------------------
1. Ensure Python 3.10+ and required dependencies are installed.
   Recommended Windows matrix for wider wheel support:
   - Python: 3.10.x
   - numpy: 1.26.4
   - numba: 0.59.1
   - llvmlite: 0.42.0

2. Create and activate a virtual environment (recommended):
   - Windows (PowerShell):
     ```
     py -3.10 -m venv env310
     .\env310\Scripts\Activate.ps1
     ```

3. Install dependencies:
   ```
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python -m run_app
   ```

Notes
-----
- All source files are ASCII-only by design (no Unicode in code).
- BIOS, VHW, miner, prediction_engine, and Neuralis_AI are wired through
  bios/event_bus.py and VHW/system_utils.py.
- Configure mining and trading via miner/miner_runtime_config.json
  and your .env / API key files as documented in your project.

Windows backend notes (ETC/RVN)
-------------------------------
- Some PoW backends (e.g., ethash/pyethash, kawpow) have limited Windows wheel
   availability and may require Microsoft C++ Build Tools to compile.
- If installation fails on Windows, consider:
   - Running ETC/RVN on Linux for easier wheel availability.
   - Temporarily disabling coins that require failing backends.
   - Verifying the recommended Python/numpy/numba/llvmlite versions above.
