# PyInstaller spec for QuantumMiner Win64 desktop application
# Production launcher: Qt Control Center window with BIOS/VHW/miner/prediction runtime

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules


here = Path(os.getcwd()).resolve()
app_name = "Quantum Miner"

# Use the existing PNG icon from Control_Center/assets if present.
icon_path = here / "Control_Center" / "assets" / "NeuralisIcon.png"

block_cipher = None

hiddenimports = ["scripts.worker_session_probe"]
for package in ("Control_Center", "bios", "miner", "prediction_engine", "VHW", "config"):
	try:
		hiddenimports.extend(collect_submodules(package))
	except Exception:
		pass

datas = []
asset_dir = here / "Control_Center" / "assets"
if asset_dir.is_dir():
	datas.append((str(asset_dir), "Control_Center/assets"))
research_dir = here / "ResearchConfinement"
if research_dir.is_dir():
	datas.append((str(research_dir), "ResearchConfinement"))
miner_cfg = here / "miner" / "miner_runtime_config.json"
if miner_cfg.is_file():
	datas.append((str(miner_cfg), "miner"))
settings_file = here / "config" / "user_settings.json"
if settings_file.is_file():
	datas.append((str(settings_file), "config"))


a = Analysis(
	["Control_Center/control_center_win.py"],
	pathex=[str(here)],
	binaries=[],
	datas=datas,
	hiddenimports=hiddenimports,
	hookspath=[],
	runtime_hooks=[],
	excludes=[],
	win_no_prefer_redirects=False,
	win_private_assemblies=False,
	cipher=block_cipher,
	noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
	pyz,
	a.scripts,
	a.binaries,
	a.zipfiles,
	a.datas,
	name=app_name,
	icon=str(icon_path) if icon_path.is_file() else None,
	console=False,
	disable_windowed_traceback=False,
	strip=False,
	upx=True,
	upx_exclude=[],
	runtime_tmpdir=None,
	version=None,
)

coll = COLLECT(
	exe,
	a.binaries,
	a.zipfiles,
	a.datas,
	strip=False,
	upx=True,
	upx_exclude=[],
	name=app_name,
)