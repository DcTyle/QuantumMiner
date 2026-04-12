; Inno Setup script for Neuralis desktop application
; This is a starter config; adjust paths and metadata as needed.

[Setup]
AppName=Neuralis Control Center
AppVersion=1.0.0
DefaultDirName={pf}\Neuralis
DefaultGroupName=Neuralis
DisableDirPage=no
DisableProgramGroupPage=no
OutputBaseFilename=Neuralis_Installer
Compression=lzma
SolidCompression=yes
WizardStyle=modern

; Update this if you move or rename the exe.
SourceDir=..\dist\Neuralis

[Files]
Source="Neuralis.exe"; DestDir="{app}"; Flags: ignoreversion

[Icons]
Name="{group}\Neuralis"; Filename="{app}\Neuralis.exe"
Name="{commondesktop}\Neuralis"; Filename="{app}\Neuralis.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename="{app}\Neuralis.exe"; Description="Launch Neuralis"; Flags: nowait postinstall skipifsilent