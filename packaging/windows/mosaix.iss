; Windows installer for Mosaïx (Inno Setup 6).
;
;   iscc /DAppVersion=0.1.0 /DSourceExe=target\release\mosaix.exe /DOutputDir=dist packaging\windows\mosaix.iss
;
; Installs for the current user by default (no admin prompt), or for everyone when the user
; picks it; adds a Start menu entry, an optional desktop shortcut and an uninstaller.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceExe
  #define SourceExe "..\..\target\release\mosaix.exe"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
AppId={{2C4954C5-3E84-471F-AF9D-E91F5E0F1CAD}
AppName=Mosaïx
AppVersion={#AppVersion}
AppVerName=Mosaïx {#AppVersion}
AppPublisher=Mosaïx
AppPublisherURL=https://github.com/adamhlt/Mosaix
AppSupportURL=https://github.com/adamhlt/Mosaix/issues
DefaultDirName={autopf}\Mosaix
DefaultGroupName=Mosaïx
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=mosaix-windows-x64-setup
SetupIconFile=..\..\assets\icon\icon.ico
LicenseFile=..\..\LICENSE
UninstallDisplayIcon={app}\mosaix.exe
UninstallDisplayName=Mosaïx
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceExe}"; DestDir: "{app}"; DestName: "mosaix.exe"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Mosaïx"; Filename: "{app}\mosaix.exe"
Name: "{autodesktop}\Mosaïx"; Filename: "{app}\mosaix.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\mosaix.exe"; Description: "{cm:LaunchProgram,Mosaïx}"; Flags: nowait postinstall skipifsilent
