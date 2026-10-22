<#
.SYNOPSIS
  Remove the per-user LCTIMSpy install (run from "Apps & features" or directly).

.DESCRIPTION
  Removes the program folder, the shortcuts and the "Apps & features" entry.
  Your datasets list and extraction cache in %LOCALAPPDATA%\LCTIMSpy are kept;
  delete that folder yourself if you want them gone too.
#>
$ErrorActionPreference = "Stop"
if (Get-Process -Name LCTIMSpy -ErrorAction SilentlyContinue) { throw "LCTIMSpy is running. Close it, then uninstall again." }

$dest = Join-Path $env:LOCALAPPDATA "Programs\LCTIMSpy"
$links = @(
    (Join-Path ([Environment]::GetFolderPath("Programs")) "LCTIMSpy.lnk"),
    (Join-Path ([Environment]::GetFolderPath("Desktop")) "LCTIMSpy.lnk")
)
foreach ($l in $links) { if (Test-Path $l) { Remove-Item -Force $l } }
$key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\LCTIMSpy"
if (Test-Path $key) { Remove-Item -Recurse -Force $key }
if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }

Write-Host "LCTIMSpy was removed. Your data in $(Join-Path $env:LOCALAPPDATA 'LCTIMSpy') was kept."
