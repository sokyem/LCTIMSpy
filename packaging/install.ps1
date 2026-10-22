<#
.SYNOPSIS
  Install the built LCTIMSpy app for the current user (no admin rights needed).

.DESCRIPTION
  Copies dist\LCTIMSpy to %LOCALAPPDATA%\Programs\LCTIMSpy, adds Start Menu and
  Desktop shortcuts, and registers an "Apps & features" entry so it can be
  uninstalled like any other app. Re-running upgrades in place. Your datasets
  list and extraction cache live elsewhere (%LOCALAPPDATA%\LCTIMSpy) and are
  never touched.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\install.ps1
#>
param(
    [string]$Source = (Join-Path (Split-Path -Parent $PSScriptRoot) "dist\LCTIMSpy"),
    [switch]$NoDesktopShortcut
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $Source "LCTIMSpy.exe"
if (-not (Test-Path $exe)) { throw "Nothing to install: $exe not found. Build first with packaging\build.ps1." }
if (Get-Process -Name LCTIMSpy -ErrorAction SilentlyContinue) { throw "LCTIMSpy is running. Close it, then run the installer again." }

$dest = Join-Path $env:LOCALAPPDATA "Programs\LCTIMSpy"
$init = Join-Path (Split-Path -Parent $PSScriptRoot) "LCTIMSpy\__init__.py"
$version = "0.0.0"
if (Test-Path $init) {
    $m = Select-String -Path $init -Pattern '__version__\s*=\s*"([^"]+)"'
    if ($m) { $version = $m.Matches[0].Groups[1].Value }
}

Write-Host "Installing LCTIMSpy $version to $dest"
if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
New-Item -ItemType Directory -Force (Split-Path -Parent $dest) | Out-Null
Copy-Item -Recurse $Source $dest
Copy-Item (Join-Path $PSScriptRoot "uninstall.ps1") $dest

$shell = New-Object -ComObject WScript.Shell
function New-Shortcut([string]$path) {
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = Join-Path $dest "LCTIMSpy.exe"
    $link.WorkingDirectory = $dest
    $link.IconLocation = (Join-Path $dest "LCTIMSpy.exe") + ",0"
    $link.Description = "LC-TIMS-MS isomer analysis"
    $link.Save()
    Write-Host "  shortcut: $path"
}
New-Shortcut (Join-Path ([Environment]::GetFolderPath("Programs")) "LCTIMSpy.lnk")
if (-not $NoDesktopShortcut) {
    New-Shortcut (Join-Path ([Environment]::GetFolderPath("Desktop")) "LCTIMSpy.lnk")
}

# Per-user "Apps & features" entry.
$key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\LCTIMSpy"
New-Item -Path $key -Force | Out-Null
$sizeKb = [int]((Get-ChildItem $dest -Recurse -File | Measure-Object Length -Sum).Sum / 1KB)
$values = @{
    DisplayName     = "LCTIMSpy"
    DisplayVersion  = $version
    Publisher       = "LCTIMSpy"
    InstallLocation = $dest
    DisplayIcon     = Join-Path $dest "LCTIMSpy.exe"
    UninstallString = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $dest 'uninstall.ps1')`""
}
foreach ($name in $values.Keys) { Set-ItemProperty -Path $key -Name $name -Value $values[$name] }
Set-ItemProperty -Path $key -Name NoModify -Value 1 -Type DWord
Set-ItemProperty -Path $key -Name NoRepair -Value 1 -Type DWord
Set-ItemProperty -Path $key -Name EstimatedSize -Value $sizeKb -Type DWord

Write-Host "Done. Start LCTIMSpy from the Start Menu or the Desktop shortcut." -ForegroundColor Green
Write-Host "Needs the Microsoft Edge WebView2 Runtime (built into Windows 11)."
