<#
.SYNOPSIS
  Build dist\LCTIMSpy\LCTIMSpy.exe (one folder, no console).

.DESCRIPTION
  Gates, in order, each stopping the build on failure:
    1. ruff and pytest (skip with -SkipTests)
    2. rebuild the web UI; warn if the committed LCTIMSpy\server\static is stale
    3. PyInstaller (packaging\lctimspy.spec)
    4. the frozen app's own --self-test: real extraction in worker processes
       re-launched from the exe, statistics, export, bundled SDK probe

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#>
param([switch]$SkipTests)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No virtualenv at .venv; create it and pip install -e "".[desktop,bruker,dev]"" pyinstaller" }

function Step([string]$name, [scriptblock]$body) {
    Write-Host "==> $name" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $body
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit code $LASTEXITCODE)" }
}

if (-not $SkipTests) {
    Step "ruff" { & $py -m ruff check LCTIMSpy tests }
    Step "pytest" { & $py -m pytest -q -p no:warnings tests }
}

Step "web UI build" {
    Push-Location web
    try { npm run build } finally { Pop-Location }
}
git diff --quiet -- LCTIMSpy/server/static
if ($LASTEXITCODE -ne 0) {
    Write-Host "    note: the rebuilt UI differs from the committed LCTIMSpy\server\static; commit it." -ForegroundColor Yellow
}

Step "PyInstaller" {
    & $py -m PyInstaller --noconfirm --clean --log-level WARN `
        --distpath dist --workpath build\pyinstaller packaging\lctimspy.spec
}

# The exe is a windowed app: piping its output makes PowerShell wait for it
# and shows what it prints.
Step "frozen self-test" { & (Join-Path $root "dist\LCTIMSpy\LCTIMSpy.exe") --self-test | Out-Host }

$size = (Get-ChildItem dist\LCTIMSpy -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Built dist\LCTIMSpy ({0:N0} MB). Install with packaging\install.ps1" -f $size) -ForegroundColor Green
