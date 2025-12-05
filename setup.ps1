
#!/usr/bin/env pwsh
param (
    [string]$Python = "",
    [string]$Venv = ".venv",
    [switch]$NoVenv,
    [switch]$Help
)

function Show-Usage {
    @"
Usage: .\setup.ps1 [options]

Options:
  --python PATH     Use a specific Python interpreter (default: auto-detect).
  --venv DIR        Create/use a virtual environment in DIR (default: .venv).
  --no-venv         Install packages into the chosen interpreter without creating a venv.
  -h, --help        Show this help message.
"@
}

# --- Handle --help early ---
if ($Help -or $args -contains "-h" -or $args -contains "--help") {
    Show-Usage
    exit 0
}

# --- Repo root path ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# --- Detect Python ---
if (-not $Python) {
    if (Get-Command python3 -ErrorAction SilentlyContinue) {
        $Python = "python3"
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $Python = "python"
    } else {
        Write-Error "Python 3 is required but was not found in PATH."
        exit 1
    }
}

# --- Setup venv if not skipped ---
if (-not $NoVenv) {
    if (-not (Test-Path $Venv)) {
        Write-Host "Creating virtual environment at $Venv"
        & $Python -m venv $Venv
    } else {
        Write-Host "Using existing virtual environment at $Venv"
    }

    # Activate the venv
    $ActivateScript = Join-Path $Venv "Scripts\Activate.ps1"
    if (-not (Test-Path $ActivateScript)) {
        Write-Error "Could not find activate script at $ActivateScript"
        exit 1
    }
    & $ActivateScript

    # After activation, ensure Python refers to venv’s interpreter
    $Python = (Get-Command python).Source
}

# --- Pip commands ---
$Pip = "$Python -m pip"

Write-Host "Upgrading pip and build tooling..."
Invoke-Expression "$Pip install --upgrade pip setuptools wheel"

if (-not $env:NUMBA_CPU_NAME) {
    $env:NUMBA_CPU_NAME = "host"
}
if ($env:NUMBA_CPU_FEATURES) {
    Write-Host "Using custom NUMBA_CPU_FEATURES=$($env:NUMBA_CPU_FEATURES)"
} else {
    Write-Host "NUMBA_CPU_FEATURES not set; Numba will auto-detect host capabilities."
}

Write-Host "Installing Python dependencies for the audio GUI with NUMBA_CPU_NAME=$($env:NUMBA_CPU_NAME)..."
Invoke-Expression "$Pip install -r requirements.txt"

# --- Create default audio configuration ---
Write-Host "Creating default audio configuration if needed..."
& $Python "audio/setup_audio.py"

# --- Check optional tools ---
$MissingTools = @()
foreach ($tool in @("ffmpeg", "ffplay")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        $MissingTools += $tool
    }
}

if ($MissingTools.Count -gt 0) {
    Write-Warning "`nThe following optional command-line tools were not found: $($MissingTools -join ', ')"
    Write-Warning "They are required for certain audio preview features. Install them via your OS package manager."
}

Write-Host "`nSetup complete!"
if (-not $NoVenv) {
    Write-Host "Activate the environment with: `n    .\$Venv\Scripts\Activate.ps1"
}
Write-Host "Launch the GUI with: `n    python audio\src\main.py"
