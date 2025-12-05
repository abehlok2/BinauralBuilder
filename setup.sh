#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR=".venv"
PYTHON_BIN=""
SKIP_VENV=0

usage() {
    cat <<'USAGE'
Usage: ./setup.sh [options]

Options:
  --python PATH     Use a specific Python interpreter (default: auto-detect).
  --venv DIR        Create/use a virtual environment in DIR (default: .venv).
  --no-venv         Install packages into the chosen interpreter without creating a venv.
  -h, --help        Show this help message.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            if [[ $# -lt 2 ]]; then
                echo "Error: --python requires a value" >&2
                exit 1
            fi
            PYTHON_BIN="$2"
            shift 2
            ;;
        --venv)
            if [[ $# -lt 2 ]]; then
                echo "Error: --venv requires a value" >&2
                exit 1
            fi
            VENV_DIR="$2"
            shift 2
            ;;
        --no-venv)
            SKIP_VENV=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "Error: Python 3 is required but was not found." >&2
        exit 1
    fi
fi

if [[ $SKIP_VENV -eq 0 ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Creating virtual environment at $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        echo "Using existing virtual environment at $VENV_DIR"
    fi

    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    PYTHON_BIN="$(command -v python)"
fi

PIP_CMD=("$PYTHON_BIN" -m pip)

echo "Upgrading pip and build tooling..."
"${PIP_CMD[@]}" install --upgrade pip setuptools wheel

export NUMBA_CPU_NAME="${NUMBA_CPU_NAME:-host}"
if [[ -n "${NUMBA_CPU_FEATURES:-}" ]]; then
    echo "Using custom NUMBA_CPU_FEATURES=${NUMBA_CPU_FEATURES}"
    export NUMBA_CPU_FEATURES
else
    echo "NUMBA_CPU_FEATURES not set; Numba will auto-detect host capabilities."
fi
echo "Installing Python dependencies for the audio GUI with NUMBA_CPU_NAME=${NUMBA_CPU_NAME}..."
"${PIP_CMD[@]}" install -r requirements.txt

if [[ $SKIP_VENV -eq 1 ]]; then
    ACTIVE_PYTHON="$PYTHON_BIN"
else
    ACTIVE_PYTHON="$(command -v python)"
fi

echo "Creating default audio configuration if needed..."
"$ACTIVE_PYTHON" audio/setup_audio.py

missing_tools=()
for tool in ffmpeg ffplay; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        missing_tools+=("$tool")
    fi
done

if [[ ${#missing_tools[@]} -gt 0 ]]; then
    echo "\nWARNING: The following optional command-line tools were not found: ${missing_tools[*]}" >&2
    echo "They are required for certain audio preview features. Install them via your OS package manager." >&2
fi

echo "\nSetup complete!"
if [[ $SKIP_VENV -eq 0 ]]; then
    echo "Activate the environment with: source $VENV_DIR/bin/activate"
fi
echo "Launch the GUI with: python audio/src/main.py"
