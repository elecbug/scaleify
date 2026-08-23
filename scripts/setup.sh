#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$ROOT_DIR/venv"
REQ_FILE="$ROOT_DIR/requirements.txt"

die() {
    echo "Error: $*" >&2
    exit 1
}

command -v python3 >/dev/null 2>&1 || die "python3 is not installed."
[[ -f "$REQ_FILE" ]] || die "requirements.txt not found: $REQ_FILE"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> Creating virtual environment"
    python3 -m venv "$VENV_DIR" || {
        echo "Failed to create virtual environment." >&2
        echo "On Debian/Ubuntu, install python3-venv first:" >&2
        echo "  sudo apt install python3-venv" >&2
        exit 1
    }
else
    echo "==> Virtual environment already exists"
fi

PYTHON="$VENV_DIR/bin/python"
[[ -x "$PYTHON" ]] || die "Invalid virtual environment: $VENV_DIR"

echo "==> Upgrading pip"
"$PYTHON" -m pip install --upgrade pip

echo "==> Installing requirements"
"$PYTHON" -m pip install -r "$REQ_FILE"

echo
echo "Scaleify environment is ready."
echo "Activate it with:"
echo "  source $VENV_DIR/bin/activate"