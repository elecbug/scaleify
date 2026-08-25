#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$ROOT_DIR/.venv"
REQ_FILE="$ROOT_DIR/requirements.txt"

PYTHON_VERSION="3.11"


die() {
    echo "Error: $*" >&2
    exit 1
}


# ============================================================
# Check requirements
# ============================================================

[[ -f "$REQ_FILE" ]] || die "requirements.txt not found: $REQ_FILE"


# ============================================================
# Install / locate uv
# ============================================================

if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
else
    echo "==> uv is not installed"

    command -v curl >/dev/null 2>&1 || {
        echo "curl is required to install uv." >&2
        echo "Install it with:" >&2
        echo "  sudo apt install curl" >&2
        exit 1
    }

    echo "==> Installing uv"

    curl -LsSf https://astral.sh/uv/install.sh | sh

    # uv installer normally places the binary here.
    if command -v uv >/dev/null 2>&1; then
        UV="$(command -v uv)"
    elif [[ -x "$HOME/.local/bin/uv" ]]; then
        UV="$HOME/.local/bin/uv"
    else
        die "uv installation completed, but the uv executable was not found."
    fi
fi

echo "==> Using uv: $UV"


# ============================================================
# Install Python
# ============================================================

echo "==> Ensuring Python $PYTHON_VERSION is available"

"$UV" python install "$PYTHON_VERSION"


# ============================================================
# Validate existing virtual environment
# ============================================================

if [[ -d "$VENV_DIR" ]]; then
    PYTHON="$VENV_DIR/bin/python"

    if [[ ! -x "$PYTHON" ]]; then
        echo "==> Invalid virtual environment found"
        echo "==> Removing: $VENV_DIR"

        rm -rf "$VENV_DIR"

    else
        CURRENT_VERSION="$(
            "$PYTHON" -c \
                'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
        )"

        if [[ "$CURRENT_VERSION" != "$PYTHON_VERSION" ]]; then
            echo "==> Existing virtual environment uses Python $CURRENT_VERSION"
            echo "==> Python $PYTHON_VERSION is required"
            echo "==> Recreating virtual environment"

            rm -rf "$VENV_DIR"
        else
            echo "==> Virtual environment already exists"
            echo "==> Python version: $CURRENT_VERSION"
        fi
    fi
fi


# ============================================================
# Create virtual environment
# ============================================================

if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> Creating virtual environment with Python $PYTHON_VERSION"

    "$UV" venv \
        --python "$PYTHON_VERSION" \
        "$VENV_DIR"
fi


PYTHON="$VENV_DIR/bin/python"

[[ -x "$PYTHON" ]] || die "Invalid virtual environment: $VENV_DIR"


# ============================================================
# Verify Python version
# ============================================================

ACTUAL_VERSION="$(
    "$PYTHON" -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'
)"

echo "==> Virtual environment Python: $ACTUAL_VERSION"


# ============================================================
# Install dependencies
# ============================================================

echo "==> Installing requirements"

"$UV" pip install \
    --python "$PYTHON" \
    -r "$REQ_FILE"


# ============================================================
# Check system dependencies
# ============================================================

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg is not installed." >&2
    echo "Install it with:" >&2
    echo "  sudo apt install ffmpeg" >&2
    exit 1
fi


# ============================================================
# Done
# ============================================================

echo
echo "Scaleify environment is ready."
echo
echo "Python:"
echo "  $PYTHON"
echo
echo "Activate it with:"
echo "  source $VENV_DIR/bin/activate"