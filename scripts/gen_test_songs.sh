#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$ROOT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
GEN_DIR="$SCRIPT_DIR/gen/test"

die() {
    echo "Error: $*" >&2
    exit 1
}

[[ -x "$PYTHON" ]] || die "Virtual environment not found. Run: ./scripts/setup.sh"
[[ -d "$GEN_DIR" ]] || die "Test generator directory not found: $GEN_DIR"

mapfile -d '' GENERATORS < <(
    find "$GEN_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*.py' \
        ! -name '__init__.py' \
        -print0 |
    sort -z
)

if (( ${#GENERATORS[@]} == 0 )); then
    die "No test-song generators found in: $GEN_DIR"
fi

echo "==> Running ${#GENERATORS[@]} test-song generator(s)"

cd "$ROOT_DIR"

for generator in "${GENERATORS[@]}"; do
    echo
    echo "==> $(basename "$generator")"
    "$PYTHON" "$generator"
done

echo
echo "==> All test-song generators completed."
echo "    Results directory: $ROOT_DIR/results"