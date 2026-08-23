#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

RESULTS_DIR="$ROOT_DIR/results"
VENV_DIR="$ROOT_DIR/.venv"

PYTHON="$VENV_DIR/bin/python"
MAKE_BLIND="$SCRIPT_DIR/make_blind_test.py"

RAW_DIR="$RESULTS_DIR/raw"
TUNED_DIR="$RESULTS_DIR/tuned"
BLIND_DIR="$RESULTS_DIR/blind"
DEPLOY_DIR="$RESULTS_DIR/blind_deploy"

TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
INTEGRITY_DIR="$RESULTS_DIR/blind_$TIMESTAMP"
INTEGRITY_ARCHIVE="$INTEGRITY_DIR.tar.gz"


die() {
    echo "Error: $*" >&2
    exit 1
}


log() {
    echo
    echo "==> $*"
}


require() {
    local path="$1"
    local description="$2"

    [[ -e "$path" ]] || die "$description not found: $path"
}


cleanup_workdirs() {
    rm -rf -- "$BLIND_DIR" "$DEPLOY_DIR"
}


prepare_tuned() {
    log "Preparing tuned input"

    mkdir -p -- "$TUNED_DIR"

    cp -a -- "$RAW_DIR"/. "$TUNED_DIR"/
}


generate_blind() {
    log "Generating blind test"

    mkdir -p -- "$BLIND_DIR"

    "$PYTHON" "$MAKE_BLIND" \
        "$TUNED_DIR" \
        --output "$BLIND_DIR"
}


prepare_deploy() {
    log "Preparing deploy package"

    mkdir -p -- "$DEPLOY_DIR"
    cp -a -- "$BLIND_DIR"/. "$DEPLOY_DIR"/

    find "$DEPLOY_DIR" \
        -type f \
        -name '*.csv' \
        -delete

    rename_deploy_cases
}


rename_deploy_cases() {
    local subdir
    local basename
    local first_char
    local target

    while IFS= read -r -d '' subdir; do
        basename="$(basename -- "$subdir")"
        first_char="${basename:0:1}"
        target="$DEPLOY_DIR/case-$first_char"

        if [[ "$subdir" == "$target" ]]; then
            continue
        fi

        [[ ! -e "$target" ]] ||
            die "Case name collision: '$basename' -> 'case-$first_char'"

        mv -- "$subdir" "$target"
    done < <(
        find "$DEPLOY_DIR" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -print0
    )
}


create_deploy_archive() {
    log "Creating deploy archive"

    tar -czf "$RESULTS_DIR/blind_deploy.tar.gz" \
        -C "$DEPLOY_DIR" \
        .
}


create_integrity_archive() {
    log "Creating integrity archive"

    mkdir -p -- "$INTEGRITY_DIR"

    mv -- \
        "$RESULTS_DIR/blind_deploy.tar.gz" \
        "$INTEGRITY_DIR/"

    find "$BLIND_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*.csv' \
        -exec mv -t "$INTEGRITY_DIR" -- {} +

    tar -czf "$INTEGRITY_ARCHIVE" \
        -C "$INTEGRITY_DIR" \
        .
}


cleanup() {
    log "Cleaning temporary directories"

    rm -rf -- \
        "$BLIND_DIR" \
        "$DEPLOY_DIR"
}


main() {
    require "$PYTHON" "Virtual environment Python"
    require "$MAKE_BLIND" "Make blind test script"
    require "$RAW_DIR" "Raw results directory"

    cd -- "$ROOT_DIR"

    log "Starting blind test packaging"

    cleanup_workdirs
    prepare_tuned
    generate_blind
    prepare_deploy
    create_deploy_archive
    create_integrity_archive
    cleanup

    echo
    echo "==> Blind test package completed."
    echo "    Integrity directory: $INTEGRITY_DIR"
    echo "    Integrity archive:   $INTEGRITY_ARCHIVE"
}


main "$@"