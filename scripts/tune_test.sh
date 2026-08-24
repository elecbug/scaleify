#!/usr/bin/env bash
set -euo pipefail

# Scaleify dataset generation -> style training -> listening test helper.
#
# Usage:
#   ./tune_test.sh japan
#   ./tune_test.sh korea
#   ./tune_test.sh china
#   ./tune_test.sh jsmel
#   ./tune_test.sh vocaloid
#
# Force dataset regeneration/download:
#   ./tune_test.sh japan --download
#
# Force style retraining:
#   ./tune_test.sh japan --training
#
# Force both:
#   ./tune_test.sh japan --download --training
#
# Run every configured corpus:
#   ./tune_test.sh all
#
# Default behavior:
#   - dataset generation is skipped if at least one WAV already exists
#   - training is skipped if at least one matching *_cluster_*.json already exists
#   - listening tests are always run for every matching cluster JSON
#
# Assumes this script is executed from the Scaleify repository root.

CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
SETUP_SCRIPT="$SCRIPT_DIR/setup.sh"
RESULTS_DIR="$ROOT_DIR/results"
TUNED_RESULTS_DIR="$RESULTS_DIR/tuned"
STYLE_DIR="$RESULTS_DIR/styles"
GENERATOR_DIR="$SCRIPT_DIR/gen"
DATASETS_DIR="$ROOT_DIR/datasets"
REPORTS_DIR="$RESULTS_DIR/reports"
RAW_RESULTS_DIR="$RESULTS_DIR/raw"

ERIKA="$RAW_RESULTS_DIR/erika_test.wav"
KOROBEINIKI="$RAW_RESULTS_DIR/korobeiniki_test.wav"
TWINKLE="$RAW_RESULTS_DIR/twinkle_twinkle_test.wav"

FORCE_DOWNLOAD=0
FORCE_TRAINING=0

STYLE_AMOUNT=1.0
RHYTHM_AMOUNT=0.7
TIMBRE=flute

ensure_venv() {
    if [[ ! -x "$PYTHON" ]]; then
        echo "==> Virtual environment not found; running setup"

        [[ -x "$SETUP_SCRIPT" ]] || die "Missing setup script: $SETUP_SCRIPT"

        "$SETUP_SCRIPT"
    fi

    [[ -x "$PYTHON" ]] || die "Python executable not found after setup: $PYTHON"

    echo "==> Using Python: $PYTHON"
}

usage() {
    cat <<'EOF'
Usage:
  ./tune_test.sh <dataset> [--download] [--training]

Datasets:
  japan       Japan 1892 corpus
  korea       Korean traditional corpus
  china       Chinese traditional corpus
  jsmel       JSMel public-domain corpus
  vocaloid    Official Vocaloid corpus
  all         Run all of the above

Options:
  --download   Re-run dataset generation/download even if WAV files exist
  --training   Re-run train_style.py even if trained style JSON files exist
  -h, --help   Show this help

Examples:
  ./tune_test.sh japan
  ./tune_test.sh jsmel --training
  ./tune_test.sh vocaloid --download --training
  ./tune_test.sh all
EOF
}

die() {
    echo "Error: $*" >&2
    cd "$CURRENT_DIR" || echo "Failed to cd back to original directory: $CURRENT_DIR" >&2
    exit 1
}

has_wavs() {
    local dir="$1"
    [[ -d "$dir" ]] && find "$dir" -maxdepth 1 -type f -name '*.wav' -print -quit | grep -q .
}

has_styles() {
    local prefix="$1"
    compgen -G "${STYLE_DIR}/${prefix}_cluster_*.json" >/dev/null
}

cleanup_training_outputs() {
    mkdir -p "$REPORTS_DIR"
    mv "${STYLE_DIR}"/*.csv "$REPORTS_DIR/"
    mv "${STYLE_DIR}"/*_report.json "$REPORTS_DIR/"
}

run_generator() {
    local dataset="$1"
    local generator_cmd="$2"
    local dataset_dir="$3"

    if (( FORCE_DOWNLOAD )); then
        echo "==> [${dataset}] Re-running dataset generation (--download)"
        eval "$generator_cmd"
        return
    fi

    if has_wavs "$dataset_dir"; then
        local count
        count="$(find "$dataset_dir" -maxdepth 1 -type f -name '*.wav' | wc -l | tr -d ' ')"
        echo "==> [${dataset}] Dataset exists (${count} WAVs); skipping generation"
    else
        echo "==> [${dataset}] Dataset not found; generating"
        eval "$generator_cmd"
    fi
}

run_training() {
    local dataset="$1"
    local dataset_dir="$2"
    local style_prefix="$3"
    shift 3

    local other_args=("$@")

    mkdir -p "$STYLE_DIR"

    if (( FORCE_TRAINING )); then
        echo "==> [${dataset}] Re-running style training (--training)"
        "$PYTHON" "$SCRIPT_DIR/train_style.py" \
            "$dataset_dir/" \
            --output "$STYLE_DIR" \
            "${other_args[@]}"
        cleanup_training_outputs
        return
    fi

    if has_styles "$style_prefix"; then
        local count
        count="$(
            find "$STYLE_DIR" -maxdepth 1 \
                -type f \
                -name "${style_prefix}_cluster_*.json" |
            wc -l |
            tr -d ' '
        )"
        echo "==> [${dataset}] Trained styles exist (${count} clusters); skipping training"
    else
        echo "==> [${dataset}] Trained styles not found; training"
        "$PYTHON" "$SCRIPT_DIR/train_style.py" \
            "$dataset_dir/" \
            --output "$STYLE_DIR" \
            "${other_args[@]}"
        cleanup_training_outputs
    fi
}

run_listening_tests() {
    local dataset="$1"
    local style_prefix="$2"

    [[ -f "$ERIKA" ]] || die "Missing test file: $ERIKA"
    [[ -f "$KOROBEINIKI" ]] || die "Missing test file: $KOROBEINIKI"
    [[ -f "$TWINKLE" ]] || die "Missing test file: $TWINKLE"

    shopt -s nullglob
    local style_files=( "${STYLE_DIR}/${style_prefix}_cluster_"*.json )
    shopt -u nullglob

    if (( ${#style_files[@]} == 0 )); then
        die "[${dataset}] No style JSONs found: ${STYLE_DIR}/${style_prefix}_cluster_*.json"
    fi

    echo "==> [${dataset}] Running listening tests for ${#style_files[@]} cluster(s)"

    local style_file style_id

    for style_file in "${style_files[@]}"; do
        style_id="$(basename "$style_file" .json)"

        echo "    Erika <- ${style_id}"
        "$PYTHON" "$SCRIPT_DIR/scaleify.py" "$ERIKA" \
            --style "$style_id" \
            --style-dir "$STYLE_DIR/" \
            --root G \
            --style-amount "$STYLE_AMOUNT" \
            --rhythm-amount "$RHYTHM_AMOUNT" \
            --output "$TUNED_RESULTS_DIR/erika_test_${style_id}.wav" \
            --timbre "$TIMBRE"
    done

    for style_file in "${style_files[@]}"; do
        style_id="$(basename "$style_file" .json)"

        echo "    Korobeiniki <- ${style_id}"
        "$PYTHON" "$SCRIPT_DIR/scaleify.py" "$KOROBEINIKI" \
            --style "$style_id" \
            --style-dir "$STYLE_DIR/" \
            --root A \
            --style-amount "$STYLE_AMOUNT" \
            --rhythm-amount "$RHYTHM_AMOUNT" \
            --output "$TUNED_RESULTS_DIR/korobeiniki_test_${style_id}.wav" \
            --timbre "$TIMBRE"
    done

    for style_file in "${style_files[@]}"; do
        style_id="$(basename "$style_file" .json)"

        echo "    Twinkle <- ${style_id}"
        "$PYTHON" "$SCRIPT_DIR/scaleify.py" "$TWINKLE" \
            --style "$style_id" \
            --style-dir "$STYLE_DIR/" \
            --root C \
            --style-amount "$STYLE_AMOUNT" \
            --rhythm-amount "$RHYTHM_AMOUNT" \
            --output "$TUNED_RESULTS_DIR/twinkle_twinkle_test_${style_id}.wav" \
            --timbre "$TIMBRE"
    done
}

run_dataset() {
    local dataset="$1"
    local generator_cmd dataset_dir style_prefix
    local other_args=()

    case "$dataset" in
        japan)
            generator_cmd="$PYTHON $GENERATOR_DIR/dataset/japan_1892_dataset_generator.py"
            dataset_dir="$DATASETS_DIR/japan"
            style_prefix="japan"
            other_args=()
            ;;
        korea)
            generator_cmd="$PYTHON $GENERATOR_DIR/dataset/korea_traditional_dataset_generator.py"
            dataset_dir="$DATASETS_DIR/korea"
            style_prefix="korea"
            other_args=()
            ;;
        china)
            generator_cmd="$PYTHON $GENERATOR_DIR/dataset/china_traditional_dataset_generator.py"
            dataset_dir="$DATASETS_DIR/china"
            style_prefix="china"
            other_args=()
            ;;
        jsmel|japan-jsmel|japan_jsmel)
            dataset="jsmel"
            generator_cmd="$PYTHON $GENERATOR_DIR/dataset/jsmel_pd_dataset_generator.py"
            dataset_dir="$DATASETS_DIR/japan_jsmel"
            style_prefix="japan_jsmel"
            other_args=()
            ;;
        vocaloid)
            # Keep sources separate from generated training WAVs.
            # The generator writes output to dataset/vocaloid by default.
            generator_cmd="$PYTHON $GENERATOR_DIR/dataset/vocaloid_dataset_generator.py $DATASETS_DIR/vocaloid/sources --preset official --accept-source-terms --output $DATASETS_DIR/vocaloid"
            dataset_dir="$DATASETS_DIR/vocaloid"
            style_prefix="vocaloid"
            other_args=("--scale-max-notes" "12")
            ;;
        arabic)
            generator_cmd="$PYTHON $GENERATOR_DIR/dataset/arabic_taqasim_dataset_generator.py"
            dataset_dir="$DATASETS_DIR/arabic"
            style_prefix="arabic"
            other_args=("--scale-max-notes" "12")
            ;;
        asia)
            if [[ ! -d "$DATASETS_DIR/asia" ]]; then
                mkdir -p "$DATASETS_DIR/asia"
            fi
            if [[ ! -d "$DATASETS_DIR/japan" || ! -d "$DATASETS_DIR/korea" || ! -d "$DATASETS_DIR/china" ]]; then
                die "Missing one or more required datasets for 'asia': japan, korea, china"
            else
                echo "==> [asia] Using existing datasets: japan, korea, china"
                rm -rf "$DATASETS_DIR/asia"/* || true
                cp -r "$DATASETS_DIR/japan"/* "$DATASETS_DIR/asia/"
                cp -r "$DATASETS_DIR/korea"/* "$DATASETS_DIR/asia/"
                cp -r "$DATASETS_DIR/china"/* "$DATASETS_DIR/asia/"
            fi

            generator_cmd="null"
            dataset_dir="$DATASETS_DIR/asia"
            style_prefix="asia"
            other_args=("--scale-max-notes" "12" "--max-clusters" "8")
            ;;
        *)
            die "Unknown dataset: $dataset"
            ;;
    esac

    echo
    echo "============================================================"
    echo "Scaleify tune test: ${dataset}"
    echo "============================================================"

    if [[ "$generator_cmd" == "null" ]]; then
        echo "==> [${dataset}] Skipping dataset generation (already exists)"
    else
        echo "==> [${dataset}] Dataset generator command: $generator_cmd"
        run_generator "$dataset" "$generator_cmd" "$dataset_dir"
    fi

    has_wavs "$dataset_dir" || die "[${dataset}] No WAV files found after generation: $dataset_dir"

    run_training "$dataset" "$dataset_dir" "$style_prefix" "${other_args[@]}"
    run_listening_tests "$dataset" "$style_prefix"

    echo "==> [${dataset}] Done"
}

[[ $# -ge 1 ]] || {
    usage
    exit 1
}

cd "$SCRIPT_DIR/.." || die "Failed to cd to repository root: $SCRIPT_DIR/.."

ensure_venv

TARGET="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --download)
            FORCE_DOWNLOAD=1
            ;;
        --training)
            FORCE_TRAINING=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
    shift
done

case "$TARGET" in
    all)
        for dataset in japan korea china jsmel vocaloid arabic asia; do
            run_dataset "$dataset"
        done
        ;;
    japan|korea|china|jsmel|japan-jsmel|japan_jsmel|vocaloid|arabic|asia)
        run_dataset "$TARGET"
        ;;
    -h|--help)
        usage
        ;;
    *)
        die "Unknown dataset: $TARGET (use --help)"
        ;;
esac

cd "$CURRENT_DIR" || die "Failed to cd back to original directory: $CURRENT_DIR"