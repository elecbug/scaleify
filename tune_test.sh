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

STYLE_DIR="data/styles_tuned"
ERIKA="test/erika_test.wav"
KOROBEINIKI="test/korobeiniki_test.wav"
TWINKLE="test/twinkle_twinkle_test.wav"

FORCE_DOWNLOAD=0
FORCE_TRAINING=0

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
    mkdir -p "$STYLE_DIR"
    rm -f "${STYLE_DIR}"/*.csv
    rm -f "${STYLE_DIR}"/*_report.json
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
        python3 train_style.py \
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
        python3 train_style.py \
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
        python3 scaleify.py "$ERIKA" \
            --style "$style_id" \
            --style-dir "$STYLE_DIR/" \
            --root G \
            --style-amount 0.9 \
            --rhythm-amount 0.55 \
            --timbre reed
    done

    for style_file in "${style_files[@]}"; do
        style_id="$(basename "$style_file" .json)"

        echo "    Korobeiniki <- ${style_id}"
        python3 scaleify.py "$KOROBEINIKI" \
            --style "$style_id" \
            --style-dir "$STYLE_DIR/" \
            --root A \
            --style-amount 0.9 \
            --rhythm-amount 0.55 \
            --timbre reed
    done

    for style_file in "${style_files[@]}"; do
        style_id="$(basename "$style_file" .json)"

        echo "    Twinkle <- ${style_id}"
        python3 scaleify.py "$TWINKLE" \
            --style "$style_id" \
            --style-dir "$STYLE_DIR/" \
            --root C \
            --style-amount 0.9 \
            --rhythm-amount 0.55 \
            --timbre reed
    done
}

run_dataset() {
    local dataset="$1"
    local generator_cmd dataset_dir style_prefix
    local other_args=()

    case "$dataset" in
        japan)
            generator_cmd="python3 generator/dataset/japan_1892_dataset_generator.py"
            dataset_dir="dataset/japan"
            style_prefix="japan"
            other_args=()
            ;;
        korea)
            generator_cmd="python3 generator/dataset/korea_traditional_dataset_generator.py"
            dataset_dir="dataset/korea"
            style_prefix="korea"
            other_args=()
            ;;
        china)
            generator_cmd="python3 generator/dataset/china_traditional_dataset_generator.py"
            dataset_dir="dataset/china"
            style_prefix="china"
            other_args=()
            ;;
        jsmel|japan-jsmel|japan_jsmel)
            dataset="jsmel"
            generator_cmd="python3 generator/dataset/jsmel_pd_dataset_generator.py"
            dataset_dir="dataset/japan_jsmel"
            style_prefix="japan_jsmel"
            other_args=()
            ;;
        vocaloid)
            # Keep sources separate from generated training WAVs.
            # The generator writes output to dataset/vocaloid by default.
            generator_cmd="python3 generator/dataset/vocaloid_dataset_generator.py dataset/vocaloid/sources --preset official --accept-source-terms --output dataset/vocaloid"
            dataset_dir="dataset/vocaloid"
            style_prefix="vocaloid"
            other_args=("--scale-max-notes" "12")
            ;;
        *)
            die "Unknown dataset: $dataset"
            ;;
    esac

    echo
    echo "============================================================"
    echo "Scaleify tune test: ${dataset}"
    echo "============================================================"

    run_generator "$dataset" "$generator_cmd" "$dataset_dir"

    has_wavs "$dataset_dir" || die "[${dataset}] No WAV files found after generation: $dataset_dir"

    run_training "$dataset" "$dataset_dir" "$style_prefix" "${other_args[@]}"
    run_listening_tests "$dataset" "$style_prefix"

    echo "==> [${dataset}] Done"
}

[[ $# -ge 1 ]] || {
    usage
    exit 1
}

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
        for dataset in japan korea china jsmel vocaloid; do
            run_dataset "$dataset"
        done
        ;;
    japan|korea|china|jsmel|japan-jsmel|japan_jsmel|vocaloid)
        run_dataset "$TARGET"
        ;;
    -h|--help)
        usage
        ;;
    *)
        die "Unknown dataset: $TARGET (use --help)"
        ;;
esac