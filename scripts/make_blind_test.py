#!/usr/bin/env python3

import argparse
import csv
import random
import re
import shutil
from pathlib import Path


def get_case_name(filename: str) -> str | None:
    """
    Examples:
      erika_test.wav
        -> erika_test

      erika_test_china_cluster_1_v9_1.wav
        -> erika_test

      twinkle_twinkle_test_vocaloid_cluster_2_v9_1.wav
        -> twinkle_twinkle_test
    """
    match = re.match(r"^(.+?_test)(?:_|\.wav$)", filename, re.IGNORECASE)
    if not match:
        return None

    return match.group(1)


def extract_condition(filename: str, case_name: str) -> str:
    stem = Path(filename).stem

    if stem == case_name:
        return "original"

    prefix = case_name + "_"
    if stem.startswith(prefix):
        return stem[len(prefix):]

    return stem


def main():
    parser = argparse.ArgumentParser(
        description="Create randomized blind-listening WAV sets."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing Scaleify listening-test WAV files",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/blind_test"),
        help="Output directory (default: results/blind_test)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible shuffling",
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of digits in blind IDs (default: 3)",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    rng = random.Random(args.seed)

    # ---------------------------------------------------------
    # Find and group WAV files
    # ---------------------------------------------------------

    cases: dict[str, list[Path]] = {}

    for wav in sorted(input_dir.glob("*.wav")):
        case_name = get_case_name(wav.name)

        if case_name is None:
            print(f"Skipping unmatched file: {wav.name}")
            continue

        cases.setdefault(case_name, []).append(wav)

    if not cases:
        raise SystemExit("No matching *_test*.wav files found.")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_mapping_rows = []

    # ---------------------------------------------------------
    # Create one randomized set per case
    # ---------------------------------------------------------

    for case_name, files in sorted(cases.items()):
        case_dir = output_dir / case_name

        if case_dir.exists():
            shutil.rmtree(case_dir)

        case_dir.mkdir(parents=True)

        raw_file = None
        transformed_files = []

        for source in files:
            if source.stem == case_name:
                raw_file = source
            else:
                transformed_files.append(source)

        if raw_file is None:
            print(f"Warning: raw file not found for {case_name}")
        else:
            shutil.copy2(raw_file, case_dir / "_raw.wav")

        shuffled = transformed_files.copy()
        rng.shuffle(shuffled)

        rows = []

        for index, source in enumerate(shuffled, start=1):
            blind_id = f"{index:0{args.digits}d}"
            destination = case_dir / f"{blind_id}.wav"

            shutil.copy2(source, destination)

            condition = extract_condition(source.name, case_name)

            row = {
                "case": case_name,
                "blind_id": blind_id,
                "blind_filename": destination.name,
                "original_filename": source.name,
                "condition": condition,
            }

            rows.append(row)
            all_mapping_rows.append(row)

        # Per-case secret key
        mapping_file = case_dir / "mapping.csv"

        with mapping_file.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "case",
                    "blind_id",
                    "blind_filename",
                    "original_filename",
                    "condition",
                    "is_original",
                ],
            )

            writer.writeheader()
            writer.writerows(rows)

        print(
            f"{case_name}: "
            f"{len(files)} files -> {case_dir}"
        )

    # ---------------------------------------------------------
    # Master mapping table
    # ---------------------------------------------------------

    master_mapping = output_dir / "mapping_all.csv"

    with master_mapping.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "blind_id",
                "blind_filename",
                "original_filename",
                "condition",
                "is_original",
            ],
        )

        writer.writeheader()
        writer.writerows(all_mapping_rows)

    print()
    print("Done.")
    print(f"Output:  {output_dir}")
    print(f"Mapping: {master_mapping}")

    if args.seed is not None:
        print(f"Seed:    {args.seed}")


if __name__ == "__main__":
    main()