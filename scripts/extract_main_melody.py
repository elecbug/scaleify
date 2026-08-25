#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import essentia.standard as es
import numpy as np
import soundfile as sf

SR = 44100
FRAME_SIZE = 2048
HOP_SIZE = 128


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract predominant melody from every MP3 in a folder and save monophonic WAV files."
    )
    p.add_argument("input_dir", type=Path)
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--confidence", type=float, default=0.0,
                   help="Discard frames below this pitch confidence (default: 0.0)")
    p.add_argument("--quantize", action="store_true",
                   help="Quantize extracted pitch to equal-tempered semitones")
    return p.parse_args()


def find_mp3s(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.mp3" if recursive else "*.mp3"
    return sorted(p for p in root.glob(pattern) if p.is_file())


def smooth_voicing(mask: np.ndarray, width: int = 5) -> np.ndarray:
    if len(mask) == 0:
        return mask.astype(np.float64)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(mask.astype(np.float64), kernel, mode="same")


def synthesize_pitch_contour(
    pitch: np.ndarray,
    confidence: np.ndarray,
    n_samples: int,
    confidence_threshold: float,
    quantize: bool,
) -> np.ndarray:
    pitch = np.asarray(pitch, dtype=np.float64).copy()
    confidence = np.asarray(confidence, dtype=np.float64)

    voiced = (pitch > 0.0) & np.isfinite(pitch)
    if confidence_threshold > 0.0:
        voiced &= confidence >= confidence_threshold

    pitch[~voiced] = 0.0

    if quantize and np.any(voiced):
        midi = 69.0 + 12.0 * np.log2(pitch[voiced] / 440.0)
        midi = np.round(midi)
        pitch[voiced] = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

    frame_pos = np.arange(len(pitch), dtype=np.float64) * HOP_SIZE
    sample_pos = np.arange(n_samples, dtype=np.float64)

    valid_idx = np.flatnonzero(voiced)
    if len(valid_idx) < 2:
        return np.zeros(n_samples, dtype=np.float32)

    # Interpolate the detected F0 values across time, then apply a voiced mask.
    interp_pitch = np.interp(
        sample_pos,
        frame_pos[valid_idx],
        pitch[valid_idx],
        left=pitch[valid_idx[0]],
        right=pitch[valid_idx[-1]],
    )

    frame_mask = voiced.astype(np.float64)
    sample_mask = np.interp(
        sample_pos,
        frame_pos,
        frame_mask,
        left=0.0,
        right=0.0,
    )
    sample_mask = smooth_voicing(sample_mask >= 0.5, width=max(3, int(0.008 * SR)))
    sample_mask = np.clip(sample_mask, 0.0, 1.0)

    phase = 2.0 * np.pi * np.cumsum(interp_pitch) / SR

    # Neutral harmonic tone: easy for later pitch tracking, without copying timbre.
    audio = (
        0.82 * np.sin(phase)
        + 0.12 * np.sin(2.0 * phase)
        + 0.06 * np.sin(3.0 * phase)
    )
    audio *= sample_mask

    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio / peak * 0.86

    return audio.astype(np.float32)


def process_file(src: Path, dst: Path, args) -> tuple[int, float]:
    loader = es.EqloudLoader(filename=str(src), sampleRate=SR)
    audio = loader()

    extractor = es.PredominantPitchMelodia(
        frameSize=FRAME_SIZE,
        hopSize=HOP_SIZE,
        guessUnvoiced=False,
    )
    pitch, confidence = extractor(audio)

    pitch = np.asarray(pitch)
    confidence = np.asarray(confidence)
    voiced = (pitch > 0.0) & np.isfinite(pitch)
    if args.confidence > 0.0:
        voiced &= confidence >= args.confidence

    melody = synthesize_pitch_contour(
        pitch=pitch,
        confidence=confidence,
        n_samples=len(audio),
        confidence_threshold=args.confidence,
        quantize=args.quantize,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, melody, SR, subtype="PCM_16")

    return int(np.sum(voiced)), float(np.mean(voiced)) if len(voiced) else 0.0


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = find_mp3s(input_dir, args.recursive)
    if not files:
        raise SystemExit(f"No MP3 files found in: {input_dir}")

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files : {len(files)}")

    failed = 0

    for i, src in enumerate(files, 1):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel.with_suffix(".wav")

        print(f"[{i}/{len(files)}] {rel}")
        try:
            voiced_frames, voiced_ratio = process_file(src, dst, args)
            print(
                f"  -> {dst.relative_to(output_dir)} "
                f"| voiced={voiced_ratio:.1%} ({voiced_frames} frames)"
            )
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")

    print(f"Done: success={len(files) - failed}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())