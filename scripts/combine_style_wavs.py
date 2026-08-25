#!/usr/bin/env python3
"""
combine_style_wavs.py

Combine two Scaleify outputs generated from the same source melody:
- bass-profile transformed WAV
- main/lead-profile transformed WAV

Default behavior:
- lower the bass result by 12 semitones
- keep the main result at its original register
- RMS-balance both tracks
- sum them
- soft-limit and peak-normalize

Dependencies:
    numpy
    scipy
    librosa
    soundfile
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Combine bass-profile and main-profile Scaleify WAV files "
            "with independent register shifting."
        )
    )

    p.add_argument("bass", type=Path, help="Bass-profile WAV")
    p.add_argument("main", type=Path, help="Main/lead-profile WAV")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output WAV",
    )

    p.add_argument(
        "--bass-semitones",
        type=float,
        default=-12.0,
        help=(
            "Pitch shift applied to the bass-profile WAV. "
            "Default: -12 semitones."
        ),
    )
    p.add_argument(
        "--main-semitones",
        type=float,
        default=0.0,
        help=(
            "Pitch shift applied to the main-profile WAV. "
            "Default: 0 semitones."
        ),
    )

    p.add_argument(
        "--bass-gain",
        type=float,
        default=0.85,
        help="Linear gain for bass after RMS matching. Default: 0.85",
    )
    p.add_argument(
        "--main-gain",
        type=float,
        default=1.0,
        help="Linear gain for main after RMS matching. Default: 1.0",
    )
    p.add_argument(
        "--bass-rms-ratio",
        type=float,
        default=0.75,
        help=(
            "Target bass RMS relative to main RMS before final gains. "
            "Default: 0.75"
        ),
    )

    p.add_argument(
        "--crossover",
        type=float,
        default=0.0,
        help=(
            "Optional crossover frequency in Hz. "
            "Bass gets low-pass, main gets high-pass. "
            "0 disables crossover. Default: 0"
        ),
    )
    p.add_argument(
        "--filter-order",
        type=int,
        default=4,
        help="Butterworth crossover order. Default: 4",
    )

    p.add_argument(
        "--bass-delay-ms",
        type=float,
        default=0.0,
        help="Delay bass by milliseconds. Negative values advance it.",
    )
    p.add_argument(
        "--main-delay-ms",
        type=float,
        default=0.0,
        help="Delay main by milliseconds. Negative values advance it.",
    )

    p.add_argument(
        "--soft-limit",
        type=float,
        default=1.4,
        help=(
            "Tanh soft-limiter drive. "
            "0 disables soft limiting. Default: 1.4"
        ),
    )
    p.add_argument(
        "--peak-dbfs",
        type=float,
        default=-1.0,
        help="Final peak target in dBFS. Default: -1.0",
    )

    return p.parse_args()


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        return x

    if x.ndim == 2:
        return np.mean(x, axis=1)

    raise ValueError(f"Unsupported audio shape: {x.shape}")


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, always_2d=False)
    return to_mono(x), int(sr)


def resample_audio(
    x: np.ndarray,
    src_sr: int,
    dst_sr: int,
) -> np.ndarray:
    if src_sr == dst_sr:
        return x

    g = math.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g

    return resample_poly(x, up, down)


def pitch_shift_audio(
    x: np.ndarray,
    sr: int,
    semitones: float,
) -> np.ndarray:
    """
    Shift pitch while preserving duration.

    librosa.effects.pitch_shift uses a phase-vocoder based process.
    For octave/register placement of synthesized Scaleify outputs,
    this is sufficient and keeps both transformed tracks time-aligned.
    """
    if abs(semitones) < 1e-9:
        return x

    shifted = librosa.effects.pitch_shift(
        y=np.asarray(x, dtype=np.float32),
        sr=sr,
        n_steps=float(semitones),
        bins_per_octave=12,
    )

    # librosa should preserve duration, but force the original sample count
    # so both branches remain exactly alignable.
    if len(shifted) > len(x):
        shifted = shifted[:len(x)]
    elif len(shifted) < len(x):
        shifted = np.pad(
            shifted,
            (0, len(x) - len(shifted)),
        )

    return np.asarray(shifted, dtype=np.float64)


def apply_delay(
    x: np.ndarray,
    sr: int,
    delay_ms: float,
) -> np.ndarray:
    samples = int(round(sr * delay_ms / 1000.0))

    if samples == 0:
        return x

    if samples > 0:
        return np.pad(x, (samples, 0))

    advance = -samples

    if advance >= len(x):
        return np.zeros(1, dtype=x.dtype)

    return x[advance:]


def pad_to_same_length(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = max(len(a), len(b))

    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))

    if len(b) < n:
        b = np.pad(b, (0, n - len(b)))

    return a, b


def rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0

    return float(
        np.sqrt(
            np.mean(np.square(x)) + EPS
        )
    )


def crossover_filter(
    bass: np.ndarray,
    main: np.ndarray,
    sr: int,
    cutoff_hz: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cutoff_hz <= 0:
        return bass, main

    nyquist = sr / 2.0

    if not 20.0 < cutoff_hz < nyquist * 0.95:
        raise ValueError(
            f"Invalid crossover {cutoff_hz} Hz "
            f"for sample rate {sr} Hz"
        )

    low_sos = butter(
        order,
        cutoff_hz,
        btype="lowpass",
        fs=sr,
        output="sos",
    )

    high_sos = butter(
        order,
        cutoff_hz,
        btype="highpass",
        fs=sr,
        output="sos",
    )

    bass_out = sosfiltfilt(
        low_sos,
        bass,
    )

    main_out = sosfiltfilt(
        high_sos,
        main,
    )

    return bass_out, main_out


def balance_tracks(
    bass: np.ndarray,
    main: np.ndarray,
    bass_rms_ratio: float,
    bass_gain: float,
    main_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    bass_rms = rms(bass)
    main_rms = rms(main)

    if bass_rms > EPS and main_rms > EPS:
        target_bass_rms = (
            main_rms * bass_rms_ratio
        )

        bass = bass * (
            target_bass_rms / bass_rms
        )

    bass = bass * bass_gain
    main = main * main_gain

    return bass, main


def soft_limit(
    x: np.ndarray,
    drive: float,
) -> np.ndarray:
    if drive <= 0:
        return x

    denom = math.tanh(drive)

    return np.tanh(
        x * drive
    ) / max(denom, EPS)


def normalize_peak(
    x: np.ndarray,
    peak_dbfs: float,
) -> np.ndarray:
    target = 10.0 ** (
        peak_dbfs / 20.0
    )

    peak = (
        float(np.max(np.abs(x)))
        if len(x)
        else 0.0
    )

    if peak <= EPS:
        return x

    return x * (
        target / peak
    )


def main() -> int:
    args = parse_args()

    bass, bass_sr = load_wav(
        args.bass
    )

    main_track, main_sr = load_wav(
        args.main
    )

    # Preserve the higher source sample rate.
    sr = max(
        bass_sr,
        main_sr,
    )

    bass = resample_audio(
        bass,
        bass_sr,
        sr,
    )

    main_track = resample_audio(
        main_track,
        main_sr,
        sr,
    )

    print(
        f"Bass pitch shift: "
        f"{args.bass_semitones:+.2f} semitones"
    )

    print(
        f"Main pitch shift: "
        f"{args.main_semitones:+.2f} semitones"
    )

    bass = pitch_shift_audio(
        bass,
        sr,
        args.bass_semitones,
    )

    main_track = pitch_shift_audio(
        main_track,
        sr,
        args.main_semitones,
    )

    bass = apply_delay(
        bass,
        sr,
        args.bass_delay_ms,
    )

    main_track = apply_delay(
        main_track,
        sr,
        args.main_delay_ms,
    )

    bass, main_track = pad_to_same_length(
        bass,
        main_track,
    )

    bass, main_track = crossover_filter(
        bass,
        main_track,
        sr,
        args.crossover,
        args.filter_order,
    )

    bass, main_track = balance_tracks(
        bass,
        main_track,
        bass_rms_ratio=args.bass_rms_ratio,
        bass_gain=args.bass_gain,
        main_gain=args.main_gain,
    )

    mix = bass + main_track

    mix = soft_limit(
        mix,
        args.soft_limit,
    )

    mix = normalize_peak(
        mix,
        args.peak_dbfs,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        args.output,
        mix.astype(np.float32),
        sr,
        subtype="PCM_16",
    )

    print(f"Output: {args.output}")
    print(f"Sample rate: {sr} Hz")
    print(
        f"Length: "
        f"{len(mix) / sr:.3f} s"
    )
    print(
        f"Bass RMS: "
        f"{rms(bass):.6f}"
    )
    print(
        f"Main RMS: "
        f"{rms(main_track):.6f}"
    )
    print(
        f"Peak: "
        f"{np.max(np.abs(mix)):.6f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())