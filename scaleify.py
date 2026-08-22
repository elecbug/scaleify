#!/usr/bin/env python3
"""
scaleify_v2.py

Audio -> robust decode -> optional Demucs stem separation -> key/root estimation
-> scale-constrained pitch quantization -> optional heuristic ornamentation
-> stem remix.

This is a stylization tool, not an ethnomusicologically faithful
reconstruction of any traditional music system.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]

# 12-TET approximations.
SCALES: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "chinese": [0, 2, 4, 7, 9],          # major pentatonic
    "japanese_in": [0, 1, 5, 7, 8],
    "japanese_yo": [0, 2, 5, 7, 9],
    "korean_pyeongjo": [0, 2, 5, 7, 9],  # simplified approximation
    "arabic_hijaz": [0, 1, 4, 5, 7, 8, 10],
    "indian_bhairav": [0, 1, 4, 5, 7, 8, 11],
}

# These are intentionally mild, heuristic "grace note" biases.
# They are NOT claims about authentic performance practice.
@dataclass(frozen=True)
class Ornament:
    probability: float
    offset_semitones: float
    duration_ms: float


ORNAMENTS: dict[str, Ornament] = {
    "major": Ornament(0.00, 0.0, 0.0),
    "chinese": Ornament(0.16, +2.0, 45.0),
    "japanese_in": Ornament(0.22, +1.0, 55.0),
    "japanese_yo": Ornament(0.15, -2.0, 45.0),
    "korean_pyeongjo": Ornament(0.14, -2.0, 60.0),
    "arabic_hijaz": Ornament(0.22, -1.0, 50.0),
    "indian_bhairav": Ornament(0.18, +1.0, 55.0),
}

MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
], dtype=np.float64)

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
], dtype=np.float64)


def ensure_channels_first(y: np.ndarray) -> np.ndarray:
    """Return audio as float32 array shaped (channels, samples)."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    if y.ndim != 2:
        raise ValueError(f"Expected mono/stereo audio, got shape={y.shape}")
    return y


def mono_mix(y: np.ndarray) -> np.ndarray:
    return ensure_channels_first(y).mean(axis=0)


def parse_root(root: str) -> int | None:
    if root.lower() == "auto":
        return None

    name = root.strip().upper()
    aliases = {
        "DB": "C#",
        "EB": "D#",
        "GB": "F#",
        "AB": "G#",
        "BB": "A#",
    }
    name = aliases.get(name, name)

    if name not in NOTE_NAMES:
        raise ValueError(f"Unknown root note: {root}")
    return NOTE_NAMES.index(name)


def detect_key(y: np.ndarray, sr: int) -> tuple[int, str]:
    """
    Simple chroma + Krumhansl-Schmuckler-style key estimate.
    Only the root is used for non-major target scales.
    """
    mono = mono_mix(y)

    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    norm = np.linalg.norm(chroma_mean)
    if norm > 0:
        chroma_mean = chroma_mean / norm

    major_profile = MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE)
    minor_profile = MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE)

    best_score = -np.inf
    best_root = 0
    best_mode = "major"

    for root in range(12):
        major = np.roll(major_profile, root)
        minor = np.roll(minor_profile, root)

        major_score = float(np.dot(chroma_mean, major))
        minor_score = float(np.dot(chroma_mean, minor))

        if major_score > best_score:
            best_score = major_score
            best_root = root
            best_mode = "major"

        if minor_score > best_score:
            best_score = minor_score
            best_root = root
            best_mode = "minor"

    return best_root, best_mode


def nearest_scale_midi(
    midi_note: float,
    root_pc: int,
    scale_intervals: list[int],
) -> float:
    """
    Return the nearest 12-TET MIDI pitch whose pitch class belongs
    to the selected scale.
    """
    allowed_pcs = {(root_pc + x) % 12 for x in scale_intervals}

    center = int(round(midi_note))
    candidates = [
        note
        for note in range(center - 12, center + 13)
        if note % 12 in allowed_pcs
    ]

    return float(min(candidates, key=lambda n: abs(n - midi_note)))


def estimate_f0(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin_note: str,
    fmax_note: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mono = mono_mix(y)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        mono,
        fmin=float(librosa.note_to_hz(fmin_note)),
        fmax=float(librosa.note_to_hz(fmax_note)),
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    return f0, voiced_flag, voiced_prob


def merge_boundaries(
    samples: list[int],
    n_samples: int,
    min_gap_samples: int,
) -> np.ndarray:
    samples = sorted(set(max(0, min(n_samples, int(x))) for x in samples))
    out = [0]

    for x in samples:
        if x <= 0 or x >= n_samples:
            continue
        if x - out[-1] >= min_gap_samples:
            out.append(x)

    if n_samples - out[-1] < min_gap_samples and len(out) > 1:
        out[-1] = n_samples
    else:
        out.append(n_samples)

    return np.asarray(sorted(set(out)), dtype=np.int64)


def build_note_boundaries(
    y: np.ndarray,
    sr: int,
    f0: np.ndarray,
    voiced_prob: np.ndarray,
    hop_length: int,
    min_note_ms: float,
    pitch_jump_semitones: float,
) -> np.ndarray:
    """
    Combine spectral onset boundaries with persistent F0 jumps.
    This is still only a heuristic note segmentation.
    """
    mono = mono_mix(y)
    n_samples = y.shape[-1]

    onset_frames = librosa.onset.onset_detect(
        y=mono,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units="frames",
    )
    onset_samples = librosa.frames_to_samples(
        onset_frames, hop_length=hop_length
    ).tolist()

    midi = np.full_like(f0, np.nan, dtype=np.float64)
    valid = np.isfinite(f0) & (voiced_prob >= 0.55)
    midi[valid] = librosa.hz_to_midi(f0[valid])

    pitch_change_samples: list[int] = []
    last_change_frame = -999999
    debounce_frames = max(1, int(round((min_note_ms / 1000.0) * sr / hop_length)))

    for i in range(1, len(midi)):
        if not (np.isfinite(midi[i - 1]) and np.isfinite(midi[i])):
            continue
        if abs(midi[i] - midi[i - 1]) < pitch_jump_semitones:
            continue
        if i - last_change_frame < debounce_frames:
            continue

        pitch_change_samples.append(i * hop_length)
        last_change_frame = i

    min_gap_samples = max(1, int(sr * min_note_ms / 1000.0))

    return merge_boundaries(
        [*onset_samples, *pitch_change_samples],
        n_samples=n_samples,
        min_gap_samples=min_gap_samples,
    )


def pitch_shift_multichannel(
    y: np.ndarray,
    sr: int,
    n_steps: float,
) -> np.ndarray:
    """
    Apply the same pitch shift independently to every channel.
    """
    y = ensure_channels_first(y)

    if abs(n_steps) < 1e-4:
        return y.copy()

    shifted = [
        librosa.effects.pitch_shift(
            channel,
            sr=sr,
            n_steps=float(n_steps),
            bins_per_octave=12,
        )
        for channel in y
    ]
    return np.vstack(shifted).astype(np.float32, copy=False)


def edge_blend(
    original: np.ndarray,
    processed: np.ndarray,
    fade_samples: int,
) -> np.ndarray:
    """
    Blend processed audio back toward the original at segment edges
    to reduce hard clicks between independently pitch-shifted notes.
    """
    original = ensure_channels_first(original)
    processed = ensure_channels_first(processed)

    n = original.shape[-1]
    fade = min(fade_samples, n // 3)

    if fade <= 1:
        return processed

    weight = np.ones(n, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    weight[:fade] = ramp
    weight[-fade:] = ramp[::-1]

    return (
        processed * weight[np.newaxis, :]
        + original * (1.0 - weight[np.newaxis, :])
    ).astype(np.float32, copy=False)


def apply_segment_shift(
    segment: np.ndarray,
    sr: int,
    main_shift: float,
    fade_ms: float,
    ornament: Ornament | None,
    ornament_amount: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Pitch-shift one detected note segment.

    Ornamentation is a deliberately conservative heuristic:
    a short grace-note-like offset at the beginning of some notes.
    """
    segment = ensure_channels_first(segment)
    n = segment.shape[-1]

    shifted = pitch_shift_multichannel(segment, sr, main_shift)

    if (
        ornament is not None
        and ornament_amount > 0.0
        and ornament.probability > 0.0
        and rng.random() < ornament.probability * ornament_amount
    ):
        grace_n = min(
            n // 3,
            max(0, int(sr * ornament.duration_ms / 1000.0)),
        )

        if grace_n >= 64:
            grace_shift = (
                main_shift
                + ornament.offset_semitones * ornament_amount
            )
            grace = pitch_shift_multichannel(
                segment[:, :grace_n],
                sr,
                grace_shift,
            )

            # Small crossfade from grace pitch to main pitch.
            xf = min(max(16, int(sr * 0.012)), grace_n // 2)
            if xf > 1:
                w = np.linspace(
                    1.0, 0.0, xf, dtype=np.float32
                )[np.newaxis, :]
                shifted[:, grace_n - xf:grace_n] = (
                    grace[:, grace_n - xf:grace_n] * w
                    + shifted[:, grace_n - xf:grace_n] * (1.0 - w)
                )
                shifted[:, :grace_n - xf] = grace[:, :grace_n - xf]
            else:
                shifted[:, :grace_n] = grace

    fade_samples = int(sr * fade_ms / 1000.0)
    return edge_blend(segment, shifted, fade_samples)


def quantize_stem(
    stem: np.ndarray,
    sr: int,
    root_pc: int,
    style: str,
    strength: float,
    ornament_amount: float,
    hop_length: int,
    min_note_ms: float,
    pitch_jump_semitones: float,
    max_shift: float,
    seed: int,
    fmin_note: str,
    fmax_note: str,
) -> np.ndarray:
    stem = ensure_channels_first(stem)
    scale = SCALES[style]

    f0, _, voiced_prob = estimate_f0(
        stem,
        sr,
        hop_length,
        fmin_note,
        fmax_note,
    )

    boundaries = build_note_boundaries(
        stem,
        sr,
        f0,
        voiced_prob,
        hop_length,
        min_note_ms,
        pitch_jump_semitones,
    )

    output = stem.copy()
    rng = np.random.default_rng(seed)
    ornament = ORNAMENTS.get(style)

    for i in range(len(boundaries) - 1):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])

        if end - start < 64:
            continue

        frame_start = max(0, start // hop_length)
        frame_end = min(len(f0), int(np.ceil(end / hop_length)) + 1)

        seg_f0 = f0[frame_start:frame_end]
        seg_prob = voiced_prob[frame_start:frame_end]

        valid = np.isfinite(seg_f0) & (seg_prob >= 0.55)
        if np.count_nonzero(valid) < 2:
            continue

        # Weighted median would be ideal, but ordinary median is robust enough here.
        median_hz = float(np.median(seg_f0[valid]))
        midi = float(librosa.hz_to_midi(median_hz))

        target_midi = nearest_scale_midi(
            midi,
            root_pc,
            scale,
        )

        shift = (target_midi - midi) * strength
        shift = float(np.clip(shift, -max_shift, max_shift))

        if abs(shift) < 0.03:
            continue

        segment = stem[:, start:end]

        output[:, start:end] = apply_segment_shift(
            segment,
            sr,
            main_shift=shift,
            fade_ms=12.0,
            ornament=ornament,
            ornament_amount=ornament_amount,
            rng=rng,
        )

    return output


def load_without_demucs(input_path: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(
        input_path,
        sr=None,
        mono=False,
    )
    return ensure_channels_first(y), int(sr)


def decode_audio_for_demucs(
    input_path: Path,
) -> tuple[np.ndarray, int]:
    """
    Decode input audio to float32 PCM shaped (channels, samples).

    First try libsndfile via soundfile. If that fails, use the FFmpeg
    executable bundled by imageio-ffmpeg. This bypasses Demucs' own file
    loader, which can fail on some MP3 bitstreams before reaching FFmpeg.
    """
    soundfile_error: Exception | None = None

    try:
        audio, sr = sf.read(
            str(input_path),
            dtype="float32",
            always_2d=True,
        )
        print("Decoded input with soundfile.")
        return ensure_channels_first(audio.T), int(sr)
    except Exception as exc:
        soundfile_error = exc
        print(
            "[warn] soundfile could not decode the input; "
            "falling back to bundled FFmpeg."
        )

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "The input could not be decoded by soundfile, and "
            "imageio-ffmpeg is not installed.\n"
            "Install it with:\n"
            "  python -m pip install -U imageio-ffmpeg\n"
            f"soundfile error: {soundfile_error}"
        ) from exc

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "imageio-ffmpeg is installed, but no usable FFmpeg executable "
            "could be found.\n"
            "Try reinstalling it with:\n"
            "  python -m pip install -U --force-reinstall imageio-ffmpeg"
        ) from exc

    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

        cmd = [
            str(ffmpeg_exe),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_f32le",
            str(tmp_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg could not decode the input audio.\n"
                f"FFmpeg output:\n{result.stderr.strip()}"
            )

        audio, sr = sf.read(
            str(tmp_path),
            dtype="float32",
            always_2d=True,
        )

        print(
            "Decoded input with bundled FFmpeg "
            f"({Path(ffmpeg_exe).name})."
        )

        return ensure_channels_first(audio.T), int(sr)

    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def separate_with_demucs(
    input_path: Path,
    model: str,
    device: str | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    try:
        import torch
        from demucs.api import Separator
    except ImportError as e:
        raise RuntimeError(
            "Demucs/PyTorch is not installed. Run:\n"
            "  python -m pip install demucs torch torchaudio"
        ) from e

    kwargs = {
        "model": model,
        "shifts": 1,
        "overlap": 0.25,
        "split": True,
    }
    if device:
        kwargs["device"] = device

    separator = Separator(**kwargs)

    decoded, decoded_sr = decode_audio_for_demucs(input_path)

    wav = torch.from_numpy(
        np.ascontiguousarray(decoded, dtype=np.float32)
    )

    print(
        f"Sending decoded PCM to Demucs: "
        f"{wav.shape[0]} ch, {decoded_sr} Hz"
    )

    origin, separated = separator.separate_tensor(
        wav,
        sr=decoded_sr,
    )

    origin_np = origin.detach().cpu().numpy().astype(np.float32)
    stems_np = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in separated.items()
    }

    return (
        ensure_channels_first(origin_np),
        stems_np,
        int(separator.samplerate),
    )


def remix(
    stems: dict[str, np.ndarray],
) -> np.ndarray:
    if not stems:
        raise ValueError("No stems to remix.")

    lengths = [x.shape[-1] for x in stems.values()]
    n = min(lengths)
    channels = max(x.shape[0] for x in stems.values())

    result = np.zeros((channels, n), dtype=np.float32)

    for stem in stems.values():
        stem = ensure_channels_first(stem)[:, :n]

        if stem.shape[0] == 1 and channels == 2:
            stem = np.repeat(stem, 2, axis=0)
        elif stem.shape[0] != channels:
            raise ValueError(
                f"Incompatible stem channels: {stem.shape[0]} vs {channels}"
            )

        result += stem

    return result


def peak_protect(
    y: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    y = ensure_channels_first(y)

    peak = float(np.max(np.abs(y)))
    if peak <= 0:
        return y

    if reference is not None:
        ref_peak = float(np.max(np.abs(reference)))
        target = min(0.99, max(0.80, ref_peak))
    else:
        target = 0.99

    if peak > target:
        y = y * (target / peak)

    return y.astype(np.float32, copy=False)


def save_wav(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = ensure_channels_first(y)
    sf.write(
        path,
        y.T,
        sr,
        subtype="PCM_24",
    )


def parse_stem_list(value: str) -> list[str]:
    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def convert_style_from_stems(
    original: np.ndarray,
    stems: dict[str, np.ndarray],
    sr: int,
    style: str,
    root_pc: int,
    stem_names: list[str],
    strength: float,
    ornament_amount: float,
    hop_length: int,
    min_note_ms: float,
    pitch_jump_semitones: float,
    max_shift: float,
    seed: int,
) -> np.ndarray:
    processed = {
        name: audio.copy()
        for name, audio in stems.items()
    }

    # Recommended ranges differ by stem.
    pitch_ranges = {
        "vocals": ("C2", "C7"),
        "other": ("C2", "C7"),
        "bass": ("E1", "C5"),
    }

    for stem_name in stem_names:
        if stem_name not in processed:
            print(
                f"[warn] stem '{stem_name}' is unavailable; "
                f"available={list(processed)}"
            )
            continue

        if stem_name == "drums":
            print("[warn] skipping drums: pitch quantization is not appropriate")
            continue

        fmin_note, fmax_note = pitch_ranges.get(
            stem_name,
            ("C2", "C7"),
        )

        print(f"  transforming stem: {stem_name}")

        processed[stem_name] = quantize_stem(
            processed[stem_name],
            sr=sr,
            root_pc=root_pc,
            style=style,
            strength=strength,
            ornament_amount=ornament_amount,
            hop_length=hop_length,
            min_note_ms=min_note_ms,
            pitch_jump_semitones=pitch_jump_semitones,
            max_shift=max_shift,
            seed=seed,
            fmin_note=fmin_note,
            fmax_note=fmax_note,
        )

    out = remix(processed)
    return peak_protect(out, reference=original)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scale-constrained musical stylizer with optional Demucs stem separation."
        )
    )

    parser.add_argument("input", type=Path)

    parser.add_argument(
        "--style",
        default="all",
        choices=[*SCALES.keys(), "all"],
    )

    parser.add_argument(
        "--root",
        default="auto",
        help="auto, C, C#, D, Eb, ...",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Scale quantization amount, recommended 0.5-1.0",
    )

    parser.add_argument(
        "--ornament",
        type=float,
        default=0.0,
        help=(
            "Heuristic ornament amount, 0.0 disables it. "
            "Try 0.25-0.6 for stronger stylization."
        ),
    )

    parser.add_argument(
        "--stems",
        default="vocals",
        help=(
            "Comma-separated Demucs stems to transform. "
            "Default: vocals. Try vocals,other for a stronger effect."
        ),
    )

    parser.add_argument(
        "--no-demucs",
        action="store_true",
        help="Treat the entire input as one melodic signal.",
    )

    parser.add_argument(
        "--demucs-model",
        default="htdemucs",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Demucs device, e.g. cuda or cpu. Default: auto.",
    )

    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--min-note-ms",
        type=float,
        default=80.0,
    )

    parser.add_argument(
        "--pitch-jump",
        type=float,
        default=0.80,
        help="F0 jump in semitones used as a possible note boundary.",
    )

    parser.add_argument(
        "--max-shift",
        type=float,
        default=3.0,
        help="Maximum absolute pitch correction per note in semitones.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1479,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )

    args = parser.parse_args()

    if not 0.0 <= args.strength <= 1.5:
        parser.error("--strength should normally be between 0.0 and 1.5")

    if not 0.0 <= args.ornament <= 1.0:
        parser.error("--ornament must be between 0.0 and 1.0")

    if args.no_demucs:
        original, sr = load_without_demucs(args.input)
        stems = {"mix": original.copy()}
        transform_stems = ["mix"]
        print(f"Loaded full mix: {original.shape}, {sr} Hz")
    else:
        print("Separating stems with Demucs...")
        original, stems, sr = separate_with_demucs(
            args.input,
            model=args.demucs_model,
            device=args.device,
        )
        transform_stems = parse_stem_list(args.stems)
        print(f"Demucs stems: {list(stems)}")
        print(f"Sample rate: {sr} Hz")

    root_pc = parse_root(args.root)
    if root_pc is None:
        # Detect key from the original mixture rather than a single separated stem.
        root_pc, mode = detect_key(original, sr)
        print(f"Detected key: {NOTE_NAMES[root_pc]} {mode}")
    else:
        print(f"Root: {NOTE_NAMES[root_pc]}")

    if args.style == "all":
        styles = [name for name in SCALES if name != "major"]
    else:
        styles = [args.style]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for style in styles:
        print(f"\n[{style}]")

        if args.no_demucs:
            transformed = quantize_stem(
                original,
                sr=sr,
                root_pc=root_pc,
                style=style,
                strength=args.strength,
                ornament_amount=args.ornament,
                hop_length=args.hop_length,
                min_note_ms=args.min_note_ms,
                pitch_jump_semitones=args.pitch_jump,
                max_shift=args.max_shift,
                seed=args.seed,
                fmin_note="C2",
                fmax_note="C7",
            )
            output = peak_protect(transformed, reference=original)
        else:
            output = convert_style_from_stems(
                original=original,
                stems=stems,
                sr=sr,
                style=style,
                root_pc=root_pc,
                stem_names=transform_stems,
                strength=args.strength,
                ornament_amount=args.ornament,
                hop_length=args.hop_length,
                min_note_ms=args.min_note_ms,
                pitch_jump_semitones=args.pitch_jump,
                max_shift=args.max_shift,
                seed=args.seed,
            )

        out_path = (
            args.output_dir
            / f"{args.input.stem}_{style}.wav"
        )
        save_wav(out_path, output, sr)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()