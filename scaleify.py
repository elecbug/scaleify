#!/usr/bin/env python3
"""
scaleify_symbolic_v6.py

Clean monophonic scale stylizer.

Unlike the previous phase-vocoder version, this script does not pitch-shift
the original waveform. It detects isolated note regions, estimates one pitch
per note, maps the note to a target scale, and synthesizes a clean replacement.

Best suited to:
- monophonic test melodies
- isolated flute/whistle/synth lines
- melodies with short gaps between notes

Not suited to:
- full mixes
- chords/polyphonic instruments
- legato vocals with no detectable note boundaries
"""

from __future__ import annotations

import argparse
import csv
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

# Pitch-class offsets from the selected root in 12-tone equal temperament.
#
# Important:
# - Entries with cultural/traditional names are only 12-TET pitch-set
#   approximations. They do not reproduce tuning, melodic grammar,
#   ascending/descending forms, characteristic tones, or ornamentation.
# - Duplicate pitch sets are intentional. Different names can represent
#   different theoretical or stylistic interpretations of the same notes.
SCALES: dict[str, list[int]] = {
    # ------------------------------------------------------------
    # Western diatonic modes and common minor forms
    # ------------------------------------------------------------
    "major": [0, 2, 4, 5, 7, 9, 11],
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],

    # ------------------------------------------------------------
    # Pentatonic, blues, synthetic, and symmetric scales
    # ------------------------------------------------------------
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "octatonic_wh": [0, 2, 3, 5, 6, 8, 9, 11],
    "octatonic_hw": [0, 1, 3, 4, 6, 7, 9, 10],
    "hungarian_minor": [0, 2, 3, 6, 7, 8, 11],
    "double_harmonic": [0, 1, 4, 5, 7, 8, 11],
    "neapolitan_minor": [0, 1, 3, 5, 7, 8, 11],
    "lydian_dominant": [0, 2, 4, 6, 7, 9, 10],
    "enigmatic": [0, 1, 4, 6, 8, 10, 11],

    # ------------------------------------------------------------
    # Chinese pentatonic modes, represented in 12-TET
    # ------------------------------------------------------------
    "chinese": [0, 2, 4, 7, 9],
    "chinese_gong": [0, 2, 4, 7, 9],
    "chinese_shang": [0, 2, 5, 7, 10],
    "chinese_jue": [0, 3, 5, 8, 10],
    "chinese_zhi": [0, 2, 5, 7, 9],
    "chinese_yu": [0, 3, 5, 7, 10],

    # ------------------------------------------------------------
    # Japanese pentatonic pitch-set approximations
    # Naming/formulas vary by source and modal rotation.
    # ------------------------------------------------------------
    "japanese_in": [0, 1, 5, 7, 8],
    "japanese_yo": [0, 2, 5, 7, 9],
    "hirajoshi_12tet": [0, 2, 3, 7, 8],
    "in_sen_12tet": [0, 1, 5, 7, 10],
    "iwato_12tet": [0, 1, 5, 6, 10],
    "kumoi_12tet": [0, 2, 3, 7, 9],

    # ------------------------------------------------------------
    # Korean mode approximations
    # ------------------------------------------------------------
    "korean_pyeongjo": [0, 2, 5, 7, 9],
    "korean_gyemyeonjo_approx": [0, 3, 5, 7, 10],

    # ------------------------------------------------------------
    # Arabic / Middle Eastern 12-TET approximations
    # Quarter-tones and maqam melodic pathways are not represented.
    # ------------------------------------------------------------
    "arabic_hijaz": [0, 1, 4, 5, 7, 8, 10],
    "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],
    "maqam_ajam_12tet": [0, 2, 4, 5, 7, 9, 11],
    "maqam_kurd_12tet": [0, 1, 3, 5, 7, 8, 10],
    "maqam_nahawand_12tet": [0, 2, 3, 5, 7, 8, 10],
    "maqam_nikriz_12tet": [0, 2, 3, 6, 7, 9, 10],
    "maqam_nawa_athar_12tet": [0, 2, 3, 6, 7, 8, 11],
    "maqam_zanjaran_12tet": [0, 1, 4, 5, 7, 9, 10],
    "hijazkar_12tet": [0, 1, 4, 5, 7, 8, 11],

    # ------------------------------------------------------------
    # Hindustani thaat/raga pitch-set approximations
    # Raga-specific ascent, descent, vadi/samvadi, and phrases are omitted.
    # ------------------------------------------------------------
    "indian_bhairav": [0, 1, 4, 5, 7, 8, 11],
    "indian_bhairavi": [0, 1, 3, 5, 7, 8, 10],
    "indian_kafi": [0, 2, 3, 5, 7, 9, 10],
    "indian_khamaj": [0, 2, 4, 5, 7, 9, 10],
    "indian_kalyan": [0, 2, 4, 6, 7, 9, 11],
    "indian_marwa": [0, 1, 4, 6, 7, 9, 11],
    "indian_purvi": [0, 1, 4, 6, 7, 8, 11],
    "indian_todi": [0, 1, 3, 6, 7, 8, 11],

    # ------------------------------------------------------------
    # Gamelan-inspired coarse 12-TET approximations
    # Real slendro/pelog tunings vary by ensemble and are not 12-TET.
    # ------------------------------------------------------------
    "slendro_approx": [0, 2, 5, 7, 9],
    "pelog_approx": [0, 1, 3, 7, 8],
}


@dataclass(frozen=True)
class StyleRule:
    grace_probability: float
    grace_scale_steps: int
    grace_fraction: float
    vibrato_cents: float
    vibrato_hz: float


# Deliberately stylized heuristics, not authentic performance models.
# Scales not listed here use DEFAULT_STYLE_RULE instead of failing.
DEFAULT_STYLE_RULE = StyleRule(
    grace_probability=0.12,
    grace_scale_steps=+1,
    grace_fraction=0.10,
    vibrato_cents=5.0,
    vibrato_hz=5.2,
)

STYLE_RULES: dict[str, StyleRule] = {
    # Neutral / Western
    "major": StyleRule(0.00, 0, 0.00, 0.0, 0.0),
    "ionian": StyleRule(0.00, 0, 0.00, 0.0, 0.0),
    "natural_minor": StyleRule(0.08, -1, 0.08, 4.0, 5.0),
    "harmonic_minor": StyleRule(0.18, -1, 0.10, 7.0, 5.2),
    "melodic_minor": StyleRule(0.10, +1, 0.08, 5.0, 5.0),
    "blues": StyleRule(0.32, -1, 0.12, 18.0, 5.4),
    "hungarian_minor": StyleRule(0.28, -1, 0.12, 14.0, 5.3),
    "double_harmonic": StyleRule(0.35, -1, 0.12, 13.0, 5.4),

    # Chinese
    "chinese": StyleRule(0.35, +1, 0.14, 10.0, 5.0),
    "chinese_gong": StyleRule(0.35, +1, 0.14, 10.0, 5.0),
    "chinese_shang": StyleRule(0.32, -1, 0.13, 10.0, 5.0),
    "chinese_jue": StyleRule(0.38, +1, 0.14, 11.0, 5.1),
    "chinese_zhi": StyleRule(0.30, -1, 0.13, 9.0, 5.0),
    "chinese_yu": StyleRule(0.36, -1, 0.15, 12.0, 4.9),

    # Japanese
    "japanese_in": StyleRule(0.55, -1, 0.16, 7.0, 5.4),
    "japanese_yo": StyleRule(0.30, +1, 0.13, 6.0, 5.2),
    "hirajoshi_12tet": StyleRule(0.52, -1, 0.16, 8.0, 5.4),
    "in_sen_12tet": StyleRule(0.58, -1, 0.17, 8.0, 5.5),
    "iwato_12tet": StyleRule(0.48, +1, 0.15, 7.0, 5.3),
    "kumoi_12tet": StyleRule(0.42, -1, 0.14, 7.0, 5.2),

    # Korean
    "korean_pyeongjo": StyleRule(0.38, -1, 0.18, 22.0, 4.6),
    "korean_gyemyeonjo_approx": StyleRule(0.46, -1, 0.20, 28.0, 4.4),

    # Arabic / Middle Eastern approximations
    "arabic_hijaz": StyleRule(0.50, +1, 0.13, 18.0, 5.8),
    "phrygian_dominant": StyleRule(0.46, +1, 0.13, 16.0, 5.7),
    "maqam_kurd_12tet": StyleRule(0.34, -1, 0.13, 15.0, 5.6),
    "maqam_nahawand_12tet": StyleRule(0.30, +1, 0.12, 14.0, 5.5),
    "maqam_nikriz_12tet": StyleRule(0.44, +1, 0.13, 18.0, 5.7),
    "maqam_nawa_athar_12tet": StyleRule(0.48, -1, 0.14, 19.0, 5.8),
    "maqam_zanjaran_12tet": StyleRule(0.48, +1, 0.13, 18.0, 5.8),
    "hijazkar_12tet": StyleRule(0.52, -1, 0.14, 18.0, 5.8),

    # Indian approximations
    "indian_bhairav": StyleRule(0.45, -1, 0.15, 28.0, 5.0),
    "indian_bhairavi": StyleRule(0.42, -1, 0.15, 26.0, 5.0),
    "indian_kafi": StyleRule(0.38, +1, 0.14, 24.0, 5.0),
    "indian_khamaj": StyleRule(0.36, -1, 0.14, 22.0, 5.1),
    "indian_kalyan": StyleRule(0.40, +1, 0.15, 24.0, 5.1),
    "indian_marwa": StyleRule(0.44, -1, 0.15, 27.0, 5.0),
    "indian_purvi": StyleRule(0.46, -1, 0.16, 28.0, 5.0),
    "indian_todi": StyleRule(0.48, -1, 0.16, 30.0, 4.9),

    # Gamelan-inspired approximations
    "slendro_approx": StyleRule(0.22, +1, 0.12, 8.0, 5.0),
    "pelog_approx": StyleRule(0.32, -1, 0.14, 9.0, 5.1),
}


def parse_root(root: str) -> int:
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


def ensure_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    return np.mean(y, axis=1 if y.shape[1] <= 2 else 0).astype(np.float32)


def detect_note_regions(
    y: np.ndarray,
    top_db: float,
    frame_length: int,
    hop_length: int,
    min_note_ms: float,
    sr: int,
) -> np.ndarray:
    """
    Detect audible note regions separated by low-energy gaps.
    """
    regions = librosa.effects.split(
        y,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    min_samples = int(sr * min_note_ms / 1000.0)
    return np.asarray(
        [
            (int(start), int(end))
            for start, end in regions
            if end - start >= min_samples
        ],
        dtype=np.int64,
    )


def estimate_fundamental_fft(
    segment: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
) -> float:
    """
    Estimate a monophonic note pitch.

    The harmonic-product spectrum helps avoid selecting a strong overtone.
    """
    segment = np.asarray(segment, dtype=np.float64)

    trim = int(0.04 * sr)
    if len(segment) > 2 * trim + 256:
        segment = segment[trim:-trim]

    segment = segment - np.mean(segment)

    if len(segment) < 256 or np.max(np.abs(segment)) < 1e-6:
        return np.nan

    window = np.hanning(len(segment))
    n_fft = 1 << int(np.ceil(np.log2(max(2048, len(segment) * 4))))

    spectrum = np.abs(np.fft.rfft(segment * window, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    valid = (freqs >= fmin) & (freqs <= fmax)
    indices = np.flatnonzero(valid)

    if len(indices) == 0:
        return np.nan

    hps = spectrum.copy()
    for factor in (2, 3, 4):
        down = spectrum[::factor]
        hps[:len(down)] *= down

    local = indices[np.argmax(hps[indices])]
    return float(freqs[local])


def allowed_midi_notes(
    root_pc: int,
    scale: list[int],
    low: int = 0,
    high: int = 127,
) -> list[int]:
    pcs = {(root_pc + interval) % 12 for interval in scale}
    return [note for note in range(low, high + 1) if note % 12 in pcs]


def nearest_scale_note(
    midi_note: float,
    allowed: list[int],
) -> int:
    return min(allowed, key=lambda n: abs(n - midi_note))


def adjacent_scale_note(
    midi_note: int,
    allowed: list[int],
    scale_steps: int,
) -> int:
    index = min(range(len(allowed)), key=lambda i: abs(allowed[i] - midi_note))
    target_index = int(np.clip(index + scale_steps, 0, len(allowed) - 1))
    return allowed[target_index]


def midi_to_hz(midi: float) -> float:
    return float(440.0 * 2.0 ** ((midi - 69.0) / 12.0))


def oscillator(
    frequency: np.ndarray,
    sr: int,
    timbre: str,
) -> np.ndarray:
    phase = 2.0 * np.pi * np.cumsum(frequency) / sr

    if timbre == "sine":
        return np.sin(phase)

    if timbre == "reed":
        return (
            0.72 * np.sin(phase)
            + 0.19 * np.sin(2.0 * phase)
            + 0.07 * np.sin(3.0 * phase)
            + 0.02 * np.sin(4.0 * phase)
        )

    if timbre == "pluck":
        return (
            0.66 * np.sin(phase)
            + 0.20 * np.sin(2.0 * phase)
            + 0.09 * np.sin(3.0 * phase)
            + 0.05 * np.sin(4.0 * phase)
        )

    raise ValueError(f"Unknown timbre: {timbre}")


def amplitude_envelope(
    n: int,
    sr: int,
    timbre: str,
) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr

    attack = min(n, max(1, int(0.008 * sr)))
    release = min(n, max(1, int(0.045 * sr)))

    env = np.ones(n, dtype=np.float64)
    env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)

    if timbre == "pluck":
        env *= np.exp(-3.1 * t / max(n / sr, 1e-4))
    elif timbre == "reed":
        env *= np.exp(-0.35 * t / max(n / sr, 1e-4))

    if release > 1:
        env[-release:] *= np.linspace(1.0, 0.0, release)

    return env


def synth_note(
    main_midi: int,
    n: int,
    sr: int,
    timbre: str,
    rule: StyleRule,
    style_amount: float,
    use_grace: bool,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    main_hz = midi_to_hz(main_midi)
    frequency = np.full(n, main_hz, dtype=np.float64)

    if use_grace and rule.grace_fraction > 0:
        grace_n = min(
            n // 3,
            max(1, int(n * rule.grace_fraction * style_amount)),
        )
        if grace_n > 1:
            # The caller writes the grace MIDI into this temporary convention:
            # the first element is replaced afterward.
            pass

    if rule.vibrato_cents > 0 and style_amount > 0:
        t = np.arange(n, dtype=np.float64) / sr
        cents = (
            rule.vibrato_cents
            * style_amount
            * np.sin(2.0 * np.pi * rule.vibrato_hz * t)
        )
        frequency *= 2.0 ** (cents / 1200.0)

    tone = oscillator(frequency, sr, timbre)
    env = amplitude_envelope(n, sr, timbre)
    return (tone * env).astype(np.float32)


def synth_styled_note(
    main_midi: int,
    grace_midi: int | None,
    n: int,
    sr: int,
    timbre: str,
    rule: StyleRule,
    style_amount: float,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    frequency = np.full(n, midi_to_hz(main_midi), dtype=np.float64)

    if grace_midi is not None and style_amount > 0:
        grace_n = min(
            n // 3,
            max(8, int(n * rule.grace_fraction * style_amount)),
        )

        if grace_n > 8:
            frequency[:grace_n] = midi_to_hz(grace_midi)

            # Brief glide into the main note instead of a hard frequency jump.
            glide_n = min(grace_n, max(8, int(0.018 * sr)))
            start = max(0, grace_n - glide_n)
            frequency[start:grace_n] = np.geomspace(
                midi_to_hz(grace_midi),
                midi_to_hz(main_midi),
                grace_n - start,
            )

    if rule.vibrato_cents > 0 and style_amount > 0:
        t = np.arange(n, dtype=np.float64) / sr
        delay = min(n, int(0.08 * sr))
        depth = np.ones(n, dtype=np.float64)
        if delay > 1:
            depth[:delay] = np.linspace(0.0, 1.0, delay)

        cents = (
            rule.vibrato_cents
            * style_amount
            * depth
            * np.sin(2.0 * np.pi * rule.vibrato_hz * t)
        )
        frequency *= 2.0 ** (cents / 1200.0)

    tone = oscillator(frequency, sr, timbre)
    env = amplitude_envelope(n, sr, timbre)
    return (tone * env).astype(np.float32)


def render(
    y: np.ndarray,
    sr: int,
    style: str,
    root_pc: int,
    timbre: str,
    style_amount: float,
    top_db: float,
    min_note_ms: float,
    fmin: float,
    fmax: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    regions = detect_note_regions(
        y,
        top_db=top_db,
        frame_length=256,
        hop_length=64,
        min_note_ms=min_note_ms,
        sr=sr,
    )

    if len(regions) == 0:
        raise RuntimeError(
            "No isolated notes were detected. "
            "Use a monophonic melody with short gaps between notes."
        )

    allowed = allowed_midi_notes(root_pc, SCALES[style])
    rule = STYLE_RULES.get(style, DEFAULT_STYLE_RULE)
    rng = np.random.default_rng(seed)

    output = np.zeros_like(y, dtype=np.float32)
    report: list[dict[str, object]] = []

    for index, (start, end) in enumerate(regions):
        segment = y[start:end]
        f0 = estimate_fundamental_fft(
            segment,
            sr=sr,
            fmin=fmin,
            fmax=fmax,
        )

        if not np.isfinite(f0):
            continue

        input_midi = float(librosa.hz_to_midi(f0))
        target_midi = nearest_scale_note(input_midi, allowed)

        grace_midi: int | None = None
        grace_probability = rule.grace_probability * style_amount

        if (
            rule.grace_scale_steps != 0
            and rng.random() < grace_probability
        ):
            grace_midi = adjacent_scale_note(
                target_midi,
                allowed,
                rule.grace_scale_steps,
            )
            if grace_midi == target_midi:
                grace_midi = None

        note = synth_styled_note(
            main_midi=target_midi,
            grace_midi=grace_midi,
            n=end - start,
            sr=sr,
            timbre=timbre,
            rule=rule,
            style_amount=style_amount,
        )

        source_rms = float(np.sqrt(np.mean(segment * segment)))
        note_rms = float(np.sqrt(np.mean(note * note)))

        if note_rms > 1e-8:
            note *= source_rms / note_rms

        output[start:end] = note

        report.append({
            "index": index,
            "start_s": start / sr,
            "end_s": end / sr,
            "input_midi": input_midi,
            "input_note": librosa.midi_to_note(input_midi),
            "target_midi": target_midi,
            "target_note": librosa.midi_to_note(target_midi),
            "shift_semitones": target_midi - input_midi,
            "grace_note": (
                librosa.midi_to_note(grace_midi)
                if grace_midi is not None
                else ""
            ),
        })

    peak = float(np.max(np.abs(output)))
    if peak > 0.98:
        output *= 0.98 / peak

    return output, report


def write_report(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def load_audio_robust(path: Path) -> tuple[np.ndarray, int]:
    """
    Load WAV/FLAC/OGG/MP3 etc.

    1) Try soundfile first.
    2) Fall back to the FFmpeg executable bundled with imageio-ffmpeg.
    """
    try:
        audio, sr = sf.read(
            path,
            dtype="float32",
            always_2d=False,
        )
        return ensure_mono(audio), int(sr)

    except Exception as sf_error:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "Could not decode the input with soundfile.\n"
                "For MP3 support, install imageio-ffmpeg:\n"
                "  python -m pip install -U imageio-ffmpeg\n"
                f"soundfile error: {sf_error}"
            ) from exc

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)

            cmd = [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(path),
                "-vn",
                "-map", "0:a:0",
                "-c:a", "pcm_f32le",
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
                    + result.stderr.strip()
                )

            audio, sr = sf.read(
                tmp_path,
                dtype="float32",
                always_2d=False,
            )

            print(f"Decoded with bundled FFmpeg: {path.name}")
            return ensure_mono(audio), int(sr)

        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean symbolic-resynthesis scale stylizer."
    )

    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--style",
        required=True,
        choices=list(SCALES),
    )
    parser.add_argument(
        "--root",
        default="C",
    )
    parser.add_argument(
        "--timbre",
        default="pluck",
        choices=["sine", "reed", "pluck"],
    )
    parser.add_argument(
        "--style-amount",
        type=float,
        default=0.7,
        help="0 disables ornaments/vibrato; 1 gives the strongest preset.",
    )
    parser.add_argument(
        "--top-db",
        type=float,
        default=35.0,
        help="Silence split threshold.",
    )
    parser.add_argument(
        "--min-note-ms",
        type=float,
        default=55.0,
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=2000.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1479,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if not 0.0 <= args.style_amount <= 1.0:
        parser.error("--style-amount must be between 0 and 1")

    audio, sr = load_audio_robust(args.input)

    root_pc = parse_root(args.root)

    output, report = render(
        audio,
        sr=int(sr),
        style=args.style,
        root_pc=root_pc,
        timbre=args.timbre,
        style_amount=args.style_amount,
        top_db=args.top_db,
        min_note_ms=args.min_note_ms,
        fmin=args.fmin,
        fmax=args.fmax,
        seed=args.seed,
    )

    output_path = args.output or (
        args.input.parent
        / f"{args.input.stem}_{args.style}_symbolic.wav"
    )

    sf.write(
        output_path,
        output,
        int(sr),
        subtype="PCM_16",
    )

    report_path = output_path.with_suffix(".csv")
    write_report(report_path, report)

    print(f"Detected notes: {len(report)}")
    print(f"Saved audio: {output_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()