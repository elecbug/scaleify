#!/usr/bin/env python3
"""
scaleify_fullmix_v8.py

Full-mix experimental scale stylizer.

Pipeline
--------
audio file (MP3/WAV/FLAC/...)
  -> robust decode
  -> optional Demucs stem separation
  -> pYIN F0 tracking on selected stem(s)
  -> frame-wise mapping to a target scale
  -> continuous-phase symbolic resynthesis
  -> optional style ornaments/vibrato
  -> hybrid/add/replace remix with untouched stems

Why this exists
---------------
Previous versions used short independent phase-vocoder pitch shifts. That can
produce phasiness/interference while changing the musical character only a
little. v8 synthesizes a clean target-pitch layer from the detected
melodic contour.

Limitations
-----------
- This is a stylizer, not an authentic model of a musical tradition.
- 12-TET scale tables cannot reproduce microtonal tuning or full melodic grammar.
- Demucs "other" is often polyphonic; pYIN may follow the strongest pitch only.
- In "replace" mode, replacing vocals with a synthesizer removes intelligible lyrics.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from style_profiles import (StyleProfile, allowed_midi_notes,
                            load_style_profiles, map_melody_viterbi)

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]

MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
], dtype=np.float64)

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
], dtype=np.float64)


def ensure_channels_first(y: np.ndarray) -> np.ndarray:
    """Convert audio to float32 (channels, samples)."""
    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 1:
        return y[np.newaxis, :]

    if y.ndim != 2:
        raise ValueError(f"Expected mono/stereo audio, got shape={y.shape}")

    # soundfile normally returns (samples, channels), while Demucs uses
    # (channels, samples). Use the small dimension as the likely channel axis.
    if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
        return y

    if y.shape[1] <= 8 and y.shape[0] > y.shape[1]:
        return y.T

    return y


def ensure_stereo(y: np.ndarray) -> np.ndarray:
    y = ensure_channels_first(y)

    if y.shape[0] == 1:
        return np.repeat(y, 2, axis=0)

    if y.shape[0] >= 2:
        return y[:2]

    raise ValueError("Audio has no channels.")


def mono_mix(y: np.ndarray) -> np.ndarray:
    return ensure_channels_first(y).mean(axis=0).astype(np.float32)


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
    """Simple chroma/profile key estimate used only for an automatic root."""
    mono = mono_mix(y)

    # Keep key estimation reasonably cheap for long files.
    if len(mono) > sr * 180:
        mono = mono[: sr * 180]

    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    profile = np.mean(chroma, axis=1)

    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm

    maj = MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE)
    min_ = MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE)

    best_score = -np.inf
    best_root = 0
    best_mode = "major"

    for root in range(12):
        s_major = float(np.dot(profile, np.roll(maj, root)))
        s_minor = float(np.dot(profile, np.roll(min_, root)))

        if s_major > best_score:
            best_score = s_major
            best_root = root
            best_mode = "major"

        if s_minor > best_score:
            best_score = s_minor
            best_root = root
            best_mode = "minor"

    return best_root, best_mode


def decode_audio_robust(path: Path) -> tuple[np.ndarray, int]:
    """
    Decode an audio file to stereo float32 PCM.

    First try libsndfile. If that rejects an MP3, fall back to the FFmpeg
    executable bundled with imageio-ffmpeg.
    """
    sf_error: Exception | None = None

    try:
        audio, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=True,
        )
        print(f"Decoded with soundfile: {path.name}")
        return ensure_stereo(audio), int(sr)

    except Exception as exc:
        sf_error = exc
        print("[warn] soundfile decode failed; trying bundled FFmpeg.")

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "MP3/audio decoding fallback requires imageio-ffmpeg.\n"
            "Install it with:\n"
            "  python -m pip install -U imageio-ffmpeg\n"
            f"soundfile error: {sf_error}"
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        cmd = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(path),
            "-vn",
            "-map", "0:a:0",
            "-ac", "2",
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
                "FFmpeg could not decode the input.\n"
                + result.stderr.strip()
            )

        audio, sr = sf.read(
            str(tmp_path),
            dtype="float32",
            always_2d=True,
        )

        print(f"Decoded with bundled FFmpeg: {path.name}")
        return ensure_stereo(audio), int(sr)

    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def separate_with_demucs(
    decoded: np.ndarray,
    decoded_sr: int,
    model: str,
    device: str | None,
    segment: int | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Run Demucs on already-decoded PCM, avoiding its file decoder path."""
    try:
        import torch
        from demucs.api import Separator
    except ImportError as exc:
        raise RuntimeError(
            "Demucs/PyTorch is not installed. Run:\n"
            "  python -m pip install demucs torch torchaudio"
        ) from exc

    kwargs = {
        "model": model,
        "shifts": 1,
        "overlap": 0.25,
        "split": True,
        "progress": True,
    }

    if device:
        kwargs["device"] = device

    if segment is not None:
        kwargs["segment"] = segment

    separator = Separator(**kwargs)

    wav = torch.from_numpy(
        np.ascontiguousarray(ensure_stereo(decoded), dtype=np.float32)
    )

    print(
        f"Demucs input: {wav.shape[0]} ch, {decoded_sr} Hz, "
        f"model={model}, device={separator.device if hasattr(separator, 'device') else device or 'auto'}"
    )

    origin, separated = separator.separate_tensor(wav, sr=decoded_sr)

    origin_np = origin.detach().cpu().numpy().astype(np.float32)
    stems_np = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in separated.items()
    }

    return (
        ensure_stereo(origin_np),
        {name: ensure_stereo(stem) for name, stem in stems_np.items()},
        int(separator.samplerate),
    )


def nearest_allowed_vector(
    midi: np.ndarray,
    allowed: np.ndarray,
) -> np.ndarray:
    """Map every finite MIDI value to its nearest target-scale MIDI note."""
    out = np.full(midi.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(midi)

    if not np.any(valid):
        return out

    values = midi[valid][:, np.newaxis]
    nearest_indices = np.argmin(np.abs(values - allowed[np.newaxis, :]), axis=1)
    out[valid] = allowed[nearest_indices]

    return out


def median_smooth_pitch(
    midi: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """NaN-aware short median smoother."""
    if window <= 1:
        return midi.copy()

    radius = window // 2
    out = midi.copy()

    for i in range(len(midi)):
        if not np.isfinite(midi[i]):
            continue

        lo = max(0, i - radius)
        hi = min(len(midi), i + radius + 1)
        vals = midi[lo:hi]
        vals = vals[np.isfinite(vals)]

        if len(vals):
            out[i] = np.median(vals)

    return out


def fill_short_unvoiced_gaps(
    midi: np.ndarray,
    voiced_prob: np.ndarray,
    max_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill very short pitch dropouts when the surrounding target pitch agrees.
    Useful for brief pYIN misses inside sustained notes.
    """
    midi = midi.copy()
    prob = voiced_prob.copy()

    i = 0
    while i < len(midi):
        if np.isfinite(midi[i]):
            i += 1
            continue

        start = i
        while i < len(midi) and not np.isfinite(midi[i]):
            i += 1
        end = i

        gap = end - start

        if (
            gap <= max_gap_frames
            and start > 0
            and end < len(midi)
            and np.isfinite(midi[start - 1])
            and np.isfinite(midi[end])
            and abs(midi[start - 1] - midi[end]) < 0.5
        ):
            midi[start:end] = midi[start - 1]
            prob[start:end] = np.maximum(
                prob[start:end],
                min(prob[start - 1], prob[end]) * 0.7,
            )

    return midi, prob


def apply_style_ornaments(
    target_midi: np.ndarray,
    allowed: np.ndarray,
    profile: StyleProfile,
    style_amount: float,
    seed: int,
    hop_length: int,
    sr: int,
) -> np.ndarray:
    """
    Modify the first few frames of selected target-note events with an
    adjacent scale tone. This is a deliberately simple stylization rule.
    """
    result = target_midi.copy()
    rule = profile.ornament

    if style_amount <= 0 or rule.grace_scale_steps == 0:
        return result

    rng = np.random.default_rng(seed)

    events: list[tuple[int, int]] = []
    i = 0

    while i < len(result):
        if not np.isfinite(result[i]):
            i += 1
            continue

        start = i
        note = result[i]

        while (
            i < len(result)
            and np.isfinite(result[i])
            and abs(result[i] - note) < 0.5
        ):
            i += 1

        events.append((start, i))

    for start, end in events:
        if rng.random() >= rule.grace_probability * style_amount:
            continue

        main_note = int(round(result[start]))
        idx = int(np.argmin(np.abs(allowed - main_note)))
        grace_idx = int(np.clip(
            idx + rule.grace_scale_steps,
            0,
            len(allowed) - 1,
        ))
        grace_note = int(allowed[grace_idx])

        if grace_note == main_note:
            continue

        event_frames = end - start
        duration_frames = max(
            1,
            int(round(
                rule.grace_fraction
                * style_amount
                * event_frames
            )),
        )

        # Keep the ornament short on long notes.
        max_grace_frames = max(
            1,
            int(round(0.09 * sr / hop_length)),
        )
        duration_frames = min(duration_frames, max_grace_frames, event_frames)

        result[start:start + duration_frames] = grace_note

    return result


def extract_target_pitch(
    stem: np.ndarray,
    sr: int,
    root_pc: int,
    profile: StyleProfile,
    fmin_note: str,
    fmax_note: str,
    hop_length: int,
    voiced_threshold: float,
    smoothing_frames: int,
    gap_ms: float,
    style_amount: float,
    seed: int,
    pitch_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Track F0 with pYIN and map it frame-by-frame to the target scale.

    Returns
    -------
    target_midi:
        Target pitch per analysis frame; NaN means unvoiced.
    voiced_strength:
        0..1 voicing confidence per frame.
    source_midi:
        Smoothed source F0 in MIDI units for reporting.
    """
    mono = mono_mix(stem)

    fmin = float(librosa.note_to_hz(fmin_note))
    fmax = float(librosa.note_to_hz(fmax_note))

    if pitch_method == "pyin":
        f0, voiced_flag, voiced_prob = librosa.pyin(
            mono,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=2048,
            hop_length=hop_length,
            fill_na=np.nan,
        )

        voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)
        voiced = (
            np.isfinite(f0)
            & voiced_flag.astype(bool)
            & (voiced_prob >= voiced_threshold)
        )

    elif pitch_method == "yin":
        # YIN is much faster than pYIN. Since it estimates a pitch even in
        # unvoiced/noisy regions, derive a conservative voicing confidence
        # from local RMS energy and spectral flatness.
        f0 = librosa.yin(
            mono,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=2048,
            hop_length=hop_length,
        )

        rms = librosa.feature.rms(
            y=mono,
            frame_length=2048,
            hop_length=hop_length,
            center=True,
        )[0]

        flatness = librosa.feature.spectral_flatness(
            y=mono,
            n_fft=2048,
            hop_length=hop_length,
            center=True,
        )[0]

        # Match frame counts defensively across librosa versions.
        n = min(len(f0), len(rms), len(flatness))
        f0 = f0[:n]
        rms = rms[:n]
        flatness = flatness[:n]

        positive_rms = rms[rms > 1e-8]
        if len(positive_rms):
            reference_rms = float(np.percentile(positive_rms, 85))
        else:
            reference_rms = 1e-6

        energy_score = np.clip(
            rms / max(reference_rms * 0.22, 1e-8),
            0.0,
            1.0,
        )

        # Tonal frames have low spectral flatness; noise/consonants tend higher.
        tonal_score = np.clip(
            (0.35 - flatness) / 0.32,
            0.0,
            1.0,
        )

        voiced_prob = np.sqrt(energy_score * tonal_score)
        voiced = (
            np.isfinite(f0)
            & (voiced_prob >= voiced_threshold)
            & (f0 >= fmin)
            & (f0 <= fmax)
        )

    else:
        raise ValueError(f"Unknown pitch method: {pitch_method}")

    source_midi = np.full_like(f0, np.nan, dtype=np.float64)
    source_midi[voiced] = librosa.hz_to_midi(f0[voiced])
    source_midi = median_smooth_pitch(source_midi, smoothing_frames)

    allowed = allowed_midi_notes(root_pc, profile.scale)
    target_midi = map_melody_viterbi(
        source_midi=source_midi,
        root_pc=root_pc,
        profile=profile,
        style_amount=style_amount,
    )

    max_gap_frames = max(
        0,
        int(round((gap_ms / 1000.0) * sr / hop_length)),
    )
    target_midi, voiced_prob = fill_short_unvoiced_gaps(
        target_midi,
        voiced_prob,
        max_gap_frames=max_gap_frames,
    )

    target_midi = apply_style_ornaments(
        target_midi,
        allowed=allowed,
        profile=profile,
        style_amount=style_amount,
        seed=seed,
        hop_length=hop_length,
        sr=sr,
    )

    return target_midi, np.clip(voiced_prob, 0.0, 1.0), source_midi


def frame_rms(
    y: np.ndarray,
    hop_length: int,
) -> np.ndarray:
    rms = librosa.feature.rms(
        y=mono_mix(y),
        frame_length=2048,
        hop_length=hop_length,
        center=True,
    )[0]
    return rms.astype(np.float64)


def interpolate_frames(
    values: np.ndarray,
    n_samples: int,
    hop_length: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Interpolate frame data to sample rate."""
    if len(values) == 0:
        return np.full(n_samples, fill_value, dtype=np.float64)

    frame_x = librosa.frames_to_samples(
        np.arange(len(values)),
        hop_length=hop_length,
    ).astype(np.float64)

    sample_x = np.arange(n_samples, dtype=np.float64)

    return np.interp(
        sample_x,
        frame_x,
        values,
        left=values[0],
        right=values[-1],
    )


def target_midi_to_frequency(
    target_midi: np.ndarray,
    voiced_strength: np.ndarray,
    n_samples: int,
    sr: int,
    hop_length: int,
    root_pc: int,
    profile: StyleProfile,
    style_amount: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert frame-level target MIDI to sample-level frequency and voicing mask.
    Pitch transitions use a short interpolation rather than independent
    phase-vocoder segments, so oscillator phase remains continuous.
    """
    valid = np.isfinite(target_midi)

    if not np.any(valid):
        return (
            np.zeros(n_samples, dtype=np.float64),
            np.zeros(n_samples, dtype=np.float64),
        )

    # Fill unvoiced positions for interpolation only. They will be muted by mask.
    midi_filled = target_midi.copy()
    valid_idx = np.flatnonzero(valid)
    all_idx = np.arange(len(midi_filled))
    midi_filled[~valid] = np.interp(
        all_idx[~valid],
        valid_idx,
        midi_filled[valid],
    )

    midi_sample = interpolate_frames(
        midi_filled,
        n_samples=n_samples,
        hop_length=hop_length,
    )

    mask_frames = np.where(valid, voiced_strength, 0.0)
    mask = interpolate_frames(
        mask_frames,
        n_samples=n_samples,
        hop_length=hop_length,
    )

    # Sharpen a little while retaining smooth consonant/note transitions.
    mask = np.clip((mask - 0.20) / 0.65, 0.0, 1.0)

    rule = profile.ornament

    if rule.vibrato_cents > 0 and style_amount > 0:
        t = np.arange(n_samples, dtype=np.float64) / sr
        degree_mask = np.ones(n_samples, dtype=np.float64)

        if rule.vibrato_degrees:
            degrees = (np.rint(midi_sample).astype(np.int16) - root_pc) % 12
            degree_mask = np.isin(
                degrees,
                np.asarray(rule.vibrato_degrees, dtype=np.int16),
            ).astype(np.float64)

        cents = (
            rule.vibrato_cents
            * style_amount
            * np.sin(2.0 * np.pi * rule.vibrato_hz * t)
            * mask
            * degree_mask
        )
        midi_sample = midi_sample + cents / 100.0

    frequency = 440.0 * (2.0 ** ((midi_sample - 69.0) / 12.0))
    frequency *= (mask > 1e-4)

    return frequency, mask


def oscillator(
    frequency: np.ndarray,
    sr: int,
    timbre: str,
) -> np.ndarray:
    """
    Continuous-phase oscillator. No independent note segments are rendered,
    which avoids the phase-reset interference heard in the old implementation.
    """
    phase = 2.0 * np.pi * np.cumsum(frequency) / sr

    if timbre == "sine":
        tone = np.sin(phase)

    elif timbre == "reed":
        tone = (
            0.72 * np.sin(phase)
            + 0.19 * np.sin(2.0 * phase)
            + 0.07 * np.sin(3.0 * phase)
            + 0.02 * np.sin(4.0 * phase)
        )

    elif timbre == "pluck":
        tone = (
            0.66 * np.sin(phase)
            + 0.20 * np.sin(2.0 * phase)
            + 0.09 * np.sin(3.0 * phase)
            + 0.05 * np.sin(4.0 * phase)
        )

    elif timbre == "flute":
        tone = (
            0.90 * np.sin(phase)
            + 0.07 * np.sin(2.0 * phase)
            + 0.03 * np.sin(3.0 * phase)
        )

    else:
        raise ValueError(f"Unknown timbre: {timbre}")

    return tone.astype(np.float64)


def synthesize_scale_layer(
    source_stem: np.ndarray,
    sr: int,
    target_midi: np.ndarray,
    voiced_strength: np.ndarray,
    root_pc: int,
    profile: StyleProfile,
    style_amount: float,
    hop_length: int,
    timbre: str,
    synth_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a stereo scale-constrained melody layer whose loudness follows
    the selected source stem.
    """
    source_stem = ensure_stereo(source_stem)
    n_samples = source_stem.shape[-1]

    frequency, mask = target_midi_to_frequency(
        target_midi,
        voiced_strength,
        n_samples=n_samples,
        sr=sr,
        hop_length=hop_length,
        root_pc=root_pc,
        profile=profile,
        style_amount=style_amount,
    )

    tone = oscillator(frequency, sr=sr, timbre=timbre)

    rms_frames = frame_rms(source_stem, hop_length=hop_length)
    amplitude = interpolate_frames(
        rms_frames,
        n_samples=n_samples,
        hop_length=hop_length,
    )

    # Convert local RMS to oscillator peak scale. Cap extreme local boosts.
    amplitude = np.minimum(amplitude * 2.2, 0.85)
    mono_layer = tone * amplitude * mask * synth_gain

    # Mild edge smoothing through mask already handles note/consonant boundaries.
    stereo = np.repeat(mono_layer[np.newaxis, :], 2, axis=0)

    return stereo.astype(np.float32), mask.astype(np.float32)


def remix_target_stem(
    original_stem: np.ndarray,
    synth_layer: np.ndarray,
    voiced_mask: np.ndarray,
    mix_mode: str,
    vocal_preserve: float,
) -> np.ndarray:
    """
    add:
        Original stem + synthetic layer.
    replace:
        Synthetic layer only.
    hybrid:
        Keep unvoiced/consonant regions; attenuate original pitched material
        while the target-scale synth carries the pitch.
    """
    original_stem = ensure_stereo(original_stem)
    synth_layer = ensure_stereo(synth_layer)

    mask = voiced_mask[np.newaxis, :]

    if mix_mode == "add":
        return original_stem + synth_layer

    if mix_mode == "replace":
        return synth_layer

    if mix_mode == "hybrid":
        retain = 1.0 - mask * (1.0 - vocal_preserve)
        return original_stem * retain + synth_layer

    raise ValueError(f"Unknown mix mode: {mix_mode}")


def peak_protect(
    y: np.ndarray,
    target_peak: float = 0.98,
) -> np.ndarray:
    y = ensure_stereo(y).astype(np.float32)
    peak = float(np.max(np.abs(y)))

    if peak > target_peak and peak > 0:
        y = y * (target_peak / peak)

    return y


def write_pitch_report(
    path: Path,
    target_midi: np.ndarray,
    source_midi: np.ndarray,
    voiced_strength: np.ndarray,
    hop_length: int,
    sr: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "time_s",
            "source_midi",
            "source_note",
            "target_midi",
            "target_note",
            "voiced_probability",
        ])

        for i in range(len(target_midi)):
            src = source_midi[i]
            tgt = target_midi[i]

            writer.writerow([
                i,
                i * hop_length / sr,
                "" if not np.isfinite(src) else float(src),
                "" if not np.isfinite(src) else str(librosa.midi_to_note(src)),
                "" if not np.isfinite(tgt) else float(tgt),
                "" if not np.isfinite(tgt) else str(librosa.midi_to_note(tgt)),
                float(voiced_strength[i]),
            ])


def transform_stem(
    stem: np.ndarray,
    sr: int,
    root_pc: int,
    profile: StyleProfile,
    stem_name: str,
    hop_length: int,
    voiced_threshold: float,
    smoothing_frames: int,
    gap_ms: float,
    style_amount: float,
    seed: int,
    timbre: str,
    synth_gain: float,
    mix_mode: str,
    vocal_preserve: float,
    pitch_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ranges = {
        "vocals": ("C2", "C7"),
        "other": ("C2", "C7"),
        "bass": ("E1", "C5"),
        "mix": ("C2", "C7"),
    }
    fmin_note, fmax_note = ranges.get(stem_name, ("C2", "C7"))

    target_midi, voiced_strength, source_midi = extract_target_pitch(
        stem=stem,
        sr=sr,
        root_pc=root_pc,
        profile=profile,
        fmin_note=fmin_note,
        fmax_note=fmax_note,
        hop_length=hop_length,
        voiced_threshold=voiced_threshold,
        smoothing_frames=smoothing_frames,
        gap_ms=gap_ms,
        style_amount=style_amount,
        seed=seed,
        pitch_method=pitch_method,
    )

    synth_layer, mask = synthesize_scale_layer(
        source_stem=stem,
        sr=sr,
        target_midi=target_midi,
        voiced_strength=voiced_strength,
        root_pc=root_pc,
        profile=profile,
        style_amount=style_amount,
        hop_length=hop_length,
        timbre=timbre,
        synth_gain=synth_gain,
    )

    processed = remix_target_stem(
        original_stem=stem,
        synth_layer=synth_layer,
        voiced_mask=mask,
        mix_mode=mix_mode,
        vocal_preserve=vocal_preserve,
    )

    return processed, target_midi, source_midi, voiced_strength


def parse_stems(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demucs + YIN/pYIN + external-profile Viterbi melodic stylizer."
    )

    parser.add_argument("input", type=Path, nargs="?")

    parser.add_argument(
        "--style",
        default=None,
        help="External style profile id.",
    )

    parser.add_argument(
        "--style-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "styles",
        help="Directory containing JSON style profiles.",
    )

    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List available style profiles and exit.",
    )

    parser.add_argument(
        "--root",
        default="auto",
        help="auto, C, C#, D, Eb, ...",
    )

    parser.add_argument(
        "--target-stems",
        default="vocals",
        help=(
            "Comma-separated stems to stylize. Default: vocals. "
            "Try 'other' for instrumental tracks, but it may be polyphonic."
        ),
    )

    parser.add_argument(
        "--mix-mode",
        choices=["hybrid", "add", "replace"],
        default="hybrid",
        help=(
            "hybrid attenuates original pitched material but keeps consonants; "
            "add overlays synth; replace removes the selected original stem."
        ),
    )

    parser.add_argument(
        "--vocal-preserve",
        type=float,
        default=0.20,
        help="Original pitched-stem amount retained in hybrid mode (0..1).",
    )

    parser.add_argument(
        "--synth-gain",
        type=float,
        default=1.15,
    )

    parser.add_argument(
        "--timbre",
        choices=["sine", "flute", "reed", "pluck"],
        default="flute",
    )

    parser.add_argument(
        "--style-amount",
        type=float,
        default=0.75,
        help="Melodic-grammar + ornament strength, 0..1.",
    )

    parser.add_argument(
        "--pitch-method",
        choices=["yin", "pyin"],
        default="yin",
        help=(
            "yin is substantially faster; pyin gives more robust voicing "
            "decisions but is slower."
        ),
    )

    parser.add_argument(
        "--voiced-threshold",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--smoothing-frames",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--gap-ms",
        type=float,
        default=45.0,
        help="Fill short pYIN dropouts up to this duration.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1479,
    )

    parser.add_argument(
        "--demucs-model",
        default="htdemucs",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Demucs device such as cuda or cpu. Default: auto.",
    )

    parser.add_argument(
        "--demucs-segment",
        type=int,
        default=None,
        help="Optional Demucs segment length in seconds for lower VRAM usage.",
    )

    parser.add_argument(
        "--no-demucs",
        action="store_true",
        help="Treat the whole file as one melodic stem (for clean test melodies).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
    )

    args = parser.parse_args()

    profiles = load_style_profiles(args.style_dir)

    if args.list_styles:
        print(f"Style directory: {args.style_dir}")
        for style_id, item in sorted(profiles.items()):
            print(f"  {style_id:20s}  {item.label} [{item.region}]")
        return

    if args.input is None:
        parser.error("input is required unless --list-styles is used")

    if args.style is None:
        parser.error("--style is required")

    if args.style not in profiles:
        available = ", ".join(sorted(profiles))
        parser.error(f"unknown style '{args.style}'. Available: {available}")

    profile = profiles[args.style]
    print(
        f"Style profile: {profile.id} | {profile.label} | "
        f"region={profile.region} | scale={list(profile.scale)}"
    )

    if not 0.0 <= args.style_amount <= 1.0:
        parser.error("--style-amount must be between 0 and 1")

    if not 0.0 <= args.vocal_preserve <= 1.0:
        parser.error("--vocal-preserve must be between 0 and 1")

    decoded, decoded_sr = decode_audio_robust(args.input)

    if args.no_demucs:
        original = decoded
        stems = {"mix": decoded.copy()}
        sr = decoded_sr
        target_stems = ["mix"]
        print("Demucs disabled; treating full input as one melodic stem.")

    else:
        print("Separating stems with Demucs...")
        original, stems, sr = separate_with_demucs(
            decoded=decoded,
            decoded_sr=decoded_sr,
            model=args.demucs_model,
            device=args.device,
            segment=args.demucs_segment,
        )
        target_stems = parse_stems(args.target_stems)
        print(f"Available stems: {list(stems)}")

    root_pc = parse_root(args.root)

    if root_pc is None:
        root_pc, mode = detect_key(original, sr)
        print(f"Detected key/root: {NOTE_NAMES[root_pc]} {mode}")
    else:
        print(f"Root: {NOTE_NAMES[root_pc]}")

    processed_stems = {
        name: ensure_stereo(stem).copy()
        for name, stem in stems.items()
    }

    for stem_name in target_stems:
        if stem_name not in processed_stems:
            print(
                f"[warn] target stem '{stem_name}' is unavailable; "
                f"available={list(processed_stems)}"
            )
            continue

        if stem_name == "drums":
            print("[warn] skipping drums; pYIN pitch conversion is inappropriate.")
            continue

        print(
            f"Stylizing stem={stem_name}, style={profile.id}, "
            f"mode={args.mix_mode}, pitch={args.pitch_method} ..."
        )

        processed, target_midi, source_midi, voiced_strength = transform_stem(
            stem=processed_stems[stem_name],
            sr=sr,
            root_pc=root_pc,
            profile=profile,
            stem_name=stem_name,
            hop_length=args.hop_length,
            voiced_threshold=args.voiced_threshold,
            smoothing_frames=args.smoothing_frames,
            gap_ms=args.gap_ms,
            style_amount=args.style_amount,
            seed=args.seed,
            timbre=args.timbre,
            synth_gain=args.synth_gain,
            mix_mode=args.mix_mode,
            vocal_preserve=args.vocal_preserve,
            pitch_method=args.pitch_method,
        )

        processed_stems[stem_name] = processed

        report_path = (
            args.report_dir
            / f"{args.input.stem}_{stem_name}_{profile.id}.csv"
        )
        write_pitch_report(
            report_path,
            target_midi=target_midi,
            source_midi=source_midi,
            voiced_strength=voiced_strength,
            hop_length=args.hop_length,
            sr=sr,
        )
        print(f"Pitch report: {report_path}")

    # Sum all Demucs stems. In --no-demucs mode there is only "mix".
    n = min(stem.shape[-1] for stem in processed_stems.values())
    remixed = np.zeros((2, n), dtype=np.float32)

    for stem in processed_stems.values():
        remixed += ensure_stereo(stem)[:, :n]

    remixed = peak_protect(remixed)

    output_path = args.output or (
        args.input.parent
        / f"{args.input.stem}_{profile.id}_v8.wav"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(output_path),
        remixed.T,
        sr,
        subtype="PCM_24",
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()