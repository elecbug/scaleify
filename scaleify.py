#!/usr/bin/env python3
"""
scaleify v9 - monophonic melodic style transfer

Pipeline
--------
MP3/WAV/FLAC/... -> robust decode -> mono melody F0 -> note events -> phrase
segmentation -> higher-order Viterbi melodic grammar -> rhythm rewrite ->
degree-conditioned ornaments + optional microtuning -> resynthesis -> reports.

v9 is intentionally focused on isolated or predominantly monophonic melody input.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from style_profiles import (
    MappingResult,
    NoteEvent,
    StyleProfile,
    allowed_midi_notes,
    degree_of_midi,
    load_style_profiles,
    map_melody_viterbi,
)


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


@dataclass(frozen=True)
class RenderEvent:
    phrase_index: int
    event_index: int
    source: NoteEvent
    source_duration_s: float
    output_duration_s: float
    gap_after_s: float
    target_midi: int
    root_pc: int
    scale: tuple[int, ...]
    modulation_name: str


@dataclass(frozen=True)
class RenderResult:
    audio: np.ndarray
    events: tuple[RenderEvent, ...]
    output_duration_s: float


def ensure_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.ndim != 2:
        raise ValueError(f"Expected audio with one or two dimensions, got {y.shape}")
    # soundfile: samples x channels
    if y.shape[1] <= 8:
        return np.mean(y, axis=1).astype(np.float32)
    # channels x samples
    if y.shape[0] <= 8:
        return np.mean(y, axis=0).astype(np.float32)
    raise ValueError(f"Cannot determine channel axis for shape={y.shape}")


def decode_audio_robust(path: Path) -> tuple[np.ndarray, int]:
    """Decode with soundfile first, then bundled imageio-ffmpeg."""
    sf_error: Exception | None = None
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        print(f"Decoded with soundfile: {path.name}")
        return ensure_mono(audio), int(sr)
    except Exception as exc:
        sf_error = exc
        print("[warn] soundfile decode failed; trying bundled FFmpeg.")

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Audio decoding fallback requires imageio-ffmpeg.\n"
            "Install it with:\n"
            "  python -m pip install -U imageio-ffmpeg\n"
            f"soundfile error: {sf_error}"
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        result = subprocess.run(
            [
                str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(path), "-vn", "-map", "0:a:0",
                "-ac", "1", "-c:a", "pcm_f32le", str(tmp_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("FFmpeg could not decode input:\n" + result.stderr.strip())

        audio, sr = sf.read(str(tmp_path), dtype="float32", always_2d=False)
        print(f"Decoded with bundled FFmpeg: {path.name}")
        return ensure_mono(audio), int(sr)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def parse_root(root: str) -> int | None:
    if root.lower() == "auto":
        return None
    name = root.strip().upper()
    aliases = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    name = aliases.get(name, name)
    if name not in NOTE_NAMES:
        raise ValueError(f"Unknown root note: {root}")
    return NOTE_NAMES.index(name)


def detect_key(y: np.ndarray, sr: int) -> tuple[int, str]:
    mono = y[: sr * 180] if len(y) > sr * 180 else y
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    profile = np.mean(chroma, axis=1)
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile /= norm

    maj = MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE)
    min_ = MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE)
    best_score = -np.inf
    best_root = 0
    best_mode = "major"
    for root in range(12):
        sm = float(np.dot(profile, np.roll(maj, root)))
        sn = float(np.dot(profile, np.roll(min_, root)))
        if sm > best_score:
            best_score, best_root, best_mode = sm, root, "major"
        if sn > best_score:
            best_score, best_root, best_mode = sn, root, "minor"
    return best_root, best_mode


def median_smooth_pitch(midi: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return midi.copy()
    radius = window // 2
    out = midi.copy()
    for i in range(len(midi)):
        if not np.isfinite(midi[i]):
            continue
        vals = midi[max(0, i - radius):min(len(midi), i + radius + 1)]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[i] = np.median(vals)
    return out


def fill_short_gaps(
    midi: np.ndarray,
    confidence: np.ndarray,
    max_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    midi = midi.copy()
    confidence = confidence.copy()
    i = 0
    while i < len(midi):
        if np.isfinite(midi[i]):
            i += 1
            continue
        start = i
        while i < len(midi) and not np.isfinite(midi[i]):
            i += 1
        end = i
        if (
            end - start <= max_gap_frames
            and start > 0 and end < len(midi)
            and np.isfinite(midi[start - 1]) and np.isfinite(midi[end])
            and abs(midi[start - 1] - midi[end]) < 0.6
        ):
            midi[start:end] = 0.5 * (midi[start - 1] + midi[end])
            confidence[start:end] = min(confidence[start - 1], confidence[end]) * 0.7
    return midi, confidence


def extract_source_pitch(
    y: np.ndarray,
    sr: int,
    pitch_method: str,
    fmin_note: str,
    fmax_note: str,
    hop_length: int,
    voiced_threshold: float,
    smoothing_frames: int,
    gap_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    fmin = float(librosa.note_to_hz(fmin_note))
    fmax = float(librosa.note_to_hz(fmax_note))

    if pitch_method == "pyin":
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=2048,
            hop_length=hop_length,
            fill_na=np.nan,
        )
        confidence = np.nan_to_num(voiced_prob, nan=0.0)
        voiced = np.isfinite(f0) & voiced_flag.astype(bool) & (confidence >= voiced_threshold)
    elif pitch_method == "yin":
        f0 = librosa.yin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=2048,
            hop_length=hop_length,
        )
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        flat = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=hop_length)[0]
        n = min(len(f0), len(rms), len(flat))
        f0, rms, flat = f0[:n], rms[:n], flat[:n]
        positive = rms[rms > 1e-8]
        reference = float(np.percentile(positive, 85)) if len(positive) else 1e-6
        energy = np.clip(rms / max(reference * 0.22, 1e-8), 0.0, 1.0)
        tonal = np.clip((0.35 - flat) / 0.32, 0.0, 1.0)
        confidence = np.sqrt(energy * tonal)
        voiced = np.isfinite(f0) & (confidence >= voiced_threshold) & (f0 >= fmin) & (f0 <= fmax)
    else:
        raise ValueError(f"Unknown pitch method: {pitch_method}")

    midi = np.full_like(f0, np.nan, dtype=np.float64)
    midi[voiced] = librosa.hz_to_midi(f0[voiced])
    midi = median_smooth_pitch(midi, smoothing_frames)
    max_gap_frames = max(0, int(round(gap_ms / 1000.0 * sr / hop_length)))
    midi, confidence = fill_short_gaps(midi, confidence, max_gap_frames)
    return midi, np.clip(confidence, 0.0, 1.0)


def nearest_scale_note(note: int, allowed: np.ndarray, steps: int) -> int:
    idx = int(np.argmin(np.abs(allowed.astype(np.float64) - note)))
    idx = int(np.clip(idx + steps, 0, len(allowed) - 1))
    return int(allowed[idx])


def degree_tuning_cents(note: int, root_pc: int, profile: StyleProfile, enabled: bool) -> float:
    if not enabled:
        return 0.0
    degree = degree_of_midi(note, root_pc)
    return float(profile.tuning.degree_cents.get(degree, 0.0))


def event_rms(y: np.ndarray, event: NoteEvent, hop_length: int) -> float:
    start = max(0, event.start_frame * hop_length)
    end = min(len(y), max(start + 1, event.end_frame * hop_length))
    segment = y[start:end]
    if len(segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))


def transform_phrase_rhythm(
    phrase_events: tuple[NoteEvent, ...],
    targets: tuple[int, ...],
    root_pc: int,
    profile: StyleProfile,
    sr: int,
    hop_length: int,
    amount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source durations, transformed durations, and transformed gaps."""
    source_durations = np.asarray([e.frames * hop_length / sr for e in phrase_events], dtype=np.float64)
    source_gaps = np.asarray([
        max(0, phrase_events[i + 1].start_frame - e.end_frame) * hop_length / sr
        if i + 1 < len(phrase_events) else 0.0
        for i, e in enumerate(phrase_events)
    ], dtype=np.float64)

    rhythm = profile.rhythm
    if not rhythm.enabled or amount <= 0:
        return source_durations, source_durations.copy(), source_gaps.copy()

    durations = source_durations.copy()
    gaps = source_gaps.copy()
    base = max(1e-4, float(np.median(source_durations)))

    q = np.clip(rhythm.quantize_strength * amount, 0.0, 1.0)
    ratios = np.asarray(rhythm.preferred_duration_ratios, dtype=np.float64)
    if len(ratios):
        for i, duration in enumerate(durations):
            ratio = duration / base
            nearest = float(ratios[np.argmin(np.abs(ratios - ratio))])
            desired = base * ((1.0 - q) * ratio + q * nearest)
            durations[i] = desired

    max_change = max(0.0, rhythm.max_duration_change * amount)
    for i, note in enumerate(targets):
        degree = degree_of_midi(note, root_pc)
        multiplier = rhythm.degree_duration_multipliers.get(degree, 1.0)
        multiplier = 1.0 + amount * (multiplier - 1.0)
        durations[i] *= multiplier

        lo = source_durations[i] * max(0.1, 1.0 - max_change)
        hi = source_durations[i] * (1.0 + max_change)
        durations[i] = np.clip(durations[i], lo, hi)

    if len(durations):
        end_mult = 1.0 + amount * (rhythm.phrase_end_multiplier - 1.0)
        durations[-1] *= end_mult

    gaps *= 1.0 + amount * (rhythm.gap_multiplier - 1.0)

    if rhythm.preserve_phrase_duration:
        src_total = float(np.sum(source_durations) + np.sum(source_gaps))
        out_total = float(np.sum(durations) + np.sum(gaps))
        if out_total > 1e-8:
            scale = src_total / out_total
            durations *= scale
            gaps *= scale

    return source_durations, durations, gaps


def synth_event(
    target_midi: int,
    duration_s: float,
    amplitude: float,
    root_pc: int,
    scale: tuple[int, ...],
    profile: StyleProfile,
    sr: int,
    timbre: str,
    style_amount: float,
    enable_ornaments: bool,
    enable_microtuning: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    n = max(1, int(round(duration_s * sr)))
    degree = degree_of_midi(target_midi, root_pc)
    midi_curve = np.full(n, float(target_midi), dtype=np.float64)

    allowed = allowed_midi_notes(root_pc, scale)
    tuning_curve = np.full(
        n,
        degree_tuning_cents(target_midi, root_pc, profile, enable_microtuning),
        dtype=np.float64,
    )

    if enable_ornaments and style_amount > 0:
        for rule in profile.ornaments:
            if rule.degrees and degree not in rule.degrees:
                continue
            if rng.random() >= np.clip(rule.probability * style_amount, 0.0, 1.0):
                continue

            span = min(
                n,
                max(1, int(round(duration_s * rule.fraction * style_amount * sr))),
                max(1, int(round(rule.max_ms / 1000.0 * sr))),
            )

            if rule.type == "grace" and rule.scale_steps != 0 and span > 2:
                grace = nearest_scale_note(target_midi, allowed, rule.scale_steps)
                midi_curve[:span] = grace
                tuning_curve[:span] = degree_tuning_cents(grace, root_pc, profile, enable_microtuning)
                glide = min(span, max(2, int(0.015 * sr)))
                midi_curve[span - glide:span] = np.linspace(grace, target_midi, glide)
                tuning_curve[span - glide:span] = np.linspace(
                    degree_tuning_cents(grace, root_pc, profile, enable_microtuning),
                    degree_tuning_cents(target_midi, root_pc, profile, enable_microtuning),
                    glide,
                )

            elif rule.type == "slide_in" and rule.scale_steps != 0 and span > 2:
                start_note = nearest_scale_note(target_midi, allowed, rule.scale_steps)
                midi_curve[:span] = np.linspace(start_note, target_midi, span)
                tuning_curve[:span] = np.linspace(
                    degree_tuning_cents(start_note, root_pc, profile, enable_microtuning),
                    degree_tuning_cents(target_midi, root_pc, profile, enable_microtuning),
                    span,
                )

            elif rule.type == "slide_out" and rule.scale_steps != 0 and span > 2:
                end_note = nearest_scale_note(target_midi, allowed, rule.scale_steps)
                midi_curve[-span:] = np.linspace(target_midi, end_note, span)
                tuning_curve[-span:] = np.linspace(
                    degree_tuning_cents(target_midi, root_pc, profile, enable_microtuning),
                    degree_tuning_cents(end_note, root_pc, profile, enable_microtuning),
                    span,
                )

            elif rule.type == "vibrato" and rule.depth_cents > 0:
                start = min(n - 1, int(round(n * rule.delay_fraction)))
                t = np.arange(n - start, dtype=np.float64) / sr
                tuning_curve[start:] += (
                    rule.depth_cents
                    * style_amount
                    * np.sin(2.0 * np.pi * rule.rate_hz * t)
                )

    freq = 440.0 * 2.0 ** (((midi_curve + tuning_curve / 100.0) - 69.0) / 12.0)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr

    if timbre == "sine":
        tone = np.sin(phase)
    elif timbre == "flute":
        tone = 0.90 * np.sin(phase) + 0.07 * np.sin(2 * phase) + 0.03 * np.sin(3 * phase)
    elif timbre == "reed":
        tone = 0.72 * np.sin(phase) + 0.19 * np.sin(2 * phase) + 0.07 * np.sin(3 * phase) + 0.02 * np.sin(4 * phase)
    elif timbre == "pluck":
        tone = 0.66 * np.sin(phase) + 0.20 * np.sin(2 * phase) + 0.09 * np.sin(3 * phase) + 0.05 * np.sin(4 * phase)
    else:
        raise ValueError(f"Unknown timbre: {timbre}")

    attack = min(n, max(1, int(0.008 * sr)))
    release = min(n, max(1, int(0.045 * sr)))
    env = np.ones(n, dtype=np.float64)
    if attack > 1:
        env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
    if timbre == "pluck":
        t = np.arange(n, dtype=np.float64) / sr
        env *= np.exp(-2.5 * t / max(duration_s, 1e-4))
    if release > 1:
        env[-release:] *= np.linspace(1.0, 0.0, release)

    return (tone * env * amplitude).astype(np.float32)


def render_mapping(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    mapping: MappingResult,
    profile: StyleProfile,
    timbre: str,
    style_amount: float,
    rhythm_amount: float,
    enable_ornaments: bool,
    enable_microtuning: bool,
    seed: int,
) -> RenderResult:
    if not mapping.phrases:
        raise RuntimeError("No note phrases were detected in the input.")

    all_events = [e for phrase in mapping.phrases for e in phrase.events]
    rms_values = np.asarray([event_rms(y, e, hop_length) for e in all_events], dtype=np.float64)
    positive = rms_values[rms_values > 1e-8]
    reference_rms = float(np.percentile(positive, 85)) if len(positive) else 0.1

    parts: list[np.ndarray] = []
    render_events: list[RenderEvent] = []
    rng = np.random.default_rng(seed)

    first_start_s = mapping.phrases[0].start_frame * hop_length / sr
    if first_start_s > 0:
        parts.append(np.zeros(int(round(first_start_s * sr)), dtype=np.float32))

    flat_rms_index = 0

    for pidx, (phrase, targets, root_pc, scale, modulation_name) in enumerate(zip(
        mapping.phrases,
        mapping.targets,
        mapping.roots,
        mapping.scales,
        mapping.modulation_names,
    )):
        src_dur, out_dur, gaps = transform_phrase_rhythm(
            phrase.events,
            targets,
            root_pc,
            profile,
            sr,
            hop_length,
            rhythm_amount,
        )

        for eidx, (event, target) in enumerate(zip(phrase.events, targets)):
            local_rms = rms_values[flat_rms_index]
            flat_rms_index += 1
            amplitude = np.clip(0.12 + 0.62 * (local_rms / max(reference_rms, 1e-8)), 0.12, 0.85)

            note_audio = synth_event(
                target_midi=target,
                duration_s=float(out_dur[eidx]),
                amplitude=float(amplitude),
                root_pc=root_pc,
                scale=scale,
                profile=profile,
                sr=sr,
                timbre=timbre,
                style_amount=style_amount,
                enable_ornaments=enable_ornaments,
                enable_microtuning=enable_microtuning,
                rng=rng,
            )
            parts.append(note_audio)

            gap_after = float(gaps[eidx])
            if eidx == len(phrase.events) - 1 and pidx + 1 < len(mapping.phrases):
                next_phrase = mapping.phrases[pidx + 1]
                source_phrase_gap = max(0, next_phrase.start_frame - event.end_frame) * hop_length / sr
                gap_after += source_phrase_gap

            if gap_after > 0:
                parts.append(np.zeros(int(round(gap_after * sr)), dtype=np.float32))

            render_events.append(RenderEvent(
                phrase_index=pidx,
                event_index=eidx,
                source=event,
                source_duration_s=float(src_dur[eidx]),
                output_duration_s=float(out_dur[eidx]),
                gap_after_s=gap_after,
                target_midi=int(target),
                root_pc=int(root_pc),
                scale=tuple(scale),
                modulation_name=modulation_name,
            ))

    # Preserve trailing silence after the last detected phrase.
    last_end_s = mapping.phrases[-1].end_frame * hop_length / sr
    trailing_s = max(0.0, len(y) / sr - last_end_s)
    if trailing_s > 0:
        parts.append(np.zeros(int(round(trailing_s * sr)), dtype=np.float32))

    audio = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio *= 0.98 / peak

    return RenderResult(audio=audio, events=tuple(render_events), output_duration_s=len(audio) / sr)


def transition_reward(
    prev_source: float,
    source: float,
    prev_target: int,
    target: int,
    root_pc: int,
    profile: StyleProfile,
) -> float:
    g = profile.grammar
    pd = degree_of_midi(prev_target, root_pc)
    d = degree_of_midi(target, root_pc)
    key = f"{pd}>{d}"
    reward = g.transition_weights.get(key, 0.0)
    delta = source - prev_source
    if delta > 0.35:
        reward += g.ascending_transition_weights.get(key, 0.0)
    elif delta < -0.35:
        reward += g.descending_transition_weights.get(key, 0.0)
    return reward


def compute_metrics(
    mapping: MappingResult,
    render: RenderResult,
    profile: StyleProfile,
) -> dict:
    source_notes: list[float] = []
    target_notes: list[int] = []
    target_roots: list[int] = []

    contour_match = 0
    contour_total = 0
    interval_rewards: list[float] = []
    transition_rewards: list[float] = []
    trigram_rewards: list[float] = []
    cadence_rewards: list[float] = []
    phrase_hits = 0
    transition_count = 0

    g = profile.grammar
    max_interval_reward = max([0.0, *g.interval_weights.values()])
    max_transition_reward = max([
        0.0,
        *g.transition_weights.values(),
        *g.ascending_transition_weights.values(),
        *g.descending_transition_weights.values(),
    ])
    max_trigram_reward = max([0.0, *g.trigram_weights.values(), *(w for _, w in g.preferred_phrases)])
    max_cadence_reward = max([0.0, *g.cadence_degrees.values(), *(w for _, w in g.cadence_patterns)])

    for phrase, targets, root in zip(mapping.phrases, mapping.targets, mapping.roots):
        src = [e.source_midi for e in phrase.events]
        tgt = list(targets)
        source_notes.extend(src)
        target_notes.extend(tgt)
        target_roots.extend([root] * len(tgt))

        degrees = [degree_of_midi(n, root) for n in tgt]

        for i in range(1, len(tgt)):
            in_delta = src[i] - src[i - 1]
            out_delta = tgt[i] - tgt[i - 1]
            if abs(in_delta) >= 0.5 and abs(out_delta) >= 0.5:
                contour_total += 1
                if np.sign(in_delta) == np.sign(out_delta):
                    contour_match += 1

            interval_rewards.append(g.interval_weights.get(min(12, int(round(abs(out_delta)))), 0.0))
            transition_rewards.append(transition_reward(src[i - 1], src[i], tgt[i - 1], tgt[i], root, profile))
            transition_count += 1

        for i in range(2, len(degrees)):
            key = ">".join(str(x) for x in degrees[i - 2:i + 1])
            trigram_rewards.append(g.trigram_weights.get(key, 0.0))

        for pattern, weight in g.preferred_phrases:
            for i in range(len(degrees) - len(pattern) + 1):
                if tuple(degrees[i:i + len(pattern)]) == pattern:
                    phrase_hits += 1
                    trigram_rewards.append(weight)

        cadence = g.cadence_degrees.get(degrees[-1], 0.0) if degrees else 0.0
        for pattern, weight in g.cadence_patterns:
            if len(degrees) >= len(pattern) and tuple(degrees[-len(pattern):]) == pattern:
                cadence += weight
        cadence_rewards.append(cadence)

    source_arr = np.asarray(source_notes, dtype=np.float64)
    target_arr = np.asarray(target_notes, dtype=np.float64)
    displacement = np.abs(target_arr - source_arr) if len(source_arr) else np.asarray([])

    def normalized_mean(values: list[float], maximum: float) -> float:
        if not values or maximum <= 0:
            return 0.0
        return float(np.clip(np.mean(values) / maximum, 0.0, 1.0))

    source_durations = np.asarray([e.source_duration_s for e in render.events], dtype=np.float64)
    output_durations = np.asarray([e.output_duration_s for e in render.events], dtype=np.float64)
    duration_change = np.abs(output_durations - source_durations) / np.maximum(source_durations, 1e-6)

    scale_compliance = 1.0
    for target, root, scale in zip(target_notes, target_roots, [e.scale for e in render.events]):
        if degree_of_midi(target, root) not in set(scale):
            scale_compliance = 0.0
            break

    interval_score = normalized_mean(interval_rewards, max_interval_reward)
    transition_score = normalized_mean(transition_rewards, max_transition_reward)
    trigram_score = normalized_mean(trigram_rewards, max_trigram_reward)
    cadence_score = normalized_mean(cadence_rewards, max_cadence_reward)
    style_grammar_score = float(np.mean([interval_score, transition_score, trigram_score, cadence_score]))

    mean_disp = float(np.mean(displacement)) if len(displacement) else 0.0
    melody_preservation = float(np.exp(-mean_disp / 3.0))

    return {
        "style": profile.id,
        "events": len(target_notes),
        "phrases": len(mapping.phrases),
        "modulated_phrases": int(sum(name != "base" for name in mapping.modulation_names)),
        "mean_pitch_displacement_semitones": mean_disp,
        "median_pitch_displacement_semitones": float(np.median(displacement)) if len(displacement) else 0.0,
        "max_pitch_displacement_semitones": float(np.max(displacement)) if len(displacement) else 0.0,
        "melody_preservation_score": melody_preservation,
        "contour_preservation_score": float(contour_match / contour_total) if contour_total else 1.0,
        "scale_compliance_score": scale_compliance,
        "interval_style_score": interval_score,
        "transition_style_score": transition_score,
        "trigram_phrase_style_score": trigram_score,
        "cadence_style_score": cadence_score,
        "style_grammar_score": style_grammar_score,
        "preferred_phrase_hits": phrase_hits,
        "mean_relative_duration_change": float(np.mean(duration_change)) if len(duration_change) else 0.0,
        "output_duration_s": render.output_duration_s,
    }


def write_event_report(path: Path, render: RenderResult, profile: StyleProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phrase", "event", "source_start_frame", "source_end_frame",
            "source_midi", "source_note", "target_midi", "target_note",
            "root", "degree", "modulation", "source_duration_s",
            "output_duration_s", "gap_after_s", "tuning_cents",
        ])
        for e in render.events:
            degree = degree_of_midi(e.target_midi, e.root_pc)
            writer.writerow([
                e.phrase_index,
                e.event_index,
                e.source.start_frame,
                e.source.end_frame,
                e.source.source_midi,
                str(librosa.midi_to_note(e.source.source_midi)),
                e.target_midi,
                str(librosa.midi_to_note(e.target_midi)),
                NOTE_NAMES[e.root_pc],
                degree,
                e.modulation_name,
                e.source_duration_s,
                e.output_duration_s,
                e.gap_after_s,
                profile.tuning.degree_cents.get(degree, 0.0),
            ])


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Monophonic phrase-aware melodic style transfer."
    )
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--style", default=None)
    parser.add_argument("--style-dir", type=Path, default=script_dir / "styles")
    parser.add_argument("--list-styles", action="store_true")
    parser.add_argument("--root", default="auto")
    parser.add_argument("--style-amount", type=float, default=0.85)
    parser.add_argument("--rhythm-amount", type=float, default=None)
    parser.add_argument("--pitch-method", choices=["yin", "pyin"], default="yin")
    parser.add_argument("--fmin", default="C2")
    parser.add_argument("--fmax", default="C7")
    parser.add_argument("--voiced-threshold", type=float, default=0.55)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--smoothing-frames", type=int, default=5)
    parser.add_argument("--gap-ms", type=float, default=12.0)
    parser.add_argument("--timbre", choices=["sine", "flute", "reed", "pluck"], default="flute")
    parser.add_argument("--seed", type=int, default=1479)
    parser.add_argument("--no-ornaments", action="store_true")
    parser.add_argument("--no-rhythm", action="store_true")
    parser.add_argument("--no-microtuning", action="store_true")
    parser.add_argument("--no-modulation", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    profiles = load_style_profiles(args.style_dir)
    if args.list_styles:
        for key in sorted(profiles):
            p = profiles[key]
            print(f"{key:20s} {p.label}")
        return

    if args.input is None:
        parser.error("input is required unless --list-styles is used")
    if args.style is None:
        parser.error("--style is required")
    if args.style not in profiles:
        parser.error(f"unknown style '{args.style}'. Use --list-styles.")
    if not 0.0 <= args.style_amount <= 1.0:
        parser.error("--style-amount must be 0..1")

    profile = profiles[args.style]
    rhythm_amount = args.style_amount if args.rhythm_amount is None else args.rhythm_amount
    if args.no_rhythm:
        rhythm_amount = 0.0
    if not 0.0 <= rhythm_amount <= 1.0:
        parser.error("--rhythm-amount must be 0..1")

    y, sr = decode_audio_robust(args.input)
    root_pc = parse_root(args.root)
    if root_pc is None:
        root_pc, mode = detect_key(y, sr)
        print(f"Detected key/root: {NOTE_NAMES[root_pc]} {mode}")
    else:
        print(f"Root: {NOTE_NAMES[root_pc]}")

    source_midi, confidence = extract_source_pitch(
        y=y,
        sr=sr,
        pitch_method=args.pitch_method,
        fmin_note=args.fmin,
        fmax_note=args.fmax,
        hop_length=args.hop_length,
        voiced_threshold=args.voiced_threshold,
        smoothing_frames=args.smoothing_frames,
        gap_ms=args.gap_ms,
    )

    mapping = map_melody_viterbi(
        source_midi=source_midi,
        sr=sr,
        hop_length=args.hop_length,
        root_pc=root_pc,
        profile=profile,
        style_amount=args.style_amount,
        enable_modulation=not args.no_modulation,
    )
    print(f"Detected events: {sum(len(p.events) for p in mapping.phrases)}")
    print(f"Detected phrases: {len(mapping.phrases)}")
    if any(name != "base" for name in mapping.modulation_names):
        print("Phrase modes:", ", ".join(mapping.modulation_names))

    render = render_mapping(
        y=y,
        sr=sr,
        hop_length=args.hop_length,
        mapping=mapping,
        profile=profile,
        timbre=args.timbre,
        style_amount=args.style_amount,
        rhythm_amount=rhythm_amount,
        enable_ornaments=not args.no_ornaments,
        enable_microtuning=not args.no_microtuning,
        seed=args.seed,
    )

    output = args.output or args.input.with_name(f"{args.input.stem}_{profile.id}_v9.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), render.audio, sr, subtype="PCM_24")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.report_dir / f"{args.input.stem}_{profile.id}_events.csv"
    metrics_path = args.report_dir / f"{args.input.stem}_{profile.id}_metrics.json"
    write_event_report(csv_path, render, profile)

    metrics = compute_metrics(mapping, render, profile)
    metrics.update({
        "root": NOTE_NAMES[root_pc],
        "style_amount": args.style_amount,
        "rhythm_amount": rhythm_amount,
        "ornaments_enabled": not args.no_ornaments,
        "microtuning_enabled": not args.no_microtuning,
        "modulation_enabled": not args.no_modulation,
        "pitch_method": args.pitch_method,
    })
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved audio:   {output}")
    print(f"Event report:  {csv_path}")
    print(f"Metrics:       {metrics_path}")
    print(f"Grammar score: {metrics['style_grammar_score']:.3f}")
    print(f"Melody score:  {metrics['melody_preservation_score']:.3f}")


if __name__ == "__main__":
    main()
