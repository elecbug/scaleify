#!/usr/bin/env python3
"""
train_style.py

Tune an existing Scaleify v9.1+ style JSON from a folder of monophonic WAV files.

Usage
-----
python train_style.py styles/japanese_in.json corpus/japanese

By default this NEVER overwrites the source JSON.
For:
    styles/japanese_in.json

it writes:
    styles/japanese_in_tuned.json

and changes the style id to:
    japanese_in_tuned

If that file already exists:
    japanese_in_tuned_2.json
    japanese_in_tuned_3.json
    ...

The source style JSON supplies:
- scale
- segmentation parameters
- transform safety weights
- ornament definitions
- modulation definitions

The WAV corpus tunes:
- degree_weights
- interval_weights
- transition_weights
- ascending_transition_weights
- descending_transition_weights
- trigram_weights
- preferred_phrases
- cadence_degrees
- cadence_patterns
- rhythm.preferred_duration_ratios
- rhythm.degree_duration_multipliers
- rhythm.phrase_end_multiplier
- tuning.degree_cents

The trainer intentionally preserves parameters which cannot be estimated
reliably from a target-only corpus, such as pitch_deviation_weight,
contour_penalty, ornament probabilities, and modulation policy.

Important
---------
The corpus should preferably contain isolated/predominantly monophonic melody.
Completed polyphonic mixes will contaminate F0 and onset statistics.

This is a target-corpus-only estimator.  Without a generic baseline corpus,
weights represent "characteristic within this corpus", not strict
discriminative evidence against all other musical styles.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from style_profiles import (NoteEvent, Phrase, StyleProfile, degree_of_midi,
                            detect_phrases, extract_note_events,
                            load_style_profile)

from scaleify import (NOTE_NAMES, decode_audio_robust, detect_note_onsets,
                      extract_source_pitch, parse_root)

EPS = 1e-12


@dataclass
class FileAnalysis:
    path: Path
    root_pc: int
    root_score: float
    scale_coverage: float
    sr: int
    events: tuple[NoteEvent, ...]
    phrases: tuple[Phrase, ...]
    global_tuning_cents: float


@dataclass
class CorpusStats:
    files: list[FileAnalysis] = field(default_factory=list)

    degree_duration: Counter = field(default_factory=Counter)
    degree_count: Counter = field(default_factory=Counter)
    interval_count: Counter = field(default_factory=Counter)

    transition_count: Counter = field(default_factory=Counter)
    ascending_transition_count: Counter = field(default_factory=Counter)
    descending_transition_count: Counter = field(default_factory=Counter)
    trigram_count: Counter = field(default_factory=Counter)

    ngram_count: dict[int, Counter] = field(
        default_factory=lambda: {2: Counter(), 3: Counter(), 4: Counter(), 5: Counter()}
    )
    ngram_file_support: dict[int, Counter] = field(
        default_factory=lambda: {2: Counter(), 3: Counter(), 4: Counter(), 5: Counter()}
    )

    cadence_degree_count: Counter = field(default_factory=Counter)
    cadence_pattern_count: dict[int, Counter] = field(
        default_factory=lambda: {2: Counter(), 3: Counter(), 4: Counter()}
    )

    duration_ratios: list[float] = field(default_factory=list)
    degree_duration_ratios: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    phrase_end_duration_ratios: list[float] = field(default_factory=list)

    tuning_cents_by_degree: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))


def note_name(pc: int) -> str:
    return NOTE_NAMES[int(pc) % 12]


def recursive_wavs(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(p for p in folder.glob(pattern) if p.is_file())


def unique_output_path(source_json: Path, explicit: Path | None) -> Path:
    """Choose an output path which never overwrites the source JSON."""
    source_resolved = source_json.resolve()

    if explicit is not None:
        candidate = explicit
        if candidate.suffix.lower() != ".json":
            candidate = candidate.with_suffix(".json")
        if candidate.resolve() == source_resolved:
            raise ValueError("--output must not point to the source JSON")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if not candidate.exists():
            return candidate

        # Explicit output is still non-destructive: create numbered sibling.
        stem = candidate.stem
        for i in range(2, 10000):
            alt = candidate.with_name(f"{stem}_{i}{candidate.suffix}")
            if not alt.exists():
                return alt
        raise RuntimeError("Could not find a free output filename")

    base = source_json.with_name(f"{source_json.stem}_tuned.json")
    if not base.exists():
        return base

    for i in range(2, 10000):
        alt = source_json.with_name(f"{source_json.stem}_tuned_{i}.json")
        if not alt.exists():
            return alt

    raise RuntimeError("Could not find a free tuned filename")


def normalize_style_id(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in "-_ ":
            out.append("_")
    value = "".join(out)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_") or "tuned_style"


def estimate_root_from_profile(
    events: tuple[NoteEvent, ...],
    phrases: tuple[Phrase, ...],
    profile: StyleProfile,
) -> tuple[int, float, float]:
    """
    Estimate tonic by testing all 12 transpositions of the target profile scale.

    The score combines:
    - pitch-class coverage inside the profile scale
    - tonic duration
    - fifth duration
    - phrase-ending tonic evidence

    This works better for modal profiles than a major/minor key detector, but
    tonic ambiguity remains possible.  Use --root when all corpus files share
    a known tonic.
    """
    if not events:
        return 0, -np.inf, 0.0

    pc_duration = np.zeros(12, dtype=np.float64)
    total = 0.0
    for event in events:
        pc = int(round(event.source_midi)) % 12
        w = max(1.0, float(event.frames))
        pc_duration[pc] += w
        total += w

    endings: list[int] = []
    for phrase in phrases:
        if phrase.events:
            endings.append(int(round(phrase.events[-1].source_midi)) % 12)

    scale = tuple(int(x) % 12 for x in profile.scale)
    best_root = 0
    best_score = -np.inf
    best_coverage = 0.0

    for root in range(12):
        allowed = {(root + d) % 12 for d in scale}
        coverage = float(sum(pc_duration[pc] for pc in allowed) / max(total, EPS))
        tonic_share = float(pc_duration[root] / max(total, EPS))

        fifth_pc = (root + 7) % 12
        fifth_share = float(pc_duration[fifth_pc] / max(total, EPS)) if 7 in scale else 0.0

        ending_share = (
            sum(1 for pc in endings if pc == root) / len(endings)
            if endings else 0.0
        )

        # Coverage dominates; tonic/cadence resolve modal rotations.
        score = (
            5.0 * coverage
            + 0.85 * tonic_share
            + 0.30 * fifth_share
            + 1.15 * ending_share
        )

        if score > best_score:
            best_score = score
            best_root = root
            best_coverage = coverage

    return best_root, float(best_score), float(best_coverage)


def cents_residual(midi_value: float) -> float:
    """Return residual cents to nearest 12-TET semitone in [-50, 50)."""
    nearest = round(float(midi_value))
    cents = (float(midi_value) - nearest) * 100.0
    while cents >= 50.0:
        cents -= 100.0
    while cents < -50.0:
        cents += 100.0
    return cents


def robust_global_tuning(events: Iterable[NoteEvent]) -> float:
    values = [cents_residual(e.source_midi) for e in events]
    if not values:
        return 0.0
    return float(np.median(values))


def sequence_ngrams(seq: list[int], n: int):
    for i in range(0, len(seq) - n + 1):
        yield tuple(seq[i:i + n])


def smoothed_log_lift(observed: float, expected: float, alpha: float = 0.5) -> float:
    return math.log((observed + alpha) / (expected + alpha))


def clipped_positive_weight(
    raw: float,
    scale: float,
    max_weight: float,
    min_weight: float = 0.05,
) -> float | None:
    value = max(0.0, float(raw)) * float(scale)
    value = min(value, float(max_weight))
    if value < min_weight:
        return None
    return round(value, 4)


def transition_pmi_weights(
    counts: Counter,
    source_degree_counts: Counter,
    target_degree_counts: Counter,
    total: int,
    min_count: int,
    weight_scale: float,
    max_weight: float,
    top_k: int,
) -> dict[str, float]:
    if total <= 0:
        return {}

    scored: list[tuple[str, float, int]] = []
    for (a, b), c in counts.items():
        if c < min_count:
            continue
        p_a = source_degree_counts[a] / total
        p_b = target_degree_counts[b] / total
        expected = total * p_a * p_b
        raw = smoothed_log_lift(c, expected)
        weight = clipped_positive_weight(raw, weight_scale, max_weight)
        if weight is not None:
            scored.append((f"{a}>{b}", weight, c))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return {key: weight for key, weight, _ in scored[:top_k]}


def ngram_association_weight(
    pattern: tuple[int, ...],
    count: int,
    degree_counts: Counter,
    total_degrees: int,
    total_windows: int,
    weight_scale: float,
    max_weight: float,
) -> float | None:
    if total_degrees <= 0 or total_windows <= 0:
        return None

    probability_product = 1.0
    for degree in pattern:
        probability_product *= max(degree_counts[degree] / total_degrees, EPS)

    expected = total_windows * probability_product
    raw = smoothed_log_lift(count, expected)
    return clipped_positive_weight(raw, weight_scale, max_weight)


def infer_preferred_duration_ratios(values: list[float], max_bins: int = 6) -> list[float]:
    if not values:
        return [0.5, 1.0, 2.0]

    # Musically useful ratio grid.  We count corpus values into nearest bins
    # rather than performing unconstrained clustering, which can produce
    # hard-to-interpret arbitrary decimals.
    grid = np.asarray(
        [0.25, 1/3, 0.5, 2/3, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )
    counts = Counter()

    for value in values:
        value = float(np.clip(value, 0.20, 4.5))
        idx = int(np.argmin(np.abs(np.log(grid) - math.log(value))))
        counts[float(grid[idx])] += 1

    selected = [ratio for ratio, _ in counts.most_common(max_bins)]
    if 1.0 not in selected:
        selected.append(1.0)

    return [round(float(x), 4) for x in sorted(set(selected))]


def analyze_wav(
    path: Path,
    profile: StyleProfile,
    forced_root: int | None,
    pitch_method: str,
    fmin: str,
    fmax: str,
    hop_length: int,
    voiced_threshold: float,
    smoothing_frames: int,
    gap_ms: float,
    onset_segmentation: bool,
    onset_delta: float,
    onset_min_separation_ms: float,
    onset_retrigger_min_ms: float,
) -> FileAnalysis | None:
    y, sr = decode_audio_robust(path)

    source_midi, _ = extract_source_pitch(
        y=y,
        sr=sr,
        pitch_method=pitch_method,
        fmin_note=fmin,
        fmax_note=fmax,
        hop_length=hop_length,
        voiced_threshold=voiced_threshold,
        smoothing_frames=smoothing_frames,
        gap_ms=gap_ms,
    )

    if onset_segmentation:
        onset_frames = detect_note_onsets(
            y=y,
            sr=sr,
            hop_length=hop_length,
            delta=onset_delta,
            min_separation_ms=onset_min_separation_ms,
        )
    else:
        onset_frames = np.asarray([], dtype=np.int64)

    onset_retrigger_min_frames = max(
        1,
        int(round(onset_retrigger_min_ms / 1000.0 * sr / hop_length)),
    )

    events = extract_note_events(
        source_midi,
        profile.grammar,
        onset_frames=onset_frames,
        onset_retrigger_min_frames=onset_retrigger_min_frames,
    )

    if len(events) < 2:
        print(f"[skip] {path.name}: too few note events ({len(events)})")
        return None

    phrases = detect_phrases(events, sr, hop_length, profile.grammar)
    if not phrases:
        print(f"[skip] {path.name}: no phrases detected")
        return None

    if forced_root is None:
        root_pc, root_score, coverage = estimate_root_from_profile(events, phrases, profile)
    else:
        root_pc = forced_root
        root_score = 0.0
        allowed = {(root_pc + d) % 12 for d in profile.scale}
        duration_total = sum(max(1, e.frames) for e in events)
        duration_inside = sum(
            max(1, e.frames)
            for e in events
            if int(round(e.source_midi)) % 12 in allowed
        )
        coverage = duration_inside / max(duration_total, 1)

    global_tuning = robust_global_tuning(events)

    return FileAnalysis(
        path=path,
        root_pc=root_pc,
        root_score=root_score,
        scale_coverage=float(coverage),
        sr=sr,
        events=events,
        phrases=phrases,
        global_tuning_cents=global_tuning,
    )


def accumulate_file(
    stats: CorpusStats,
    analysis: FileAnalysis,
    profile: StyleProfile,
    hop_length: int,
) -> None:
    stats.files.append(analysis)

    # File-local support sets stop one song from dominating preferred phrases.
    support_seen = {2: set(), 3: set(), 4: set(), 5: set()}

    for phrase in analysis.phrases:
        if not phrase.events:
            continue

        degrees = [
            degree_of_midi(round(event.source_midi), analysis.root_pc)
            for event in phrase.events
        ]

        durations = np.asarray(
            [event.frames * hop_length / analysis.sr for event in phrase.events],
            dtype=np.float64,
        )
        phrase_median = max(float(np.median(durations)), 1e-6)

        for event, degree, duration in zip(phrase.events, degrees, durations):
            stats.degree_count[degree] += 1
            stats.degree_duration[degree] += float(duration)

            ratio = float(duration / phrase_median)
            stats.duration_ratios.append(ratio)
            stats.degree_duration_ratios[degree].append(ratio)

            # Remove each file's global A4/tuning offset before estimating
            # degree-specific intonation.
            cents = cents_residual(event.source_midi) - analysis.global_tuning_cents
            while cents >= 50.0:
                cents -= 100.0
            while cents < -50.0:
                cents += 100.0
            stats.tuning_cents_by_degree[degree].append(float(cents))

        stats.phrase_end_duration_ratios.append(float(durations[-1] / phrase_median))
        stats.cadence_degree_count[degrees[-1]] += 1

        for i in range(1, len(phrase.events)):
            prev_event = phrase.events[i - 1]
            event = phrase.events[i]
            a, b = degrees[i - 1], degrees[i]

            delta = float(event.source_midi - prev_event.source_midi)
            interval = min(12, int(round(abs(delta))))
            if interval > 0:
                stats.interval_count[interval] += 1

            stats.transition_count[(a, b)] += 1
            if delta > 0.35:
                stats.ascending_transition_count[(a, b)] += 1
            elif delta < -0.35:
                stats.descending_transition_count[(a, b)] += 1

        for tri in sequence_ngrams(degrees, 3):
            stats.trigram_count[tri] += 1

        for n in (2, 3, 4, 5):
            grams = list(sequence_ngrams(degrees, n))
            for gram in grams:
                stats.ngram_count[n][gram] += 1
                support_seen[n].add(gram)

        for n in (2, 3, 4):
            if len(degrees) >= n:
                stats.cadence_pattern_count[n][tuple(degrees[-n:])] += 1

    for n in (2, 3, 4, 5):
        for gram in support_seen[n]:
            stats.ngram_file_support[n][gram] += 1


def tune_json(
    original: dict,
    profile: StyleProfile,
    stats: CorpusStats,
    output_id: str,
    output_label: str,
    min_count: int,
    min_file_support: int,
    max_weight: float,
    top_intervals: int,
    top_transitions: int,
    top_trigrams: int,
    top_phrases: int,
    top_cadences: int,
    learn_microtuning: bool,
) -> tuple[dict, dict]:
    tuned = deepcopy(original)
    grammar = tuned.setdefault("grammar", {})
    rhythm = tuned.setdefault("rhythm", {})
    tuning = tuned.setdefault("tuning", {})

    total_degree_duration = float(sum(stats.degree_duration.values()))
    total_degree_count = int(sum(stats.degree_count.values()))
    scale_size = max(1, len(profile.scale))

    # ------------------------------------------------------------------
    # Degree weights: duration-weighted lift against a uniform scale degree.
    # ------------------------------------------------------------------
    degree_weights: dict[str, float] = {}
    for degree in profile.scale:
        observed = float(stats.degree_duration.get(degree, 0.0))
        if observed <= 0 or total_degree_duration <= 0:
            continue

        p = observed / total_degree_duration
        uniform = 1.0 / scale_size
        raw = math.log((p + 1e-6) / uniform)
        weight = clipped_positive_weight(raw, 1.45, max_weight)
        if weight is not None:
            degree_weights[str(degree)] = weight

    if degree_weights:
        grammar["degree_weights"] = degree_weights

    # ------------------------------------------------------------------
    # Interval weights: frequency lift against uniform 1..12 interval bins.
    # ------------------------------------------------------------------
    interval_total = int(sum(stats.interval_count.values()))
    interval_scored: list[tuple[int, float, int]] = []
    if interval_total > 0:
        expected = interval_total / 12.0
        for interval, count in stats.interval_count.items():
            if count < min_count:
                continue
            raw = smoothed_log_lift(count, expected)
            weight = clipped_positive_weight(raw, 1.25, max_weight)
            if weight is not None:
                interval_scored.append((interval, weight, count))

    interval_scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    if interval_scored:
        grammar["interval_weights"] = {
            str(interval): weight
            for interval, weight, _ in interval_scored[:top_intervals]
        }

    # Marginals for pair PMI.
    pair_source = Counter()
    pair_target = Counter()
    for (a, b), count in stats.transition_count.items():
        pair_source[a] += count
        pair_target[b] += count
    pair_total = int(sum(stats.transition_count.values()))

    learned_transitions = transition_pmi_weights(
        stats.transition_count,
        pair_source,
        pair_target,
        pair_total,
        min_count,
        1.10,
        max_weight,
        top_transitions,
    )
    if learned_transitions:
        grammar["transition_weights"] = learned_transitions

    # Direction-specific pair PMI.
    def direction_weights(counter: Counter) -> dict[str, float]:
        src = Counter()
        dst = Counter()
        total = int(sum(counter.values()))
        for (a, b), count in counter.items():
            src[a] += count
            dst[b] += count
        return transition_pmi_weights(
            counter, src, dst, total,
            min_count, 1.15, max_weight, top_transitions
        )

    asc = direction_weights(stats.ascending_transition_count)
    desc = direction_weights(stats.descending_transition_count)
    if asc:
        grammar["ascending_transition_weights"] = asc
    if desc:
        grammar["descending_transition_weights"] = desc

    # ------------------------------------------------------------------
    # Trigrams.
    # ------------------------------------------------------------------
    trigram_total = int(sum(stats.trigram_count.values()))
    trigram_scored: list[tuple[str, float, int]] = []
    for pattern, count in stats.trigram_count.items():
        if count < min_count:
            continue
        weight = ngram_association_weight(
            pattern,
            count,
            stats.degree_count,
            total_degree_count,
            trigram_total,
            weight_scale=0.72,
            max_weight=max_weight,
        )
        if weight is not None:
            trigram_scored.append((">".join(map(str, pattern)), weight, count))

    trigram_scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    if trigram_scored:
        grammar["trigram_weights"] = {
            key: weight for key, weight, _ in trigram_scored[:top_trigrams]
        }

    # ------------------------------------------------------------------
    # Preferred phrases: 4/5-grams with cross-file support.
    # ------------------------------------------------------------------
    phrase_scored: list[tuple[tuple[int, ...], float, int, int]] = []
    actual_min_file_support = min(
        max(1, min_file_support),
        max(1, len(stats.files)),
    )

    for n in (4, 5):
        total_windows = int(sum(stats.ngram_count[n].values()))
        for pattern, count in stats.ngram_count[n].items():
            support = int(stats.ngram_file_support[n][pattern])
            if count < min_count or support < actual_min_file_support:
                continue

            weight = ngram_association_weight(
                pattern,
                count,
                stats.degree_count,
                total_degree_count,
                total_windows,
                weight_scale=0.62,
                max_weight=max_weight,
            )
            if weight is not None:
                # Cross-file support slightly boosts generalizable patterns.
                support_bonus = min(0.45, 0.08 * max(0, support - 1))
                phrase_scored.append(
                    (pattern, round(min(max_weight, weight + support_bonus), 4), count, support)
                )

    phrase_scored.sort(key=lambda x: (x[1], x[3], x[2]), reverse=True)
    if phrase_scored:
        grammar["preferred_phrases"] = [
            {
                "degrees": list(pattern),
                "weight": weight,
                "support_count": count,
                "support_files": support,
            }
            for pattern, weight, count, support in phrase_scored[:top_phrases]
        ]

    # ------------------------------------------------------------------
    # Cadence degree lift: phrase-ending frequency vs overall degree frequency.
    # ------------------------------------------------------------------
    phrase_count = sum(stats.cadence_degree_count.values())
    cadence_degree_weights: dict[str, float] = {}
    if phrase_count > 0 and total_degree_count > 0:
        for degree, end_count in stats.cadence_degree_count.items():
            if end_count < max(1, min_count // 2):
                continue
            p_end = end_count / phrase_count
            p_all = stats.degree_count[degree] / total_degree_count
            raw = math.log((p_end + 1e-6) / (p_all + 1e-6))
            weight = clipped_positive_weight(raw, 1.15, max_weight)
            if weight is not None:
                cadence_degree_weights[str(degree)] = weight

    if cadence_degree_weights:
        grammar["cadence_degrees"] = cadence_degree_weights

    # Cadence patterns: phrase-suffix overrepresentation vs all windows.
    cadence_scored: list[tuple[tuple[int, ...], float, int]] = []
    for n in (2, 3, 4):
        total_windows = int(sum(stats.ngram_count[n].values()))
        if total_windows <= 0 or phrase_count <= 0:
            continue

        for pattern, suffix_count in stats.cadence_pattern_count[n].items():
            if suffix_count < max(1, min_count // 2):
                continue
            global_count = stats.ngram_count[n][pattern]
            p_suffix = suffix_count / phrase_count
            p_global = global_count / total_windows if total_windows else 0.0
            raw = math.log((p_suffix + 1e-6) / (p_global + 1e-6))
            weight = clipped_positive_weight(raw, 0.95, max_weight)
            if weight is not None:
                cadence_scored.append((pattern, weight, suffix_count))

    cadence_scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    if cadence_scored:
        grammar["cadence_patterns"] = [
            {
                "degrees": list(pattern),
                "weight": weight,
                "support_count": count,
            }
            for pattern, weight, count in cadence_scored[:top_cadences]
        ]

    # ------------------------------------------------------------------
    # Rhythm profile.
    # ------------------------------------------------------------------
    if stats.duration_ratios:
        rhythm["preferred_duration_ratios"] = infer_preferred_duration_ratios(
            stats.duration_ratios
        )

    degree_duration_multipliers: dict[str, float] = {}
    for degree, values in stats.degree_duration_ratios.items():
        if len(values) < min_count:
            continue
        median_ratio = float(np.median(values))
        multiplier = float(np.clip(median_ratio, 0.70, 1.50))
        if abs(multiplier - 1.0) >= 0.04:
            degree_duration_multipliers[str(degree)] = round(multiplier, 4)

    if degree_duration_multipliers:
        rhythm["degree_duration_multipliers"] = degree_duration_multipliers

    if stats.phrase_end_duration_ratios:
        rhythm["phrase_end_multiplier"] = round(
            float(np.clip(np.median(stats.phrase_end_duration_ratios), 0.80, 1.80)),
            4,
        )

    # ------------------------------------------------------------------
    # Degree-specific intonation after per-file global tuning removal.
    # ------------------------------------------------------------------
    if learn_microtuning:
        degree_cents = {}
        tuning_min_count = max(5, min_count * 2)
        for degree, values in stats.tuning_cents_by_degree.items():
            if len(values) < tuning_min_count:
                continue

            arr = np.asarray(values, dtype=np.float64)
            # Reject wild pitch-tracker tails.
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            if mad > 0:
                keep = np.abs(arr - med) <= max(8.0, 3.5 * mad)
                arr = arr[keep]

            if len(arr) < tuning_min_count:
                continue

            cents = float(np.clip(np.median(arr), -35.0, 35.0))
            if abs(cents) >= 2.0:
                degree_cents[str(degree)] = round(cents, 2)

        tuning["degree_cents"] = degree_cents

    # ------------------------------------------------------------------
    # Identity/provenance.  A new ID avoids duplicate IDs if both JSON files
    # remain in the same styles directory.
    # ------------------------------------------------------------------
    original_id = str(original.get("id", profile.id))
    tuned["id"] = output_id
    tuned["label"] = output_label

    report = {
        "source_profile_id": original_id,
        "output_profile_id": output_id,
        "files_analyzed": len(stats.files),
        "events": int(sum(len(f.events) for f in stats.files)),
        "phrases": int(sum(len(f.phrases) for f in stats.files)),
        "scale": list(profile.scale),
        "mean_scale_coverage": round(
            float(np.mean([f.scale_coverage for f in stats.files])) if stats.files else 0.0,
            4,
        ),
        "roots": {
            f.path.name: {
                "root": note_name(f.root_pc),
                "scale_coverage": round(f.scale_coverage, 4),
                "global_tuning_cents_removed": round(f.global_tuning_cents, 2),
            }
            for f in stats.files
        },
        "learned_counts": {
            "degree_weights": len(grammar.get("degree_weights", {})),
            "interval_weights": len(grammar.get("interval_weights", {})),
            "transition_weights": len(grammar.get("transition_weights", {})),
            "ascending_transition_weights": len(grammar.get("ascending_transition_weights", {})),
            "descending_transition_weights": len(grammar.get("descending_transition_weights", {})),
            "trigram_weights": len(grammar.get("trigram_weights", {})),
            "preferred_phrases": len(grammar.get("preferred_phrases", [])),
            "cadence_degrees": len(grammar.get("cadence_degrees", {})),
            "cadence_patterns": len(grammar.get("cadence_patterns", [])),
            "degree_duration_multipliers": len(rhythm.get("degree_duration_multipliers", {})),
            "degree_cents": len(tuning.get("degree_cents", {})),
        },
    }

    tuned["training"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_profile_id": original_id,
        "method": "scaleify_target_corpus_v1",
        "files_analyzed": len(stats.files),
        "events": report["events"],
        "phrases": report["phrases"],
        "mean_scale_coverage": report["mean_scale_coverage"],
        "note": (
            "Weights estimated from target WAV corpus only. "
            "Transform-safety parameters, ornaments and modulation policy were preserved."
        ),
    }

    return tuned, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a Scaleify style JSON from all WAV files in a folder. "
            "The source JSON is never overwritten."
        )
    )
    parser.add_argument("style_json", type=Path, help="Existing style JSON to tune.")
    parser.add_argument("wav_folder", type=Path, help="Folder containing training WAV files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path. If omitted, writes <name>_tuned.json. "
            "Existing files are never overwritten; a numeric suffix is added."
        ),
    )
    parser.add_argument(
        "--id",
        default=None,
        help="Optional output style id. Default is derived from output filename.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional output label. Default: '<original label> (Tuned)'.",
    )
    parser.add_argument(
        "--root",
        default="auto",
        help=(
            "auto estimates tonic per WAV using the target scale. "
            "Use C, D, F#, etc. if every corpus WAV shares a known tonic."
        ),
    )
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--pitch-method", choices=["yin", "pyin"], default="yin")
    parser.add_argument("--fmin", default="C2")
    parser.add_argument("--fmax", default="C7")
    parser.add_argument("--voiced-threshold", type=float, default=0.55)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--smoothing-frames", type=int, default=5)
    parser.add_argument("--gap-ms", type=float, default=12.0)
    parser.add_argument("--no-onset-segmentation", action="store_true")
    parser.add_argument("--onset-delta", type=float, default=0.15)
    parser.add_argument("--onset-min-separation-ms", type=float, default=70.0)
    parser.add_argument("--onset-retrigger-min-ms", type=float, default=80.0)

    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--min-file-support", type=int, default=2)
    parser.add_argument("--max-weight", type=float, default=3.5)
    parser.add_argument("--top-intervals", type=int, default=8)
    parser.add_argument("--top-transitions", type=int, default=28)
    parser.add_argument("--top-trigrams", type=int, default=24)
    parser.add_argument("--top-phrases", type=int, default=16)
    parser.add_argument("--top-cadences", type=int, default=12)
    parser.add_argument("--no-microtuning", action="store_true")

    args = parser.parse_args()

    if not args.style_json.is_file():
        parser.error(f"style JSON not found: {args.style_json}")
    if not args.wav_folder.is_dir():
        parser.error(f"WAV folder not found: {args.wav_folder}")
    if args.min_count < 1:
        parser.error("--min-count must be >= 1")
    if args.min_file_support < 1:
        parser.error("--min-file-support must be >= 1")
    if args.max_weight <= 0:
        parser.error("--max-weight must be > 0")

    output_path = unique_output_path(args.style_json, args.output)

    original = json.loads(args.style_json.read_text(encoding="utf-8"))
    profile = load_style_profile(args.style_json)

    forced_root = parse_root(args.root)
    wavs = recursive_wavs(args.wav_folder, recursive=not args.no_recursive)
    if not wavs:
        parser.error(f"No WAV files found in: {args.wav_folder}")

    print(f"Source profile: {args.style_json}")
    print(f"Corpus:         {args.wav_folder}")
    print(f"WAV files:      {len(wavs)}")
    print(f"Scale:          {list(profile.scale)}")
    print()

    stats = CorpusStats()
    failed: list[tuple[str, str]] = []

    for index, path in enumerate(wavs, start=1):
        print(f"[{index}/{len(wavs)}] {path}")
        try:
            analysis = analyze_wav(
                path=path,
                profile=profile,
                forced_root=forced_root,
                pitch_method=args.pitch_method,
                fmin=args.fmin,
                fmax=args.fmax,
                hop_length=args.hop_length,
                voiced_threshold=args.voiced_threshold,
                smoothing_frames=args.smoothing_frames,
                gap_ms=args.gap_ms,
                onset_segmentation=not args.no_onset_segmentation,
                onset_delta=args.onset_delta,
                onset_min_separation_ms=args.onset_min_separation_ms,
                onset_retrigger_min_ms=args.onset_retrigger_min_ms,
            )

            if analysis is None:
                failed.append((str(path), "insufficient note/phrase data"))
                continue

            accumulate_file(stats, analysis, profile, args.hop_length)

            print(
                f"    root={note_name(analysis.root_pc)} "
                f"coverage={analysis.scale_coverage:.3f} "
                f"events={len(analysis.events)} "
                f"phrases={len(analysis.phrases)} "
                f"global_tuning={analysis.global_tuning_cents:+.1f}c"
            )
        except Exception as exc:
            failed.append((str(path), f"{type(exc).__name__}: {exc}"))
            print(f"    [failed] {type(exc).__name__}: {exc}")

    if not stats.files:
        raise RuntimeError("No WAV files could be analyzed successfully.")

    # The output file's stem becomes the new style ID by default.
    auto_id = normalize_style_id(output_path.stem)
    output_id = normalize_style_id(args.id) if args.id else auto_id

    # Never accidentally duplicate the original ID when saved beside it.
    if output_id == profile.id:
        output_id = f"{profile.id}_tuned"

    output_label = args.label or f"{profile.label} (Tuned)"

    tuned, report = tune_json(
        original=original,
        profile=profile,
        stats=stats,
        output_id=output_id,
        output_label=output_label,
        min_count=args.min_count,
        min_file_support=args.min_file_support,
        max_weight=args.max_weight,
        top_intervals=args.top_intervals,
        top_transitions=args.top_transitions,
        top_trigrams=args.top_trigrams,
        top_phrases=args.top_phrases,
        top_cadences=args.top_cadences,
        learn_microtuning=not args.no_microtuning,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(tuned, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report["failed_files"] = [
        {"path": path, "reason": reason}
        for path, reason in failed
    ]
    report["output_json"] = str(output_path)
    report_path = output_path.with_name(f"{output_path.stem}_training_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Training complete")
    print("-----------------")
    print(f"Analyzed files:   {len(stats.files)}/{len(wavs)}")
    print(f"Events:           {report['events']}")
    print(f"Phrases:          {report['phrases']}")
    print(f"Mean scale cover: {report['mean_scale_coverage']:.3f}")
    print(f"Output style id:  {output_id}")
    print(f"Saved JSON:       {output_path}")
    print(f"Training report:  {report_path}")

    if failed:
        print(f"Warnings:         {len(failed)} file(s) skipped/failed")


if __name__ == "__main__":
    main()