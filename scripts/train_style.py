#!/usr/bin/env python3
"""
train_style.py

Unsupervised corpus -> Scaleify style JSON trainer with iterative root refinement.

NO input JSON is required.

Typical usage
-------------
    python train_style.py dataset/japan

Default outputs
---------------
    styles/generated/japan_cluster_1.json
    styles/generated/japan_cluster_2.json
    ...
    styles/generated/japan_cluster_assignments.csv
    styles/generated/japan_cluster_report.json

The trainer:
1. loads every WAV in the dataset folder,
2. extracts F0 + onset-aware note events,
3. estimates a tonic/root for each file,
4. builds root-relative melodic/rhythmic feature vectors,
5. clusters the songs,
6. infers a scale independently for each cluster,
7. iteratively re-estimates each file's tonic using its cluster scale and re-clusters until stable,
8. separates a compact core scale from lower-frequency auxiliary degrees,
9. learns grammar/rhythm/tuning only from notes belonging to the core scale,
10. writes one directly-loadable Scaleify JSON per cluster.

Important design choices
------------------------
- No existing style JSON is used.
- Same-degree repetitions are kept for rhythm statistics but removed from
  melodic transition/n-gram statistics.
- Out-of-scale notes BREAK melodic runs; notes before and after them are never
  joined into a fake transition.
- Cadence patterns require both occurrence count and cross-file support.
- Root estimation is refined iteratively against the currently inferred cluster scale.
- Scale inference uses a coverage+elbow rule to prefer a compact core scale; lower-frequency notes are recorded as auxiliary degrees.
- Ornament and modulation rules are NOT hallucinated from the corpus.
  Generated profiles therefore start with ornaments=[] and modulation disabled.
- Transform-safety parameters such as pitch_deviation_weight and contour_penalty
  are generic engine defaults, because a target-only corpus cannot tell us how
  aggressively an arbitrary input melody should be rewritten.

Auto clustering
---------------
`--clusters auto` evaluates K=2..N using a small NumPy k-means implementation
and silhouette score. Cluster solutions containing tiny clusters are rejected.
If no stable multi-cluster solution is available, a single style is emitted.

This tool is intended for monophonic or strongly melody-dominant WAVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from itertools import permutations

import numpy as np

from scaleify import (
    NOTE_NAMES,
    decode_audio_robust,
    detect_note_onsets,
    extract_source_pitch,
)
from style_profiles import (
    GrammarProfile,
    NoteEvent,
    Phrase,
    detect_phrases,
    degree_of_midi,
    extract_note_events,
    load_style_profile,
)

EPS = 1e-12

# ---------------------------------------------------------------------------
# Generic engine defaults.
# These are not "musical style" measurements; they control how strongly
# Scaleify should preserve an arbitrary source melody during later conversion.
# ---------------------------------------------------------------------------

DEFAULT_GRAMMAR = GrammarProfile(
    pitch_deviation_weight=0.90,
    motion_preservation_weight=0.25,
    contour_penalty=0.80,
    leap_penalty=0.08,
    max_preferred_leap=7.0,
    candidate_shift_semitones=4.5,
    candidate_count=7,
    event_pitch_change=0.8,
    min_event_frames=4,
    phrase_gap_ms=150.0,
    phrase_long_note_factor=1.65,
    phrase_leap_semitones=7.0,
    min_phrase_events=3,
)

RHYTHM_GRID = np.asarray(
    [0.25, 1/3, 0.5, 2/3, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
    dtype=np.float64,
)


@dataclass
class FileAnalysis:
    path: Path
    sr: int
    events: tuple[NoteEvent, ...]
    phrases: tuple[Phrase, ...]
    root_pc: int
    root_score: float
    root_margin: float
    global_tuning_cents: float
    pc_duration: np.ndarray
    feature: np.ndarray | None = None


@dataclass
class ClusterStats:
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
    cadence_degree_file_support: Counter = field(default_factory=Counter)
    cadence_pattern_count: dict[int, Counter] = field(
        default_factory=lambda: {2: Counter(), 3: Counter(), 4: Counter()}
    )
    cadence_pattern_file_support: dict[int, Counter] = field(
        default_factory=lambda: {2: Counter(), 3: Counter(), 4: Counter()}
    )

    duration_ratios: list[float] = field(default_factory=list)
    degree_duration_ratios: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    phrase_end_duration_ratios: list[float] = field(default_factory=list)

    tuning_cents_by_degree: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))


def note_name(pc: int) -> str:
    return NOTE_NAMES[int(pc) % 12]


def normalize_id(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in "-_ ":
            out.append("_")
    value = "".join(out)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_") or "style"


def recursive_wavs(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(p for p in folder.glob(pattern) if p.is_file())


def cents_residual(midi_value: float) -> float:
    nearest = round(float(midi_value))
    cents = (float(midi_value) - nearest) * 100.0
    while cents >= 50.0:
        cents -= 100.0
    while cents < -50.0:
        cents += 100.0
    return cents


def robust_global_tuning(events: Iterable[NoteEvent]) -> float:
    values = [cents_residual(e.source_midi) for e in events]
    return float(np.median(values)) if values else 0.0


def estimate_root_generic(
    events: tuple[NoteEvent, ...],
    phrases: tuple[Phrase, ...],
) -> tuple[int, float, float, np.ndarray]:
    """
    Estimate tonic/root without assuming a scale.

    Folk/traditional melodies often expose tonic through:
    - the final note,
    - phrase-ending notes,
    - long-duration pitch-class occupancy,
    - fifth support.

    This is deliberately generic rather than major/minor-specific.
    """
    pc_duration = np.zeros(12, dtype=np.float64)
    pc_count = np.zeros(12, dtype=np.float64)

    for event in events:
        pc = int(round(event.source_midi)) % 12
        weight = max(1.0, float(event.frames))
        pc_duration[pc] += weight
        pc_count[pc] += 1.0

    dur_total = max(float(np.sum(pc_duration)), EPS)
    count_total = max(float(np.sum(pc_count)), EPS)
    dur_share = pc_duration / dur_total
    count_share = pc_count / count_total

    ending_counts = np.zeros(12, dtype=np.float64)
    for phrase in phrases:
        if phrase.events:
            ending_counts[int(round(phrase.events[-1].source_midi)) % 12] += 1.0
    ending_share = ending_counts / max(float(np.sum(ending_counts)), 1.0)

    final_pc = (
        int(round(phrases[-1].events[-1].source_midi)) % 12
        if phrases and phrases[-1].events
        else int(round(events[-1].source_midi)) % 12
    )

    scores = np.zeros(12, dtype=np.float64)
    for root in range(12):
        fifth = (root + 7) % 12
        fourth = (root + 5) % 12
        scores[root] = (
            2.20 * ending_share[root]
            + 1.25 * (1.0 if root == final_pc else 0.0)
            + 1.00 * dur_share[root]
            + 0.30 * count_share[root]
            + 0.30 * dur_share[fifth]
            + 0.12 * dur_share[fourth]
        )

    order = np.argsort(scores)[::-1]
    best = int(order[0])
    second = float(scores[order[1]]) if len(order) > 1 else 0.0
    margin = float(scores[best] - second)
    return best, float(scores[best]), margin, pc_duration


def sequence_ngrams(seq: list[int], n: int):
    for i in range(0, len(seq) - n + 1):
        yield tuple(seq[i:i + n])


def analyze_wav(
    path: Path,
    grammar: GrammarProfile,
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

    retrigger_frames = max(
        1,
        int(round(onset_retrigger_min_ms / 1000.0 * sr / hop_length)),
    )

    events = extract_note_events(
        source_midi,
        grammar,
        onset_frames=onset_frames,
        onset_retrigger_min_frames=retrigger_frames,
    )

    if len(events) < 3:
        return None

    phrases = detect_phrases(events, sr, hop_length, grammar)
    if not phrases:
        return None

    root_pc, root_score, root_margin, pc_duration = estimate_root_generic(events, phrases)

    return FileAnalysis(
        path=path,
        sr=sr,
        events=events,
        phrases=phrases,
        root_pc=root_pc,
        root_score=root_score,
        root_margin=root_margin,
        global_tuning_cents=robust_global_tuning(events),
        pc_duration=pc_duration,
    )


# ---------------------------------------------------------------------------
# Feature extraction and NumPy-only clustering
# ---------------------------------------------------------------------------

def normalized_hist(counter: Counter, keys: Iterable) -> np.ndarray:
    values = np.asarray([float(counter.get(k, 0.0)) for k in keys], dtype=np.float64)
    total = float(np.sum(values))
    return values / total if total > 0 else values


def file_feature(file: FileAnalysis, hop_length: int) -> np.ndarray:
    root = file.root_pc

    degree_duration = Counter()
    cadence = Counter()
    abs_intervals = Counter()
    signed_intervals = Counter()
    rhythm_bins = Counter()

    for phrase in file.phrases:
        if not phrase.events:
            continue

        durations = np.asarray(
            [e.frames * hop_length / file.sr for e in phrase.events],
            dtype=np.float64,
        )
        median = max(float(np.median(durations)), 1e-6)

        for event, duration in zip(phrase.events, durations):
            degree = degree_of_midi(event.source_midi, root)
            degree_duration[degree] += float(duration)

            ratio = float(np.clip(duration / median, 0.20, 4.5))
            idx = int(np.argmin(np.abs(np.log(RHYTHM_GRID) - math.log(ratio))))
            rhythm_bins[idx] += 1

        cadence[degree_of_midi(phrase.events[-1].source_midi, root)] += 1

        for prev, curr in zip(phrase.events, phrase.events[1:]):
            delta = float(curr.source_midi - prev.source_midi)
            semis = min(12, int(round(abs(delta))))
            if semis == 0:
                continue
            abs_intervals[semis] += 1
            signed_key = semis if delta > 0 else -semis
            signed_intervals[signed_key] += 1

    blocks = [
        normalized_hist(degree_duration, range(12)) * 2.5,
        normalized_hist(abs_intervals, range(1, 13)) * 1.4,
        normalized_hist(signed_intervals, list(range(-12, 0)) + list(range(1, 13))) * 1.15,
        normalized_hist(cadence, range(12)) * 1.6,
        normalized_hist(rhythm_bins, range(len(RHYTHM_GRID))) * 0.65,
    ]
    feature = np.concatenate(blocks).astype(np.float64)

    norm = float(np.linalg.norm(feature))
    if norm > 0:
        feature /= norm
    return feature


def squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def kmeans_once(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(X)

    # k-means++ initialization.
    first = int(rng.integers(0, n))
    centers = [X[first].copy()]
    for _ in range(1, k):
        d2 = np.min(squared_distances(X, np.asarray(centers)), axis=1)
        total = float(np.sum(d2))
        if total <= EPS:
            idx = int(rng.integers(0, n))
        else:
            probs = d2 / total
            idx = int(rng.choice(n, p=probs))
        centers.append(X[idx].copy())

    centers = np.asarray(centers)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        d2 = squared_distances(X, centers)
        new_labels = np.argmin(d2, axis=1)

        new_centers = centers.copy()
        for j in range(k):
            members = X[new_labels == j]
            if len(members):
                new_centers[j] = np.mean(members, axis=0)
            else:
                # Re-seed empty cluster at the point with largest current error.
                farthest = int(np.argmax(np.min(d2, axis=1)))
                new_centers[j] = X[farthest]

        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break

        labels = new_labels
        centers = new_centers

    inertia = float(np.sum(np.min(squared_distances(X, centers), axis=1)))
    return labels, centers, inertia


def kmeans(
    X: np.ndarray,
    k: int,
    seed: int,
    n_init: int = 24,
) -> tuple[np.ndarray, np.ndarray, float]:
    best = None
    for i in range(n_init):
        rng = np.random.default_rng(seed + 104729 * i + 17 * k)
        result = kmeans_once(X, k, rng)
        if best is None or result[2] < best[2]:
            best = result
    assert best is not None
    return best


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    n = len(X)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= n:
        return -1.0

    distances = np.sqrt(np.maximum(
        0.0,
        np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    ))

    scores = []
    for i in range(n):
        own = labels[i]
        same_idx = np.flatnonzero(labels == own)
        same_idx = same_idx[same_idx != i]
        a = float(np.mean(distances[i, same_idx])) if len(same_idx) else 0.0

        b = math.inf
        for other in unique:
            if other == own:
                continue
            idx = np.flatnonzero(labels == other)
            if len(idx):
                b = min(b, float(np.mean(distances[i, idx])))

        denom = max(a, b)
        scores.append((b - a) / denom if denom > EPS and math.isfinite(b) else 0.0)

    return float(np.mean(scores))


def relabel_clusters(labels: np.ndarray) -> np.ndarray:
    """Stable labels: largest cluster first, then lowest original label."""
    counts = Counter(int(x) for x in labels)
    order = sorted(counts, key=lambda x: (-counts[x], x))
    mapping = {old: new for new, old in enumerate(order)}
    return np.asarray([mapping[int(x)] for x in labels], dtype=np.int64)


def choose_clusters(
    X: np.ndarray,
    requested: str,
    min_cluster_files: int,
    max_clusters: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    n = len(X)

    if requested != "auto":
        k = int(requested)
        if k < 1 or k > n:
            raise ValueError(f"--clusters must be between 1 and {n}")
        if k == 1:
            return np.zeros(n, dtype=np.int64), {
                "selected_k": 1,
                "silhouette": None,
                "candidates": [],
            }

        labels, _, inertia = kmeans(X, k, seed)
        labels = relabel_clusters(labels)
        sizes = sorted(Counter(labels).values())
        score = silhouette_score(X, labels)
        return labels, {
            "selected_k": k,
            "silhouette": round(score, 5),
            "inertia": round(inertia, 6),
            "cluster_sizes": sizes,
            "candidates": [],
        }

    upper = min(max_clusters, max(1, n // max(1, min_cluster_files)))
    candidates = []
    best = None

    for k in range(2, upper + 1):
        labels, _, inertia = kmeans(X, k, seed)
        labels = relabel_clusters(labels)
        counts = Counter(labels)
        min_size = min(counts.values())

        if min_size < min_cluster_files:
            candidates.append({
                "k": k,
                "rejected": "small_cluster",
                "cluster_sizes": sorted(counts.values()),
            })
            continue

        score = silhouette_score(X, labels)
        candidates.append({
            "k": k,
            "silhouette": round(score, 5),
            "inertia": round(inertia, 6),
            "cluster_sizes": sorted(counts.values()),
        })

        # Small complexity penalty prevents weak silhouette gains from
        # fragmenting a small corpus into too many clusters.
        adjusted = score - 0.015 * max(0, k - 2)
        if best is None or adjusted > best[0]:
            best = (adjusted, score, labels, inertia, k)

    # Require some evidence that a multi-cluster solution is meaningful.
    if best is None or best[1] < 0.08:
        return np.zeros(n, dtype=np.int64), {
            "selected_k": 1,
            "silhouette": None,
            "candidates": candidates,
            "reason": "no stable multi-cluster solution",
        }

    labels = relabel_clusters(best[2])
    return labels, {
        "selected_k": int(best[4]),
        "silhouette": round(float(best[1]), 5),
        "inertia": round(float(best[3]), 6),
        "cluster_sizes": sorted(Counter(labels).values()),
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Scale inference
# ---------------------------------------------------------------------------

def infer_core_scale(
    files: list[FileAnalysis],
    min_notes: int,
    max_notes: int,
    core_min_coverage: float,
    complexity_penalty: float,
    auxiliary_min_share: float,
) -> tuple[tuple[int, ...], float, dict[int, float], dict[int, float], dict]:
    """
    Infer a compact core scale and lower-frequency auxiliary degrees.

    The old trainer kept adding scale notes until a hard coverage target was
    crossed. That can turn a clear five-note core into a six-note "scale" merely
    because the fifth note explains 93.4% instead of a 94% threshold.

    v10.1 instead evaluates every size in [min_notes, max_notes]:

        score(n) = cutoff_gap(n) - complexity_penalty * (n - min_notes)

    where cutoff_gap is the occupancy-share drop between the last selected
    non-tonic core degree and the next excluded degree. Candidate sizes must
    already explain at least core_min_coverage. This favors a compact scale when
    a natural occupancy elbow exists.

    Tonic degree 0 is always retained. Excluded degrees whose corpus duration
    share exceeds auxiliary_min_share are reported separately as auxiliary
    degrees; they are NOT inserted into melodic grammar.
    """
    duration = Counter()

    for file in files:
        for event in file.events:
            degree = degree_of_midi(event.source_midi, file.root_pc)
            duration[degree] += float(event.frames)

    total = float(sum(duration.values()))
    if total <= 0:
        default = (0, 2, 4, 5, 7)
        return default, 0.0, {}, {}, {
            "selection_reason": "empty_duration_fallback",
            "candidates": [],
        }

    shares = {d: float(duration[d] / total) for d in range(12)}
    ranked = sorted(
        [d for d in range(12) if d != 0],
        key=lambda d: (-shares[d], d),
    )

    candidates = []
    eligible = []

    for n in range(min_notes, max_notes + 1):
        selected_nonzero = ranked[: max(0, n - 1)]
        selected = [0, *selected_nonzero]
        coverage = float(sum(shares[d] for d in selected))

        if selected_nonzero:
            cutoff = float(shares[selected_nonzero[-1]])
        else:
            cutoff = float(shares[0])

        next_share = (
            float(shares[ranked[len(selected_nonzero)]])
            if len(selected_nonzero) < len(ranked)
            else 0.0
        )
        gap = max(0.0, cutoff - next_share)
        score = gap - complexity_penalty * max(0, n - min_notes)

        item = {
            "notes": n,
            "scale": sorted(selected),
            "coverage": round(coverage, 6),
            "cutoff_share": round(cutoff, 6),
            "next_share": round(next_share, 6),
            "gap": round(gap, 6),
            "score": round(score, 6),
            "eligible": coverage >= core_min_coverage,
        }
        candidates.append(item)

        if coverage >= core_min_coverage:
            eligible.append((score, -n, coverage, tuple(sorted(selected))))

    if eligible:
        # Highest elbow score wins; ties prefer the smaller scale.
        eligible.sort(reverse=True)
        _, _, coverage, core = eligible[0]
        selection_reason = "coverage_elbow"
    else:
        # If no candidate reaches the minimum coverage, use max_notes rather
        # than inventing chromatic degrees beyond the configured scale size.
        selected = [0, *ranked[: max(0, max_notes - 1)]]
        core = tuple(sorted(selected))
        coverage = float(sum(shares[d] for d in core))
        selection_reason = "max_notes_fallback"

    core_set = set(core)
    auxiliary = {
        d: float(shares[d])
        for d in range(12)
        if d not in core_set and shares[d] >= auxiliary_min_share
    }

    meta = {
        "selection_reason": selection_reason,
        "core_min_coverage": core_min_coverage,
        "complexity_penalty": complexity_penalty,
        "auxiliary_min_share": auxiliary_min_share,
        "candidates": candidates,
    }
    return core, float(coverage), shares, auxiliary, meta


def estimate_root_with_scale(
    file: FileAnalysis,
    scale: tuple[int, ...],
) -> tuple[int, float, float, float]:
    """
    Re-estimate a file's tonic using the scale inferred for its current cluster.

    Coverage dominates, but final-note / phrase-ending / tonic occupancy keep
    modal rotations with identical pitch collections from collapsing blindly
    onto whichever transposition maximizes scale membership.
    """
    pc_duration = np.zeros(12, dtype=np.float64)
    pc_count = np.zeros(12, dtype=np.float64)

    for event in file.events:
        pc = int(round(event.source_midi)) % 12
        weight = max(1.0, float(event.frames))
        pc_duration[pc] += weight
        pc_count[pc] += 1.0

    dur_total = max(float(np.sum(pc_duration)), EPS)
    count_total = max(float(np.sum(pc_count)), EPS)
    dur_share = pc_duration / dur_total
    count_share = pc_count / count_total

    ending_counts = np.zeros(12, dtype=np.float64)
    for phrase in file.phrases:
        if phrase.events:
            ending_counts[int(round(phrase.events[-1].source_midi)) % 12] += 1.0
    ending_share = ending_counts / max(float(np.sum(ending_counts)), 1.0)

    final_pc = (
        int(round(file.phrases[-1].events[-1].source_midi)) % 12
        if file.phrases and file.phrases[-1].events
        else int(round(file.events[-1].source_midi)) % 12
    )

    scores = np.zeros(12, dtype=np.float64)
    coverages = np.zeros(12, dtype=np.float64)
    scale_set = set(int(d) % 12 for d in scale)

    for root in range(12):
        allowed = {(root + d) % 12 for d in scale_set}
        coverage = float(sum(pc_duration[pc] for pc in allowed) / dur_total)
        coverages[root] = coverage

        tonic_share = float(dur_share[root])
        tonic_count = float(count_share[root])
        ending = float(ending_share[root])

        # Only award fifth/fourth support if they are actual core-scale degrees.
        fifth_support = float(dur_share[(root + 7) % 12]) if 7 in scale_set else 0.0
        fourth_support = float(dur_share[(root + 5) % 12]) if 5 in scale_set else 0.0

        scores[root] = (
            4.00 * coverage
            + 1.80 * ending
            + 0.95 * (1.0 if root == final_pc else 0.0)
            + 0.75 * tonic_share
            + 0.18 * tonic_count
            + 0.22 * fifth_support
            + 0.10 * fourth_support
        )

    order = np.argsort(scores)[::-1]
    best = int(order[0])
    second = float(scores[order[1]]) if len(order) > 1 else 0.0
    margin = float(scores[best] - second)
    return best, float(scores[best]), margin, float(coverages[best])


def align_labels_to_previous(
    previous: np.ndarray,
    current: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Align new k-means labels to the previous iteration by maximum assignment
    overlap. k <= 5 by default, so brute-force permutation is tiny and avoids
    false non-convergence from arbitrary cluster-number swaps.
    """
    if k <= 1:
        return current.copy()

    best_score = -1
    best_mapping = None
    for perm in permutations(range(k)):
        mapped = np.asarray([perm[int(x)] for x in current], dtype=np.int64)
        score = int(np.sum(mapped == previous))
        if score > best_score:
            best_score = score
            best_mapping = perm

    assert best_mapping is not None
    return np.asarray([best_mapping[int(x)] for x in current], dtype=np.int64)


def refine_roots_and_clusters(
    analyses: list[FileAnalysis],
    labels: np.ndarray,
    hop_length: int,
    seed: int,
    iterations: int,
    scale_min_notes: int,
    scale_max_notes: int,
    core_min_coverage: float,
    scale_complexity_penalty: float,
    auxiliary_min_share: float,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """
    Alternating refinement:
        roots -> file features -> fixed-K clustering -> cluster core scales -> roots

    The initially selected K is held fixed. This makes "how many styles?" a
    model-selection step, while refinement solves tonic/cluster consistency.
    """
    k = int(np.max(labels)) + 1
    history = []

    for iteration in range(1, max(1, iterations) + 1):
        old_labels = labels.copy()
        old_roots = np.asarray([f.root_pc for f in analyses], dtype=np.int64)

        cluster_models = []
        for cluster_idx in range(k):
            files = [
                f for f, lab in zip(analyses, labels)
                if int(lab) == cluster_idx
            ]
            core, coverage, shares, auxiliary, scale_meta = infer_core_scale(
                files=files,
                min_notes=scale_min_notes,
                max_notes=scale_max_notes,
                core_min_coverage=core_min_coverage,
                complexity_penalty=scale_complexity_penalty,
                auxiliary_min_share=auxiliary_min_share,
            )
            cluster_models.append({
                "cluster": cluster_idx,
                "core_scale": core,
                "coverage": coverage,
                "shares": shares,
                "auxiliary": auxiliary,
                "scale_meta": scale_meta,
            })

        root_changes = 0
        root_coverages = []
        for file, label_idx in zip(analyses, labels):
            model = cluster_models[int(label_idx)]
            root, score, margin, coverage = estimate_root_with_scale(
                file,
                model["core_scale"],
            )
            if root != file.root_pc:
                root_changes += 1
            file.root_pc = root
            file.root_score = score
            file.root_margin = margin
            root_coverages.append(coverage)
            file.feature = file_feature(file, hop_length)

        X = np.stack([f.feature for f in analyses], axis=0)

        if k > 1:
            new_labels, _, inertia = kmeans(X, k, seed + iteration * 7919)
            new_labels = align_labels_to_previous(old_labels, new_labels, k)
            labels = new_labels
            sil = silhouette_score(X, labels)
        else:
            inertia = 0.0
            sil = None
            labels = np.zeros(len(analyses), dtype=np.int64)

        label_changes = int(np.sum(labels != old_labels))
        roots_now = np.asarray([f.root_pc for f in analyses], dtype=np.int64)

        history.append({
            "iteration": iteration,
            "root_changes": int(root_changes),
            "label_changes": label_changes,
            "mean_root_scale_coverage": round(float(np.mean(root_coverages)), 6),
            "silhouette": round(float(sil), 6) if sil is not None else None,
            "inertia": round(float(inertia), 6),
            "cluster_scales": [
                {
                    "cluster": int(m["cluster"]) + 1,
                    "core_scale": list(m["core_scale"]),
                    "coverage": round(float(m["coverage"]), 6),
                    "auxiliary_degrees": {
                        str(d): round(float(v), 6)
                        for d, v in sorted(m["auxiliary"].items())
                    },
                }
                for m in cluster_models
            ],
        })

        if label_changes == 0 and np.array_equal(roots_now, old_roots):
            break

    # Recompute final models after the last assignments/root update.
    final_models = []
    for cluster_idx in range(k):
        files = [
            f for f, lab in zip(analyses, labels)
            if int(lab) == cluster_idx
        ]
        core, coverage, shares, auxiliary, scale_meta = infer_core_scale(
            files=files,
            min_notes=scale_min_notes,
            max_notes=scale_max_notes,
            core_min_coverage=core_min_coverage,
            complexity_penalty=scale_complexity_penalty,
            auxiliary_min_share=auxiliary_min_share,
        )
        final_models.append({
            "cluster": cluster_idx,
            "core_scale": core,
            "coverage": coverage,
            "shares": shares,
            "auxiliary": auxiliary,
            "scale_meta": scale_meta,
        })

    return labels, final_models, history


# ---------------------------------------------------------------------------
# Grammar statistics with the three tuner fixes:
#   1) scale-outside degrees are excluded and BREAK runs,
#   2) same-degree repetitions are collapsed for melodic grammar,
#   3) cadence requires count + file support.
# ---------------------------------------------------------------------------

def phrase_scale_runs(
    phrase: Phrase,
    root_pc: int,
    scale_set: set[int],
) -> list[list[tuple[NoteEvent, int]]]:
    runs: list[list[tuple[NoteEvent, int]]] = []
    current: list[tuple[NoteEvent, int]] = []

    for event in phrase.events:
        degree = degree_of_midi(event.source_midi, root_pc)

        if degree not in scale_set:
            if current:
                runs.append(current)
                current = []
            continue

        # Same-degree re-attacks belong to rhythm/articulation, not melodic
        # transition grammar. Collapse them here only.
        if current and current[-1][1] == degree:
            continue

        current.append((event, degree))

    if current:
        runs.append(current)

    return runs


def accumulate_cluster(
    files: list[FileAnalysis],
    scale: tuple[int, ...],
    hop_length: int,
) -> ClusterStats:
    stats = ClusterStats(files=list(files))
    scale_set = set(scale)

    for file in files:
        phrase_support = {2: set(), 3: set(), 4: set(), 5: set()}
        cadence_degree_seen = set()
        cadence_pattern_seen = {2: set(), 3: set(), 4: set()}

        for phrase in file.phrases:
            if not phrase.events:
                continue

            durations = np.asarray(
                [e.frames * hop_length / file.sr for e in phrase.events],
                dtype=np.float64,
            )
            phrase_median = max(float(np.median(durations)), 1e-6)

            # Rhythm and degree occupancy use every IN-SCALE note, including
            # repeated equal notes.
            for event, duration in zip(phrase.events, durations):
                degree = degree_of_midi(event.source_midi, file.root_pc)
                if degree not in scale_set:
                    continue

                stats.degree_count[degree] += 1
                stats.degree_duration[degree] += float(duration)

                ratio = float(duration / phrase_median)
                stats.duration_ratios.append(ratio)
                stats.degree_duration_ratios[degree].append(ratio)

                cents = cents_residual(event.source_midi) - file.global_tuning_cents
                while cents >= 50:
                    cents -= 100
                while cents < -50:
                    cents += 100
                stats.tuning_cents_by_degree[degree].append(float(cents))

            final_degree = degree_of_midi(phrase.events[-1].source_midi, file.root_pc)
            if final_degree in scale_set:
                stats.phrase_end_duration_ratios.append(float(durations[-1] / phrase_median))
                stats.cadence_degree_count[final_degree] += 1
                cadence_degree_seen.add(final_degree)

            runs = phrase_scale_runs(phrase, file.root_pc, scale_set)

            for run in runs:
                if not run:
                    continue
                degrees = [degree for _, degree in run]

                for (prev_event, a), (event, b) in zip(run, run[1:]):
                    delta = float(event.source_midi - prev_event.source_midi)
                    interval = min(12, int(round(abs(delta))))
                    if interval <= 0:
                        continue

                    stats.interval_count[interval] += 1
                    stats.transition_count[(a, b)] += 1
                    if delta > 0.35:
                        stats.ascending_transition_count[(a, b)] += 1
                    elif delta < -0.35:
                        stats.descending_transition_count[(a, b)] += 1

                for tri in sequence_ngrams(degrees, 3):
                    stats.trigram_count[tri] += 1

                for n in (2, 3, 4, 5):
                    for gram in sequence_ngrams(degrees, n):
                        stats.ngram_count[n][gram] += 1
                        phrase_support[n].add(gram)

            # A cadence suffix only counts if the phrase ends inside the scale
            # and the suffix itself is one uninterrupted in-scale run.
            if final_degree in scale_set and runs:
                last_run_degrees = [d for _, d in runs[-1]]
                # Require the final run to actually include the phrase's final
                # note degree; otherwise an out-of-scale final note broke it.
                if last_run_degrees and last_run_degrees[-1] == final_degree:
                    for n in (2, 3, 4):
                        if len(last_run_degrees) >= n:
                            pattern = tuple(last_run_degrees[-n:])
                            stats.cadence_pattern_count[n][pattern] += 1
                            cadence_pattern_seen[n].add(pattern)

        for n in (2, 3, 4, 5):
            for gram in phrase_support[n]:
                stats.ngram_file_support[n][gram] += 1

        for degree in cadence_degree_seen:
            stats.cadence_degree_file_support[degree] += 1

        for n in (2, 3, 4):
            for pattern in cadence_pattern_seen[n]:
                stats.cadence_pattern_file_support[n][pattern] += 1

    return stats


def smoothed_log_lift(observed: float, expected: float, alpha: float = 0.5) -> float:
    return math.log((observed + alpha) / (expected + alpha))


def positive_weight(
    raw: float,
    scale: float,
    max_weight: float,
    minimum: float = 0.05,
) -> float | None:
    value = max(0.0, float(raw)) * float(scale)
    value = min(float(max_weight), value)
    if value < minimum:
        return None
    return round(value, 4)


def transition_pmi_weights(
    counts: Counter,
    min_count: int,
    weight_scale: float,
    max_weight: float,
    top_k: int,
) -> dict[str, float]:
    total = int(sum(counts.values()))
    if total <= 0:
        return {}

    src = Counter()
    dst = Counter()
    for (a, b), count in counts.items():
        if a == b:
            continue
        src[a] += count
        dst[b] += count

    scored = []
    for (a, b), count in counts.items():
        if a == b or count < min_count:
            continue
        expected = total * (src[a] / total) * (dst[b] / total)
        raw = smoothed_log_lift(count, expected)
        weight = positive_weight(raw, weight_scale, max_weight)
        if weight is not None:
            scored.append((f"{a}>{b}", weight, count))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return {key: weight for key, weight, _ in scored[:top_k]}


def ngram_weight(
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

    p = 1.0
    for degree in pattern:
        p *= max(degree_counts[degree] / total_degrees, EPS)

    expected = total_windows * p
    raw = smoothed_log_lift(count, expected)
    return positive_weight(raw, weight_scale, max_weight)


def preferred_duration_ratios(values: list[float], max_bins: int = 6) -> list[float]:
    if not values:
        return [0.5, 1.0, 2.0]

    counts = Counter()
    for value in values:
        value = float(np.clip(value, 0.20, 4.5))
        idx = int(np.argmin(np.abs(np.log(RHYTHM_GRID) - math.log(value))))
        counts[float(RHYTHM_GRID[idx])] += 1

    chosen = [r for r, _ in counts.most_common(max_bins)]
    if 1.0 not in chosen:
        chosen.append(1.0)
    return [round(float(x), 4) for x in sorted(set(chosen))]


def learn_profile(
    stats: ClusterStats,
    scale: tuple[int, ...],
    style_id: str,
    label: str,
    region: str,
    description: str,
    scale_coverage: float,
    auxiliary_degrees: dict[int, float],
    scale_selection_meta: dict,
    cluster_index: int,
    cluster_count: int,
    min_count: int,
    min_file_support: int,
    max_weight: float,
    top_intervals: int,
    top_transitions: int,
    top_trigrams: int,
    top_phrases: int,
    top_cadences: int,
    learn_microtuning: bool,
) -> dict:
    grammar = {
        "pitch_deviation_weight": DEFAULT_GRAMMAR.pitch_deviation_weight,
        "motion_preservation_weight": DEFAULT_GRAMMAR.motion_preservation_weight,
        "contour_penalty": DEFAULT_GRAMMAR.contour_penalty,
        "leap_penalty": DEFAULT_GRAMMAR.leap_penalty,
        "max_preferred_leap": DEFAULT_GRAMMAR.max_preferred_leap,
        "candidate_shift_semitones": DEFAULT_GRAMMAR.candidate_shift_semitones,
        "candidate_count": min(9, max(5, len(scale) + 2)),
        "event_pitch_change": DEFAULT_GRAMMAR.event_pitch_change,
        "min_event_frames": DEFAULT_GRAMMAR.min_event_frames,
        "phrase_gap_ms": DEFAULT_GRAMMAR.phrase_gap_ms,
        "phrase_long_note_factor": DEFAULT_GRAMMAR.phrase_long_note_factor,
        "phrase_leap_semitones": DEFAULT_GRAMMAR.phrase_leap_semitones,
        "min_phrase_events": DEFAULT_GRAMMAR.min_phrase_events,
    }

    total_degree_duration = float(sum(stats.degree_duration.values()))
    total_degree_count = int(sum(stats.degree_count.values()))
    scale_size = max(1, len(scale))

    degree_weights = {}
    for degree in scale:
        observed = float(stats.degree_duration.get(degree, 0.0))
        if observed <= 0 or total_degree_duration <= 0:
            continue
        p = observed / total_degree_duration
        raw = math.log((p + 1e-6) / (1.0 / scale_size))
        weight = positive_weight(raw, 1.45, max_weight)
        if weight is not None:
            degree_weights[str(degree)] = weight
    grammar["degree_weights"] = degree_weights

    interval_total = int(sum(stats.interval_count.values()))
    interval_weights = {}
    if interval_total > 0:
        expected = interval_total / 12.0
        scored = []
        for interval, count in stats.interval_count.items():
            if count < min_count:
                continue
            weight = positive_weight(
                smoothed_log_lift(count, expected),
                1.25,
                max_weight,
            )
            if weight is not None:
                scored.append((interval, weight, count))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        interval_weights = {
            str(interval): weight
            for interval, weight, _ in scored[:top_intervals]
        }
    grammar["interval_weights"] = interval_weights

    grammar["transition_weights"] = transition_pmi_weights(
        stats.transition_count, min_count, 1.10, max_weight, top_transitions
    )
    grammar["ascending_transition_weights"] = transition_pmi_weights(
        stats.ascending_transition_count, min_count, 1.15, max_weight, top_transitions
    )
    grammar["descending_transition_weights"] = transition_pmi_weights(
        stats.descending_transition_count, min_count, 1.15, max_weight, top_transitions
    )

    trigram_total = int(sum(stats.trigram_count.values()))
    trigram_scored = []
    for pattern, count in stats.trigram_count.items():
        if count < min_count:
            continue
        weight = ngram_weight(
            pattern, count, stats.degree_count, total_degree_count,
            trigram_total, 0.72, max_weight
        )
        if weight is not None:
            trigram_scored.append((">".join(map(str, pattern)), weight, count))
    trigram_scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    grammar["trigram_weights"] = {
        key: weight for key, weight, _ in trigram_scored[:top_trigrams]
    }

    # Preferred phrases: 4/5 note patterns, count + file support.
    actual_support = min(max(1, min_file_support), max(1, len(stats.files)))
    phrase_scored = []
    for n in (4, 5):
        total_windows = int(sum(stats.ngram_count[n].values()))
        for pattern, count in stats.ngram_count[n].items():
            support = int(stats.ngram_file_support[n][pattern])
            if count < min_count or support < actual_support:
                continue
            weight = ngram_weight(
                pattern, count, stats.degree_count, total_degree_count,
                total_windows, 0.62, max_weight
            )
            if weight is None:
                continue
            weight = round(min(max_weight, weight + min(0.45, 0.08 * max(0, support - 1))), 4)
            phrase_scored.append((pattern, weight, count, support))
    phrase_scored.sort(key=lambda x: (x[1], x[3], x[2]), reverse=True)
    grammar["preferred_phrases"] = [
        {
            "degrees": list(pattern),
            "weight": weight,
            "support_count": count,
            "support_files": support,
        }
        for pattern, weight, count, support in phrase_scored[:top_phrases]
    ]

    # Cadence degree: both count and cross-file support.
    phrase_count = int(sum(stats.cadence_degree_count.values()))
    cadence_degrees = {}
    if phrase_count > 0 and total_degree_count > 0:
        for degree, end_count in stats.cadence_degree_count.items():
            support = int(stats.cadence_degree_file_support[degree])
            if end_count < min_count or support < actual_support:
                continue
            p_end = end_count / phrase_count
            p_all = stats.degree_count[degree] / max(total_degree_count, 1)
            weight = positive_weight(
                math.log((p_end + 1e-6) / (p_all + 1e-6)),
                1.15,
                max_weight,
            )
            if weight is not None:
                cadence_degrees[str(degree)] = weight
    grammar["cadence_degrees"] = cadence_degrees

    cadence_scored = []
    for n in (2, 3, 4):
        total_windows = int(sum(stats.ngram_count[n].values()))
        if total_windows <= 0 or phrase_count <= 0:
            continue
        for pattern, suffix_count in stats.cadence_pattern_count[n].items():
            support = int(stats.cadence_pattern_file_support[n][pattern])
            if suffix_count < min_count or support < actual_support:
                continue
            global_count = stats.ngram_count[n][pattern]
            p_suffix = suffix_count / phrase_count
            p_global = global_count / total_windows
            weight = positive_weight(
                math.log((p_suffix + 1e-6) / (p_global + 1e-6)),
                0.95,
                max_weight,
            )
            if weight is not None:
                cadence_scored.append((pattern, weight, suffix_count, support))
    cadence_scored.sort(key=lambda x: (x[1], x[3], x[2]), reverse=True)
    grammar["cadence_patterns"] = [
        {
            "degrees": list(pattern),
            "weight": weight,
            "support_count": count,
            "support_files": support,
        }
        for pattern, weight, count, support in cadence_scored[:top_cadences]
    ]

    rhythm = {
        "enabled": True,
        "preserve_phrase_duration": True,
        "quantize_strength": 0.20,
        "preferred_duration_ratios": preferred_duration_ratios(stats.duration_ratios),
        "degree_duration_multipliers": {},
        "phrase_end_multiplier": 1.0,
        "gap_multiplier": 1.0,
        "max_duration_change": 0.30,
    }

    multipliers = {}
    for degree, values in stats.degree_duration_ratios.items():
        if degree not in set(scale) or len(values) < min_count:
            continue
        median = float(np.median(values))
        value = float(np.clip(median, 0.70, 1.50))
        if abs(value - 1.0) >= 0.04:
            multipliers[str(degree)] = round(value, 4)
    rhythm["degree_duration_multipliers"] = multipliers

    if stats.phrase_end_duration_ratios:
        rhythm["phrase_end_multiplier"] = round(
            float(np.clip(np.median(stats.phrase_end_duration_ratios), 0.80, 1.80)),
            4,
        )

    tuning = {"degree_cents": {}}
    if learn_microtuning:
        cents_out = {}
        min_tuning = max(5, min_count * 2)
        for degree, values in stats.tuning_cents_by_degree.items():
            if degree not in set(scale) or len(values) < min_tuning:
                continue
            arr = np.asarray(values, dtype=np.float64)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            if mad > 0:
                keep = np.abs(arr - med) <= max(8.0, 3.5 * mad)
                arr = arr[keep]
            if len(arr) < min_tuning:
                continue
            cents = float(np.clip(np.median(arr), -35.0, 35.0))
            if abs(cents) >= 2.0:
                cents_out[str(degree)] = round(cents, 2)
        tuning["degree_cents"] = cents_out

    roots = Counter(note_name(f.root_pc) for f in stats.files)

    profile = {
        "id": style_id,
        "label": label,
        "region": region,
        "description": description,
        "scale": list(scale),
        "auxiliary_degrees": {
            str(d): round(float(v), 6)
            for d, v in sorted(auxiliary_degrees.items())
        },
        "grammar": grammar,
        "rhythm": rhythm,
        "tuning": tuning,
        "ornaments": [],
        "modulation": {
            "enabled": False,
            "options": [],
        },
        "notes": (
            "Unsupervised corpus-derived v10.1 profile. Ornament/modulation rules were "
            "intentionally left empty because they were not estimated robustly."
        ),
        "training": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "method": "scaleify_unsupervised_cluster_v3",
            "cluster_index": cluster_index,
            "cluster_count": cluster_count,
            "files_analyzed": len(stats.files),
            "events": int(sum(len(f.events) for f in stats.files)),
            "phrases": int(sum(len(f.phrases) for f in stats.files)),
            "inferred_scale": list(scale),
            "inferred_scale_coverage": round(float(scale_coverage), 4),
            "auxiliary_degrees": {
                str(d): round(float(v), 6)
                for d, v in sorted(auxiliary_degrees.items())
            },
            "scale_selection": scale_selection_meta,
            "root_distribution": dict(sorted(roots.items())),
            "source_files": [f.path.name for f in stats.files],
            "fixes": [
                "out-of-scale degrees excluded from melodic grammar",
                "out-of-scale notes break melodic runs",
                "same-degree repetitions excluded from melodic transitions",
                "cadence requires count and cross-file support",
                "core scale separated from auxiliary degrees using occupancy elbow",
                "roots iteratively refined against cluster core scale",
            ],
        },
    }
    return profile


def cluster_report_entry(
    cluster_idx: int,
    files: list[FileAnalysis],
    scale: tuple[int, ...],
    coverage: float,
    shares: dict[int, float],
    auxiliary: dict[int, float],
    scale_meta: dict,
) -> dict:
    return {
        "cluster": cluster_idx + 1,
        "file_count": len(files),
        "scale": list(scale),
        "scale_coverage": round(coverage, 4),
        "auxiliary_degrees": {
            str(k): round(v, 5)
            for k, v in sorted(auxiliary.items())
        },
        "scale_selection": scale_meta,
        "degree_shares": {
            str(k): round(v, 5)
            for k, v in sorted(shares.items())
            if v >= 0.005
        },
        "files": [
            {
                "filename": f.path.name,
                "root": note_name(f.root_pc),
                "root_score": round(f.root_score, 5),
                "root_margin": round(f.root_margin, 5),
                "events": len(f.events),
                "phrases": len(f.phrases),
            }
            for f in files
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Infer one or more Scaleify style JSONs directly from a folder of WAV files. "
            "No input style JSON is required."
        )
    )
    parser.add_argument("wav_folder", type=Path)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("styles/generated"),
        help="Directory for generated JSONs/reports.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output style prefix. Default: dataset folder name.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Region label stored in JSON. Default: title-cased dataset folder name.",
    )
    parser.add_argument(
        "--clusters",
        default="auto",
        help="'auto' or an integer >=1.",
    )
    parser.add_argument("--max-clusters", type=int, default=5)
    parser.add_argument("--min-cluster-files", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1479)

    parser.add_argument("--core-min-coverage", type=float, default=0.90)
    parser.add_argument("--scale-complexity-penalty", type=float, default=0.02)
    parser.add_argument("--auxiliary-min-share", type=float, default=0.015)
    parser.add_argument("--scale-min-notes", type=int, default=5)
    parser.add_argument("--scale-max-notes", type=int, default=7)
    parser.add_argument("--root-refine-iterations", type=int, default=6)

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

    parser.add_argument("--min-count", type=int, default=4)
    parser.add_argument("--min-file-support", type=int, default=3)
    parser.add_argument("--max-weight", type=float, default=3.5)
    parser.add_argument("--top-intervals", type=int, default=8)
    parser.add_argument("--top-transitions", type=int, default=28)
    parser.add_argument("--top-trigrams", type=int, default=24)
    parser.add_argument("--top-phrases", type=int, default=16)
    parser.add_argument("--top-cadences", type=int, default=12)
    parser.add_argument("--no-microtuning", action="store_true")

    args = parser.parse_args()

    if not args.wav_folder.is_dir():
        parser.error(f"WAV folder not found: {args.wav_folder}")
    if not 0.70 <= args.core_min_coverage <= 0.999:
        parser.error("--core-min-coverage should be between 0.70 and 0.999")
    if args.scale_complexity_penalty < 0:
        parser.error("--scale-complexity-penalty must be >= 0")
    if not 0.0 <= args.auxiliary_min_share < 0.25:
        parser.error("--auxiliary-min-share should be in [0, 0.25)")
    if args.root_refine_iterations < 1:
        parser.error("--root-refine-iterations must be >= 1")
    if args.scale_min_notes < 3:
        parser.error("--scale-min-notes must be >= 3")
    if args.scale_max_notes < args.scale_min_notes or args.scale_max_notes > 12:
        parser.error("--scale-max-notes must be >= min and <= 12")
    if args.min_count < 1 or args.min_file_support < 1:
        parser.error("--min-count and --min-file-support must be >= 1")

    if args.clusters != "auto":
        try:
            int(args.clusters)
        except ValueError:
            parser.error("--clusters must be 'auto' or an integer")

    wavs = recursive_wavs(args.wav_folder, recursive=not args.no_recursive)
    if not wavs:
        parser.error(f"No WAV files found in: {args.wav_folder}")

    prefix = normalize_id(args.prefix or args.wav_folder.name)
    region = args.region or args.wav_folder.name.replace("_", " ").title()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Corpus:       {args.wav_folder}")
    print(f"WAV files:    {len(wavs)}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Clusters:     {args.clusters}")
    print()

    analyses: list[FileAnalysis] = []
    failures = []

    for i, path in enumerate(wavs, start=1):
        print(f"[{i}/{len(wavs)}] {path.name}")
        try:
            result = analyze_wav(
                path=path,
                grammar=DEFAULT_GRAMMAR,
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
            if result is None:
                failures.append((path.name, "insufficient note/phrase data"))
                print("    [skip] insufficient note/phrase data")
                continue

            result.feature = file_feature(result, args.hop_length)
            analyses.append(result)
            print(
                f"    root={note_name(result.root_pc)} "
                f"margin={result.root_margin:.3f} "
                f"events={len(result.events)} "
                f"phrases={len(result.phrases)}"
            )
        except Exception as exc:
            failures.append((path.name, f"{type(exc).__name__}: {exc}"))
            print(f"    [failed] {type(exc).__name__}: {exc}")

    if not analyses:
        raise RuntimeError("No files could be analyzed.")

    X = np.stack([f.feature for f in analyses], axis=0)

    labels, clustering_info = choose_clusters(
        X=X,
        requested=args.clusters,
        min_cluster_files=args.min_cluster_files,
        max_clusters=args.max_clusters,
        seed=args.seed,
    )

    k = int(np.max(labels)) + 1
    print()
    print(
        f"Initial clusters: {k}"
        + (
            f" | silhouette={clustering_info.get('silhouette')}"
            if clustering_info.get("silhouette") is not None
            else ""
        )
    )

    labels, final_models, refinement_history = refine_roots_and_clusters(
        analyses=analyses,
        labels=labels,
        hop_length=args.hop_length,
        seed=args.seed,
        iterations=args.root_refine_iterations,
        scale_min_notes=args.scale_min_notes,
        scale_max_notes=args.scale_max_notes,
        core_min_coverage=args.core_min_coverage,
        scale_complexity_penalty=args.scale_complexity_penalty,
        auxiliary_min_share=args.auxiliary_min_share,
    )

    k = int(np.max(labels)) + 1
    X_final = np.stack([f.feature for f in analyses], axis=0)
    final_silhouette = silhouette_score(X_final, labels) if k > 1 else None

    print(
        f"Refined clusters: {k}"
        + (
            f" | silhouette={final_silhouette:.5f}"
            if final_silhouette is not None
            else ""
        )
        + f" | iterations={len(refinement_history)}"
    )

    report_clusters = []
    output_jsons = []

    assignments_path = args.output_dir / f"{prefix}_cluster_assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "cluster", "estimated_root",
            "root_score", "root_margin", "events", "phrases",
        ])

        for file, label_idx in zip(analyses, labels):
            writer.writerow([
                file.path.name,
                int(label_idx) + 1,
                note_name(file.root_pc),
                round(file.root_score, 6),
                round(file.root_margin, 6),
                len(file.events),
                len(file.phrases),
            ])

    for cluster_idx in range(k):
        files = [f for f, label_idx in zip(analyses, labels) if int(label_idx) == cluster_idx]

        model = final_models[cluster_idx]
        scale = model["core_scale"]
        coverage = model["coverage"]
        shares = model["shares"]
        auxiliary = model["auxiliary"]
        scale_meta = model["scale_meta"]

        stats = accumulate_cluster(files, scale, args.hop_length)

        style_id = f"{prefix}_cluster_{cluster_idx + 1}"
        label = f"{region} Cluster {cluster_idx + 1}"
        description = (
            f"Unsupervised style cluster {cluster_idx + 1}/{k} inferred from "
            f"{len(files)} monophonic corpus files. Scale and melodic grammar "
            f"were learned from the cluster rather than supplied by a template."
        )

        profile = learn_profile(
            stats=stats,
            scale=scale,
            style_id=style_id,
            label=label,
            region=region,
            description=description,
            scale_coverage=coverage,
            auxiliary_degrees=auxiliary,
            scale_selection_meta=scale_meta,
            cluster_index=cluster_idx + 1,
            cluster_count=k,
            min_count=args.min_count,
            min_file_support=min(args.min_file_support, max(1, len(files))),
            max_weight=args.max_weight,
            top_intervals=args.top_intervals,
            top_transitions=args.top_transitions,
            top_trigrams=args.top_trigrams,
            top_phrases=args.top_phrases,
            top_cadences=args.top_cadences,
            learn_microtuning=not args.no_microtuning,
        )

        out_path = args.output_dir / f"{style_id}.json"
        out_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Validate that Scaleify can load the generated JSON.
        load_style_profile(out_path)

        output_jsons.append(str(out_path))
        report_clusters.append(
            cluster_report_entry(
                cluster_idx, files, scale, coverage, shares, auxiliary, scale_meta
            )
        )

        print(
            f"Cluster {cluster_idx + 1}: "
            f"files={len(files)}, core={list(scale)}, "
            f"coverage={coverage:.3f}, "
            f"aux={{{', '.join(f'{d}:{v:.3f}' for d, v in sorted(auxiliary.items()))}}} "
            f"-> {out_path.name}"
        )

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "scaleify_unsupervised_cluster_v3",
        "corpus": str(args.wav_folder),
        "files_found": len(wavs),
        "files_analyzed": len(analyses),
        "failures": [
            {"filename": name, "reason": reason}
            for name, reason in failures
        ],
        "clustering": {
            **clustering_info,
            "final_silhouette": (
                round(float(final_silhouette), 5)
                if final_silhouette is not None else None
            ),
        },
        "root_cluster_refinement": refinement_history,
        "scale_inference": {
            "core_min_coverage": args.core_min_coverage,
            "complexity_penalty": args.scale_complexity_penalty,
            "auxiliary_min_share": args.auxiliary_min_share,
            "min_notes": args.scale_min_notes,
            "max_notes": args.scale_max_notes,
        },
        "clusters": report_clusters,
        "output_jsons": output_jsons,
        "assignments_csv": str(assignments_path),
    }

    report_path = args.output_dir / f"{prefix}_cluster_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Training complete")
    print("-----------------")
    print(f"Analyzed:     {len(analyses)}/{len(wavs)}")
    print(f"Styles:       {k}")
    print(f"Assignments:  {assignments_path}")
    print(f"Report:       {report_path}")
    for path in output_jsons:
        print(f"Style JSON:   {path}")


if __name__ == "__main__":
    main()