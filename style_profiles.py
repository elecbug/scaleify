from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NoteEvent:
    start_frame: int
    end_frame: int
    source_midi: float

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class Phrase:
    events: tuple[NoteEvent, ...]
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class OrnamentRule:
    type: str
    degrees: tuple[int, ...] = ()
    probability: float = 1.0
    scale_steps: int = 0
    fraction: float = 0.12
    max_ms: float = 90.0
    depth_cents: float = 0.0
    rate_hz: float = 5.0
    delay_fraction: float = 0.10


@dataclass(frozen=True)
class RhythmProfile:
    enabled: bool = False
    preserve_phrase_duration: bool = True
    quantize_strength: float = 0.0
    preferred_duration_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    degree_duration_multipliers: dict[int, float] = field(default_factory=dict)
    phrase_end_multiplier: float = 1.0
    gap_multiplier: float = 1.0
    max_duration_change: float = 0.35


@dataclass(frozen=True)
class TuningProfile:
    degree_cents: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModulationOption:
    name: str
    root_offset: int = 0
    scale: tuple[int, ...] = ()
    min_phrase_events: int = 4
    switch_penalty: float = 1.5
    activation_bonus: float = 0.0


@dataclass(frozen=True)
class ModulationProfile:
    enabled: bool = False
    options: tuple[ModulationOption, ...] = ()


@dataclass(frozen=True)
class GrammarProfile:
    pitch_deviation_weight: float = 1.0
    motion_preservation_weight: float = 0.35
    contour_penalty: float = 1.0
    leap_penalty: float = 0.12
    max_preferred_leap: float = 7.0
    candidate_shift_semitones: float = 5.0
    candidate_count: int = 7
    event_pitch_change: float = 0.8
    min_event_frames: int = 2

    phrase_gap_ms: float = 140.0
    phrase_long_note_factor: float = 2.2
    phrase_leap_semitones: float = 7.0
    min_phrase_events: int = 3

    interval_weights: dict[int, float] = field(default_factory=dict)
    degree_weights: dict[int, float] = field(default_factory=dict)
    transition_weights: dict[str, float] = field(default_factory=dict)
    ascending_transition_weights: dict[str, float] = field(default_factory=dict)
    descending_transition_weights: dict[str, float] = field(default_factory=dict)
    trigram_weights: dict[str, float] = field(default_factory=dict)
    preferred_phrases: tuple[tuple[tuple[int, ...], float], ...] = ()
    cadence_degrees: dict[int, float] = field(default_factory=dict)
    cadence_patterns: tuple[tuple[tuple[int, ...], float], ...] = ()


@dataclass(frozen=True)
class StyleProfile:
    id: str
    label: str
    region: str
    description: str
    scale: tuple[int, ...]
    grammar: GrammarProfile
    rhythm: RhythmProfile
    tuning: TuningProfile
    ornaments: tuple[OrnamentRule, ...]
    modulation: ModulationProfile
    notes: str = ""


@dataclass(frozen=True)
class MappingResult:
    phrases: tuple[Phrase, ...]
    targets: tuple[tuple[int, ...], ...]
    roots: tuple[int, ...]
    scales: tuple[tuple[int, ...], ...]
    costs: tuple[float, ...]
    modulation_names: tuple[str, ...]


def _int_key_dict(data: dict | None) -> dict[int, float]:
    if not data:
        return {}
    return {int(k): float(v) for k, v in data.items()}


def _str_float_dict(data: dict | None) -> dict[str, float]:
    if not data:
        return {}
    return {str(k): float(v) for k, v in data.items()}


def _weighted_sequences(data: Iterable[dict] | None) -> tuple[tuple[tuple[int, ...], float], ...]:
    if not data:
        return ()
    result: list[tuple[tuple[int, ...], float]] = []
    for item in data:
        degrees = tuple(int(x) % 12 for x in item.get("degrees", []))
        if len(degrees) < 2:
            continue
        result.append((degrees, float(item.get("weight", 0.0))))
    return tuple(result)


def load_style_profile(path: Path) -> StyleProfile:
    data = json.loads(path.read_text(encoding="utf-8"))
    grammar_data = data.get("grammar", {})
    rhythm_data = data.get("rhythm", {})
    tuning_data = data.get("tuning", {})
    modulation_data = data.get("modulation", {})

    grammar = GrammarProfile(
        pitch_deviation_weight=float(grammar_data.get("pitch_deviation_weight", 1.0)),
        motion_preservation_weight=float(grammar_data.get("motion_preservation_weight", 0.35)),
        contour_penalty=float(grammar_data.get("contour_penalty", 1.0)),
        leap_penalty=float(grammar_data.get("leap_penalty", 0.12)),
        max_preferred_leap=float(grammar_data.get("max_preferred_leap", 7.0)),
        candidate_shift_semitones=float(grammar_data.get("candidate_shift_semitones", 5.0)),
        candidate_count=int(grammar_data.get("candidate_count", 7)),
        event_pitch_change=float(grammar_data.get("event_pitch_change", 0.8)),
        min_event_frames=int(grammar_data.get("min_event_frames", 2)),
        phrase_gap_ms=float(grammar_data.get("phrase_gap_ms", 140.0)),
        phrase_long_note_factor=float(grammar_data.get("phrase_long_note_factor", 2.2)),
        phrase_leap_semitones=float(grammar_data.get("phrase_leap_semitones", 7.0)),
        min_phrase_events=int(grammar_data.get("min_phrase_events", 3)),
        interval_weights=_int_key_dict(grammar_data.get("interval_weights")),
        degree_weights=_int_key_dict(grammar_data.get("degree_weights")),
        transition_weights=_str_float_dict(grammar_data.get("transition_weights")),
        ascending_transition_weights=_str_float_dict(grammar_data.get("ascending_transition_weights")),
        descending_transition_weights=_str_float_dict(grammar_data.get("descending_transition_weights")),
        trigram_weights=_str_float_dict(grammar_data.get("trigram_weights")),
        preferred_phrases=_weighted_sequences(grammar_data.get("preferred_phrases")),
        cadence_degrees=_int_key_dict(grammar_data.get("cadence_degrees")),
        cadence_patterns=_weighted_sequences(grammar_data.get("cadence_patterns")),
    )

    rhythm = RhythmProfile(
        enabled=bool(rhythm_data.get("enabled", False)),
        preserve_phrase_duration=bool(rhythm_data.get("preserve_phrase_duration", True)),
        quantize_strength=float(rhythm_data.get("quantize_strength", 0.0)),
        preferred_duration_ratios=tuple(float(x) for x in rhythm_data.get(
            "preferred_duration_ratios", [0.5, 1.0, 2.0]
        )),
        degree_duration_multipliers=_int_key_dict(rhythm_data.get("degree_duration_multipliers")),
        phrase_end_multiplier=float(rhythm_data.get("phrase_end_multiplier", 1.0)),
        gap_multiplier=float(rhythm_data.get("gap_multiplier", 1.0)),
        max_duration_change=float(rhythm_data.get("max_duration_change", 0.35)),
    )

    tuning = TuningProfile(
        degree_cents=_int_key_dict(tuning_data.get("degree_cents"))
    )

    ornaments = tuple(
        OrnamentRule(
            type=str(item.get("type", "grace")),
            degrees=tuple(int(x) % 12 for x in item.get("degrees", [])),
            probability=float(item.get("probability", 1.0)),
            scale_steps=int(item.get("scale_steps", 0)),
            fraction=float(item.get("fraction", 0.12)),
            max_ms=float(item.get("max_ms", 90.0)),
            depth_cents=float(item.get("depth_cents", 0.0)),
            rate_hz=float(item.get("rate_hz", 5.0)),
            delay_fraction=float(item.get("delay_fraction", 0.10)),
        )
        for item in data.get("ornaments", [])
    )

    modulation_options: list[ModulationOption] = []
    for item in modulation_data.get("options", []):
        scale = tuple(sorted({int(x) % 12 for x in item.get("scale", [])}))
        modulation_options.append(ModulationOption(
            name=str(item.get("name", "modulation")),
            root_offset=int(item.get("root_offset", 0)) % 12,
            scale=scale,
            min_phrase_events=int(item.get("min_phrase_events", 4)),
            switch_penalty=float(item.get("switch_penalty", 1.5)),
            activation_bonus=float(item.get("activation_bonus", 0.0)),
        ))

    modulation = ModulationProfile(
        enabled=bool(modulation_data.get("enabled", False)),
        options=tuple(modulation_options),
    )

    scale = tuple(sorted({int(x) % 12 for x in data["scale"]}))
    if not scale:
        raise ValueError(f"Style {path} has an empty scale")

    return StyleProfile(
        id=str(data["id"]),
        label=str(data.get("label", data["id"])),
        region=str(data.get("region", "")),
        description=str(data.get("description", "")),
        scale=scale,
        grammar=grammar,
        rhythm=rhythm,
        tuning=tuning,
        ornaments=ornaments,
        modulation=modulation,
        notes=str(data.get("notes", "")),
    )


def load_style_profiles(style_dir: Path) -> dict[str, StyleProfile]:
    if not style_dir.exists():
        raise FileNotFoundError(f"Style directory not found: {style_dir}")
    profiles: dict[str, StyleProfile] = {}
    for path in sorted(style_dir.glob("*.json")):
        profile = load_style_profile(path)
        if profile.id in profiles:
            raise ValueError(f"Duplicate style id '{profile.id}' in {path}")
        profiles[profile.id] = profile
    if not profiles:
        raise RuntimeError(f"No JSON style profiles found in {style_dir}")
    return profiles


def allowed_midi_notes(
    root_pc: int,
    intervals: tuple[int, ...] | list[int],
    low: int = 0,
    high: int = 127,
) -> np.ndarray:
    pcs = {(root_pc + int(x)) % 12 for x in intervals}
    return np.asarray(
        [note for note in range(low, high + 1) if note % 12 in pcs],
        dtype=np.int16,
    )


def degree_of_midi(note: float | int, root_pc: int) -> int:
    return (int(round(float(note))) - root_pc) % 12


def _candidate_notes(source_pitch: float, allowed: np.ndarray, grammar: GrammarProfile) -> np.ndarray:
    distance = np.abs(allowed.astype(np.float64) - source_pitch)
    within = np.flatnonzero(distance <= grammar.candidate_shift_semitones)
    if len(within) == 0:
        within = np.asarray([int(np.argmin(distance))])
    ranked = within[np.argsort(distance[within])]
    ranked = ranked[: max(1, grammar.candidate_count)]
    return allowed[ranked].astype(np.int16)


def extract_note_events(source_midi: np.ndarray, grammar: GrammarProfile) -> tuple[NoteEvent, ...]:
    """Turn a smoothed frame-level pitch track into stable note events."""
    source_midi = np.asarray(source_midi, dtype=np.float64)
    events: list[NoteEvent] = []

    i = 0
    while i < len(source_midi):
        if not np.isfinite(source_midi[i]):
            i += 1
            continue

        start = i
        values = [float(source_midi[i])]
        i += 1

        while i < len(source_midi) and np.isfinite(source_midi[i]):
            value = float(source_midi[i])
            median = float(np.median(values))
            if abs(value - median) >= grammar.event_pitch_change:
                events.append(NoteEvent(start, i, float(np.median(values))))
                start = i
                values = [value]
            else:
                values.append(value)
            i += 1

        events.append(NoteEvent(start, i, float(np.median(values))))

    # Merge tiny events into a neighboring event with the closest pitch.
    mutable = [[e.start_frame, e.end_frame, e.source_midi] for e in events]
    min_frames = max(1, grammar.min_event_frames)
    changed = True
    while changed and len(mutable) > 1:
        changed = False
        for idx, (s, e, p) in enumerate(mutable):
            if e - s >= min_frames:
                continue
            if idx == 0:
                mutable[1][0] = s
                del mutable[0]
            elif idx == len(mutable) - 1:
                mutable[idx - 1][1] = e
                del mutable[idx]
            else:
                prev_pitch = mutable[idx - 1][2]
                next_pitch = mutable[idx + 1][2]
                if abs(p - prev_pitch) <= abs(p - next_pitch):
                    mutable[idx - 1][1] = e
                    del mutable[idx]
                else:
                    mutable[idx + 1][0] = s
                    del mutable[idx]
            changed = True
            break

    return tuple(NoteEvent(int(s), int(e), float(p)) for s, e, p in mutable)


def detect_phrases(
    events: tuple[NoteEvent, ...],
    sr: int,
    hop_length: int,
    grammar: GrammarProfile,
) -> tuple[Phrase, ...]:
    if not events:
        return ()

    durations = np.asarray([e.frames for e in events], dtype=np.float64)
    median_frames = max(1.0, float(np.median(durations)))
    gap_threshold_frames = max(1, int(round(grammar.phrase_gap_ms / 1000.0 * sr / hop_length)))

    phrases: list[Phrase] = []
    current: list[NoteEvent] = [events[0]]

    for prev, curr in zip(events, events[1:]):
        gap = max(0, curr.start_frame - prev.end_frame)
        long_prev = prev.frames >= grammar.phrase_long_note_factor * median_frames
        large_leap = abs(curr.source_midi - prev.source_midi) >= grammar.phrase_leap_semitones

        should_split = gap >= gap_threshold_frames
        if len(current) >= grammar.min_phrase_events and long_prev:
            # A sustained note followed by even a short articulation gap is a
            # useful generic phrase-boundary cue; with fully legato material,
            # require a large leap instead.
            if gap > 0 or large_leap:
                should_split = True

        if should_split:
            phrases.append(Phrase(tuple(current), current[0].start_frame, current[-1].end_frame))
            current = [curr]
        else:
            current.append(curr)

    if current:
        phrases.append(Phrase(tuple(current), current[0].start_frame, current[-1].end_frame))

    # Avoid one-note phrases by merging into the previous phrase when possible.
    merged: list[Phrase] = []
    for phrase in phrases:
        if len(phrase.events) < grammar.min_phrase_events and merged:
            prev = merged[-1]
            events2 = prev.events + phrase.events
            merged[-1] = Phrase(events2, prev.start_frame, phrase.end_frame)
        else:
            merged.append(phrase)

    return tuple(merged)


def _emission_cost(
    source_pitch: float,
    target_pitch: int,
    root_pc: int,
    profile: StyleProfile,
    style_amount: float,
) -> float:
    degree = degree_of_midi(target_pitch, root_pc)
    cost = profile.grammar.pitch_deviation_weight * abs(target_pitch - source_pitch)
    cost -= style_amount * profile.grammar.degree_weights.get(degree, 0.0)
    return float(cost)


def _transition_cost(
    prev_source: float,
    source: float,
    prev_target: int,
    target: int,
    root_pc: int,
    profile: StyleProfile,
    style_amount: float,
) -> float:
    grammar = profile.grammar
    input_delta = float(source - prev_source)
    output_delta = float(target - prev_target)

    cost = grammar.motion_preservation_weight * abs(output_delta - input_delta)

    if abs(input_delta) >= 0.5 and abs(output_delta) >= 0.5:
        if np.sign(input_delta) != np.sign(output_delta):
            cost += grammar.contour_penalty

    leap = abs(output_delta)
    if leap > grammar.max_preferred_leap:
        cost += grammar.leap_penalty * (leap - grammar.max_preferred_leap)

    interval_key = min(12, int(round(leap)))
    cost -= style_amount * grammar.interval_weights.get(interval_key, 0.0)

    prev_degree = degree_of_midi(prev_target, root_pc)
    degree = degree_of_midi(target, root_pc)
    key = f"{prev_degree}>{degree}"
    cost -= style_amount * grammar.transition_weights.get(key, 0.0)

    # Direction-aware grammar is selected using the source melodic direction.
    if input_delta > 0.35:
        cost -= style_amount * grammar.ascending_transition_weights.get(key, 0.0)
    elif input_delta < -0.35:
        cost -= style_amount * grammar.descending_transition_weights.get(key, 0.0)

    return float(cost)


def _higher_order_bonus(
    history: tuple[int, ...],
    target: int,
    root_pc: int,
    profile: StyleProfile,
    style_amount: float,
) -> float:
    grammar = profile.grammar
    notes = history + (target,)
    degrees = tuple(degree_of_midi(note, root_pc) for note in notes)
    bonus = 0.0

    if len(degrees) >= 3:
        key = ">".join(str(x) for x in degrees[-3:])
        bonus += grammar.trigram_weights.get(key, 0.0)

    for pattern, weight in grammar.preferred_phrases:
        if len(degrees) >= len(pattern) and degrees[-len(pattern):] == pattern:
            bonus += weight

    return style_amount * bonus


def _cadence_bonus(
    history: tuple[int, ...],
    root_pc: int,
    grammar: GrammarProfile,
    style_amount: float,
) -> float:
    if not history:
        return 0.0
    degrees = tuple(degree_of_midi(note, root_pc) for note in history)
    bonus = grammar.cadence_degrees.get(degrees[-1], 0.0)
    for pattern, weight in grammar.cadence_patterns:
        if len(degrees) >= len(pattern) and degrees[-len(pattern):] == pattern:
            bonus += weight
    return style_amount * bonus


def map_phrase_viterbi(
    phrase: Phrase,
    root_pc: int,
    scale: tuple[int, ...],
    profile: StyleProfile,
    style_amount: float,
) -> tuple[tuple[int, ...], float]:
    """
    Higher-order Viterbi/DP.

    State keeps the last three target notes, which is enough to score trigrams
    and four-note preferred phrases without exploding the state space.
    """
    events = phrase.events
    if not events:
        return (), 0.0

    allowed = allowed_midi_notes(root_pc, scale)
    candidates = [_candidate_notes(e.source_midi, allowed, profile.grammar) for e in events]
    max_history = max(3, max((len(p) - 1 for p, _ in profile.grammar.preferred_phrases), default=0))
    max_history = min(max_history, 4)

    # states[t][history] = best cost ending in that history
    states: dict[tuple[int, ...], float] = {}
    backs: list[dict[tuple[int, ...], tuple[int, ...] | None]] = []

    first_back: dict[tuple[int, ...], tuple[int, ...] | None] = {}
    for note in candidates[0]:
        n = int(note)
        h = (n,)
        states[h] = _emission_cost(events[0].source_midi, n, root_pc, profile, style_amount)
        first_back[h] = None
    backs.append(first_back)

    for t in range(1, len(events)):
        new_states: dict[tuple[int, ...], float] = {}
        new_back: dict[tuple[int, ...], tuple[int, ...] | None] = {}

        for prev_history, prev_cost in states.items():
            prev_note = prev_history[-1]
            for note in candidates[t]:
                n = int(note)
                cost = prev_cost
                cost += _emission_cost(events[t].source_midi, n, root_pc, profile, style_amount)
                cost += _transition_cost(
                    events[t - 1].source_midi,
                    events[t].source_midi,
                    prev_note,
                    n,
                    root_pc,
                    profile,
                    style_amount,
                )
                cost -= _higher_order_bonus(prev_history, n, root_pc, profile, style_amount)

                history = (prev_history + (n,))[-max_history:]
                if cost < new_states.get(history, np.inf):
                    new_states[history] = cost
                    new_back[history] = prev_history

        states = new_states
        backs.append(new_back)

    best_history = min(
        states,
        key=lambda h: states[h] - _cadence_bonus(h, root_pc, profile.grammar, style_amount),
    )
    best_cost = states[best_history] - _cadence_bonus(best_history, root_pc, profile.grammar, style_amount)

    result = [0] * len(events)
    history = best_history
    for t in range(len(events) - 1, -1, -1):
        result[t] = history[-1]
        prev = backs[t].get(history)
        if prev is None:
            break
        history = prev

    return tuple(result), float(best_cost)


def map_melody_viterbi(
    source_midi: np.ndarray,
    sr: int,
    hop_length: int,
    root_pc: int,
    profile: StyleProfile,
    style_amount: float,
    enable_modulation: bool = True,
) -> MappingResult:
    events = extract_note_events(source_midi, profile.grammar)
    phrases = detect_phrases(events, sr, hop_length, profile.grammar)

    targets: list[tuple[int, ...]] = []
    roots: list[int] = []
    scales: list[tuple[int, ...]] = []
    costs: list[float] = []
    modulation_names: list[str] = []

    previous_root = root_pc

    for phrase in phrases:
        options: list[tuple[str, int, tuple[int, ...], float]] = [
            ("base", root_pc, profile.scale, 0.0)
        ]

        if enable_modulation and profile.modulation.enabled:
            for option in profile.modulation.options:
                if len(phrase.events) < option.min_phrase_events:
                    continue
                option_root = (root_pc + option.root_offset) % 12
                option_scale = option.scale or profile.scale
                penalty = option.switch_penalty if option_root != previous_root else 0.0
                penalty -= style_amount * option.activation_bonus
                options.append((option.name, option_root, option_scale, penalty))

        best: tuple[str, int, tuple[int, ...], tuple[int, ...], float] | None = None
        for name, option_root, option_scale, extra_cost in options:
            mapped, cost = map_phrase_viterbi(
                phrase,
                option_root,
                option_scale,
                profile,
                style_amount,
            )
            cost += extra_cost
            if best is None or cost < best[-1]:
                best = (name, option_root, option_scale, mapped, cost)

        assert best is not None
        name, chosen_root, chosen_scale, mapped, cost = best
        targets.append(mapped)
        roots.append(chosen_root)
        scales.append(tuple(chosen_scale))
        costs.append(float(cost))
        modulation_names.append(name)
        previous_root = chosen_root

    return MappingResult(
        phrases=phrases,
        targets=tuple(targets),
        roots=tuple(roots),
        scales=tuple(scales),
        costs=tuple(costs),
        modulation_names=tuple(modulation_names),
    )
