from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OrnamentProfile:
    grace_probability: float = 0.0
    grace_scale_steps: int = 0
    grace_fraction: float = 0.0
    vibrato_cents: float = 0.0
    vibrato_hz: float = 5.0
    vibrato_degrees: tuple[int, ...] = ()


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
    interval_weights: dict[int, float] = field(default_factory=dict)
    degree_weights: dict[int, float] = field(default_factory=dict)
    transition_weights: dict[str, float] = field(default_factory=dict)
    cadence_degrees: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StyleProfile:
    id: str
    label: str
    region: str
    description: str
    scale: tuple[int, ...]
    grammar: GrammarProfile
    ornament: OrnamentProfile
    notes: str = ""


def _int_key_dict(data: dict | None) -> dict[int, float]:
    if not data:
        return {}
    return {int(k): float(v) for k, v in data.items()}


def load_style_profile(path: Path) -> StyleProfile:
    data = json.loads(path.read_text(encoding="utf-8"))

    grammar_data = data.get("grammar", {})
    ornament_data = data.get("ornament", {})

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
        interval_weights=_int_key_dict(grammar_data.get("interval_weights")),
        degree_weights=_int_key_dict(grammar_data.get("degree_weights")),
        transition_weights={
            str(k): float(v)
            for k, v in grammar_data.get("transition_weights", {}).items()
        },
        cadence_degrees=_int_key_dict(grammar_data.get("cadence_degrees")),
    )

    ornament = OrnamentProfile(
        grace_probability=float(ornament_data.get("grace_probability", 0.0)),
        grace_scale_steps=int(ornament_data.get("grace_scale_steps", 0)),
        grace_fraction=float(ornament_data.get("grace_fraction", 0.0)),
        vibrato_cents=float(ornament_data.get("vibrato_cents", 0.0)),
        vibrato_hz=float(ornament_data.get("vibrato_hz", 5.0)),
        vibrato_degrees=tuple(int(x) for x in ornament_data.get("vibrato_degrees", [])),
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
        ornament=ornament,
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


def _candidate_notes(
    source_pitch: float,
    allowed: np.ndarray,
    grammar: GrammarProfile,
) -> np.ndarray:
    distance = np.abs(allowed.astype(np.float64) - source_pitch)
    within = np.flatnonzero(distance <= grammar.candidate_shift_semitones)

    if len(within) == 0:
        within = np.asarray([int(np.argmin(distance))])

    ranked = within[np.argsort(distance[within])]
    ranked = ranked[: max(1, grammar.candidate_count)]
    return allowed[ranked].astype(np.int16)


def _build_note_chains(
    source_midi: np.ndarray,
    grammar: GrammarProfile,
) -> list[list[tuple[int, int, float]]]:
    """
    Convert a smoothed frame-level F0 track into chains of note events.

    Each event is (start_frame, end_frame_exclusive, representative_midi).
    Unvoiced gaps split chains, so Viterbi does not force transitions across
    long rests/consonants.
    """
    source_midi = np.asarray(source_midi, dtype=np.float64)
    chains: list[list[tuple[int, int, float]]] = []

    i = 0
    while i < len(source_midi):
        if not np.isfinite(source_midi[i]):
            i += 1
            continue

        chain: list[tuple[int, int, float]] = []
        start = i
        current_values = [float(source_midi[i])]
        reference = float(source_midi[i])
        i += 1

        while i < len(source_midi) and np.isfinite(source_midi[i]):
            value = float(source_midi[i])
            median = float(np.median(current_values))

            if abs(value - median) >= grammar.event_pitch_change:
                chain.append((start, i, float(np.median(current_values))))
                start = i
                current_values = [value]
                reference = value
            else:
                current_values.append(value)
                reference = median
            i += 1

        chain.append((start, i, float(np.median(current_values))))

        # Merge very short events into the neighbor whose pitch is closest.
        min_frames = max(1, grammar.min_event_frames)
        changed = True
        while changed and len(chain) > 1:
            changed = False
            for idx, (s, e, p) in enumerate(chain):
                if e - s >= min_frames:
                    continue

                if idx == 0:
                    ns, ne, npitch = chain[1]
                    chain[1] = (s, ne, npitch)
                    del chain[0]
                elif idx == len(chain) - 1:
                    ps, pe, ppitch = chain[idx - 1]
                    chain[idx - 1] = (ps, e, ppitch)
                    del chain[idx]
                else:
                    prev_pitch = chain[idx - 1][2]
                    next_pitch = chain[idx + 1][2]
                    if abs(p - prev_pitch) <= abs(p - next_pitch):
                        ps, pe, ppitch = chain[idx - 1]
                        chain[idx - 1] = (ps, e, ppitch)
                        del chain[idx]
                    else:
                        ns, ne, npitch = chain[idx + 1]
                        chain[idx + 1] = (s, ne, npitch)
                        del chain[idx]
                changed = True
                break

        chains.append(chain)

    return chains


def _emission_cost(
    source_pitch: float,
    target_pitch: int,
    root_pc: int,
    profile: StyleProfile,
    style_amount: float,
) -> float:
    grammar = profile.grammar
    degree = degree_of_midi(target_pitch, root_pc)

    cost = grammar.pitch_deviation_weight * abs(target_pitch - source_pitch)
    cost -= style_amount * grammar.degree_weights.get(degree, 0.0)
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
    transition_key = f"{prev_degree}>{degree}"
    cost -= style_amount * grammar.transition_weights.get(transition_key, 0.0)

    return float(cost)


def _map_chain_viterbi(
    chain: list[tuple[int, int, float]],
    allowed: np.ndarray,
    root_pc: int,
    profile: StyleProfile,
    style_amount: float,
) -> list[int]:
    if not chain:
        return []

    candidates = [
        _candidate_notes(event[2], allowed, profile.grammar)
        for event in chain
    ]

    dp: list[np.ndarray] = []
    back: list[np.ndarray] = []

    first = np.asarray([
        _emission_cost(chain[0][2], int(note), root_pc, profile, style_amount)
        for note in candidates[0]
    ], dtype=np.float64)
    dp.append(first)
    back.append(np.full(len(candidates[0]), -1, dtype=np.int32))

    for t in range(1, len(chain)):
        curr_cost = np.full(len(candidates[t]), np.inf, dtype=np.float64)
        curr_back = np.full(len(candidates[t]), -1, dtype=np.int32)

        for j, curr_note in enumerate(candidates[t]):
            emission = _emission_cost(
                chain[t][2], int(curr_note), root_pc, profile, style_amount
            )

            for k, prev_note in enumerate(candidates[t - 1]):
                transition = _transition_cost(
                    chain[t - 1][2], chain[t][2],
                    int(prev_note), int(curr_note),
                    root_pc, profile, style_amount,
                )
                value = dp[t - 1][k] + transition + emission
                if value < curr_cost[j]:
                    curr_cost[j] = value
                    curr_back[j] = k

        dp.append(curr_cost)
        back.append(curr_back)

    final_cost = dp[-1].copy()
    for j, note in enumerate(candidates[-1]):
        degree = degree_of_midi(int(note), root_pc)
        final_cost[j] -= (
            style_amount * profile.grammar.cadence_degrees.get(degree, 0.0)
        )

    state = int(np.argmin(final_cost))
    result = [0] * len(chain)

    for t in range(len(chain) - 1, -1, -1):
        result[t] = int(candidates[t][state])
        if t > 0:
            state = int(back[t][state])

    return result


def map_melody_viterbi(
    source_midi: np.ndarray,
    root_pc: int,
    profile: StyleProfile,
    style_amount: float,
) -> np.ndarray:
    """
    Map a smoothed frame-level melody into the style scale using event-level
    Viterbi optimization.

    The optimizer balances:
      - distance from the original pitch
      - preservation of melodic contour and interval size
      - style-preferred interval sizes
      - style-preferred scale degrees and degree-to-degree transitions
      - final cadence preference
    """
    result = np.full_like(source_midi, np.nan, dtype=np.float64)
    allowed = allowed_midi_notes(root_pc, profile.scale)
    chains = _build_note_chains(source_midi, profile.grammar)

    for chain in chains:
        mapped = _map_chain_viterbi(
            chain, allowed, root_pc, profile, style_amount
        )
        for (start, end, _), target in zip(chain, mapped):
            result[start:end] = float(target)

    return result
