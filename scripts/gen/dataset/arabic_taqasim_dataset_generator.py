#!/usr/bin/env python3
"""
arabic_taqasim_dataset_generator.py

Build a neutral monophonic Arab maqam / taqasim WAV corpus for Scaleify.

Source
------
Fadi Al-Ghawanmeh et al., TaqasimDataset:
https://github.com/FadiGhawanmeh/TaqasimDataset

Dataset archive:
"All taqasim MIDI files.zip"

Reference paper:
F. Al-Ghawanmeh, A. R. Jensenius, and K. Smaili,
"How can machine translation help generate Arab melodic improvisation?",
EAMT 2023, pp. 385-392.
https://aclanthology.org/2023.eamt-1.38/

The paper reports 717 instrumental improvisations across eight main maqamat:
Ajam, Hijaz, Bayati, Kurd, Rast, Nahawand, Huzam, and Saba.

Rights
------
The public GitHub repository asks users to cite the EAMT 2023 paper, but no
explicit dataset license is assumed by this generator. The generated corpus is
therefore intended for LOCAL research use. Verify redistribution rights before
publishing or redistributing source MIDI or rendered WAV files.

Why neutral rendering?
----------------------
Scaleify aims to learn melodic grammar rather than source instrumentation.
The generator therefore:

    public MIDI archive
        -> safe extraction
        -> detect maqam label from archive path/name
        -> inspect candidate MIDI tracks
        -> select likely melodic track
        -> reduce simultaneous notes to one line
        -> preserve source timing by default
        -> preserve note-on pitch bend when available
        -> discard MIDI program/instrument identity
        -> render every file with the same neutral timbre
        -> datasets/arabic/*.wav

The original extracted MIDI files remain under:
    datasets/arabic/_source_midi/

Usage
-----
From the Scaleify repository root:

    python3 scripts/gen/dataset/arabic_taqasim_dataset_generator.py
    python3 scripts/gen/dataset/arabic_taqasim_dataset_generator.py --list
    python3 scripts/gen/dataset/arabic_taqasim_dataset_generator.py --maqam rast
    python3 scripts/gen/dataset/arabic_taqasim_dataset_generator.py --limit 20
    python3 scripts/gen/dataset/arabic_taqasim_dataset_generator.py --force

Dependencies
------------
numpy
requests
soundfile
mido
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import mido
import numpy as np
import requests
import soundfile as sf


SR = 44100
DEFAULT_OUTPUT = Path("datasets/arabic")

REPOSITORY_URL = "https://github.com/FadiGhawanmeh/TaqasimDataset"
ARCHIVE_URL = (
    "https://github.com/FadiGhawanmeh/TaqasimDataset/raw/refs/heads/main/"
    "All%20taqasim%20MIDI%20files.zip"
)
PAPER_URL = "https://aclanthology.org/2023.eamt-1.38/"
CITATION = (
    "Al-Ghawanmeh, F., Jensenius, A. R., & Smaili, K. (2023). "
    "How can machine translation help generate Arab melodic improvisation? "
    "Proceedings of EAMT 2023, 385-392."
)

DEFAULT_FIXED_BPM = 100.0
DEFAULT_GAP_MS = 14.0
DEFAULT_PITCH_BEND_RANGE = 2.0

RIGHTS_NOTE = (
    "The source repository is publicly accessible and requests citation of the "
    "EAMT 2023 paper. This generator does not assume an explicit redistribution "
    "license for the dataset. Keep source MIDI and generated WAVs local for "
    "research unless redistribution rights are separately verified."
)

MAQAM_ORDER = (
    "ajam",
    "hijaz",
    "bayati",
    "kurd",
    "rast",
    "nahawand",
    "huzam",
    "saba",
)

MAQAM_DISPLAY = {
    "ajam": "Ajam",
    "hijaz": "Hijaz",
    "bayati": "Bayati",
    "kurd": "Kurd",
    "rast": "Rast",
    "nahawand": "Nahawand",
    "huzam": "Huzam",
    "saba": "Saba",
    "unknown": "Unknown",
}

# Archive naming is not treated as a formal API. Accept common spellings so the
# generator remains robust to folder/file naming differences.
MAQAM_ALIASES: dict[str, tuple[str, ...]] = {
    "ajam": ("ajam", "agam"),
    "hijaz": ("hijaz", "hejazi", "hejazz"),
    "bayati": ("bayati", "bayaty", "bayat"),
    "kurd": ("kurd", "kurdi"),
    "rast": ("rast", "raast"),
    "nahawand": ("nahawand", "nahwand", "nihawand", "nehawand"),
    "huzam": ("huzam", "huzzam", "hizam"),
    "saba": ("saba", "sabaa"),
}


@dataclass
class MidiNote:
    start: int
    end: int
    pitch: int
    velocity: int = 80
    bend_semitones: float = 0.0


@dataclass
class TrackCandidate:
    index: int
    name: str
    notes: list[MidiNote]
    score: float
    monophony: float
    mean_pitch: float
    unique_pitches: int
    pitchwheel_messages: int


@dataclass(frozen=True)
class SourceItem:
    global_index: int
    path: Path
    relative_path: str
    maqam: str
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_token(text: str) -> str:
    text = unquote(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def infer_maqam(relative_path: str) -> str:
    text = f" {normalize_token(relative_path)} "

    matches: list[tuple[int, str]] = []
    for maqam, aliases in MAQAM_ALIASES.items():
        for alias in aliases:
            pattern = rf"(?:^| ){re.escape(alias)}(?: |$)"
            match = re.search(pattern, text)
            if match:
                matches.append((match.start(), maqam))
                break

    if not matches:
        return "unknown"

    # Prefer the earliest label in the path, which usually corresponds to the
    # parent maqam folder rather than an incidental filename token.
    matches.sort()
    return matches[0][1]


def fetch_bytes(
    session: requests.Session,
    url: str,
    timeout: float,
) -> bytes:
    response = session.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response.content


def download_archive(
    session: requests.Session,
    archive_path: Path,
    timeout: float,
    force: bool,
) -> None:
    if archive_path.exists() and not force:
        return

    print(f"Downloading source archive:\n  {ARCHIVE_URL}")
    data = fetch_bytes(session, ARCHIVE_URL, timeout)

    if len(data) < 4 or data[:4] != b"PK\x03\x04":
        raise RuntimeError(
            "Downloaded source is not a ZIP archive. "
            f"Received {len(data)} bytes from {ARCHIVE_URL}"
        )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(data)


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    force: bool,
) -> None:
    marker = destination / ".scaleify_extract_complete"

    if marker.exists() and not force:
        return

    if force and destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with zipfile.ZipFile(archive_path) as zf:
        members = zf.infolist()

        for member in members:
            # Normalize Windows separators before validating archive paths.
            raw_name = member.filename.replace("\\", "/")
            pure = PurePosixPath(raw_name)

            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(
                    f"Unsafe ZIP member path rejected: {member.filename!r}"
                )

            if not raw_name or raw_name.endswith("/"):
                continue

            target = (destination / Path(*pure.parts)).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(
                    f"ZIP path escapes extraction directory: {member.filename!r}"
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    marker.write_text(
        f"archive_sha256={sha256_file(archive_path)}\n",
        encoding="utf-8",
    )


def discover_source_items(source_dir: Path) -> list[SourceItem]:
    midi_paths = sorted(
        (
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
        ),
        key=lambda p: p.relative_to(source_dir).as_posix().lower(),
    )

    seen_hashes: set[str] = set()
    unique: list[tuple[Path, str, str]] = []

    for path in midi_paths:
        rel = path.relative_to(source_dir).as_posix()
        digest = sha256_file(path)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique.append((path, rel, digest))

    return [
        SourceItem(
            global_index=i,
            path=path,
            relative_path=rel,
            maqam=infer_maqam(rel),
            sha256=digest,
        )
        for i, (path, rel, digest) in enumerate(unique, start=1)
    ]


def update_rpn_state(
    msg,
    channel: int,
    rpn_msb: dict[int, int],
    rpn_lsb: dict[int, int],
    bend_range_semitones: dict[int, float],
    range_coarse: dict[int, int],
    range_fine: dict[int, int],
) -> None:
    if msg.type != "control_change":
        return

    control = int(msg.control)
    value = int(msg.value)

    if control == 101:
        rpn_msb[channel] = value
        return
    if control == 100:
        rpn_lsb[channel] = value
        return

    if (rpn_msb[channel], rpn_lsb[channel]) != (0, 0):
        return

    if control == 6:
        range_coarse[channel] = value
    elif control == 38:
        range_fine[channel] = value
    else:
        return

    bend_range_semitones[channel] = (
        float(range_coarse[channel]) + float(range_fine[channel]) / 100.0
    )


def track_notes(
    track: mido.MidiTrack,
    default_pitch_bend_range: float,
) -> tuple[str, list[MidiNote], int]:
    absolute = 0
    name = ""

    active: dict[
        tuple[int, int],
        list[tuple[int, int, float]],
    ] = {}

    notes: list[MidiNote] = []

    bend_value = {ch: 0 for ch in range(16)}
    bend_range = {
        ch: float(default_pitch_bend_range)
        for ch in range(16)
    }
    rpn_msb = {ch: 127 for ch in range(16)}
    rpn_lsb = {ch: 127 for ch in range(16)}
    range_coarse = {
        ch: int(math.floor(default_pitch_bend_range))
        for ch in range(16)
    }
    range_fine = {
        ch: int(round((default_pitch_bend_range % 1.0) * 100.0))
        for ch in range(16)
    }

    pitchwheel_messages = 0

    for msg in track:
        absolute += int(msg.time)

        if msg.type == "track_name" and not name:
            name = str(msg.name)
            continue

        if not hasattr(msg, "channel"):
            continue

        channel = int(msg.channel)
        if channel == 9:
            continue

        if msg.type == "control_change":
            update_rpn_state(
                msg=msg,
                channel=channel,
                rpn_msb=rpn_msb,
                rpn_lsb=rpn_lsb,
                bend_range_semitones=bend_range,
                range_coarse=range_coarse,
                range_fine=range_fine,
            )
            continue

        if msg.type == "pitchwheel":
            bend_value[channel] = int(msg.pitch)
            pitchwheel_messages += 1
            continue

        if msg.type == "note_on" and int(msg.velocity) > 0:
            semitones = (
                float(bend_value[channel])
                / 8192.0
                * float(bend_range[channel])
            )
            key = (channel, int(msg.note))
            active.setdefault(key, []).append(
                (absolute, int(msg.velocity), semitones)
            )
            continue

        if msg.type in ("note_off", "note_on"):
            key = (channel, int(msg.note))
            stack = active.get(key)
            if not stack:
                continue

            start, velocity, bend_semitones = stack.pop(0)
            if absolute <= start:
                continue

            notes.append(
                MidiNote(
                    start=start,
                    end=absolute,
                    pitch=int(msg.note),
                    velocity=velocity,
                    bend_semitones=bend_semitones,
                )
            )

    return name, notes, pitchwheel_messages


def overlap_monophony(notes: list[MidiNote]) -> float:
    if len(notes) <= 1:
        return 1.0

    ordered = sorted(notes, key=lambda n: (n.start, n.end, n.pitch))
    overlaps = 0
    current_end = ordered[0].end

    for note in ordered[1:]:
        if note.start < current_end:
            overlaps += 1
        current_end = max(current_end, note.end)

    return max(0.0, 1.0 - overlaps / max(1, len(ordered) - 1))


def score_track(
    index: int,
    name: str,
    notes: list[MidiNote],
    pitchwheel_messages: int,
) -> TrackCandidate | None:
    if len(notes) < 6:
        return None

    pitches = np.asarray(
        [n.pitch + n.bend_semitones for n in notes],
        dtype=np.float64,
    )
    mean_pitch = float(np.mean(pitches))
    unique = len(set(round(float(x), 2) for x in pitches))
    mono = overlap_monophony(notes)

    name_l = name.lower()
    name_bonus = 0.0

    if any(token in name_l for token in ("melody", "lead", "solo", "oud")):
        name_bonus += 180.0
    if any(token in name_l for token in ("bass", "chord", "drum", "perc")):
        name_bonus -= 180.0

    early_bonus = max(0.0, 45.0 - 8.0 * index)
    low_penalty = max(0.0, 45.0 - mean_pitch) * 4.0

    score = (
        2.8 * len(notes) * mono
        + 2.0 * unique
        + 1.0 * mean_pitch
        + 140.0 * mono
        + name_bonus
        + early_bonus
        - low_penalty
    )

    return TrackCandidate(
        index=index,
        name=name,
        notes=notes,
        score=float(score),
        monophony=float(mono),
        mean_pitch=mean_pitch,
        unique_pitches=unique,
        pitchwheel_messages=pitchwheel_messages,
    )


def select_melody_track(
    midi: mido.MidiFile,
    strategy: str,
    default_pitch_bend_range: float,
) -> tuple[TrackCandidate, list[TrackCandidate]]:
    candidates: list[TrackCandidate] = []

    for index, track in enumerate(midi.tracks):
        name, notes, pitchwheel_messages = track_notes(
            track,
            default_pitch_bend_range=default_pitch_bend_range,
        )
        candidate = score_track(
            index=index,
            name=name,
            notes=notes,
            pitchwheel_messages=pitchwheel_messages,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No pitched MIDI track with enough note events")

    if strategy == "first":
        chosen = min(candidates, key=lambda c: c.index)
    elif strategy == "highest":
        mono = [c for c in candidates if c.monophony >= 0.80]
        chosen = max(mono or candidates, key=lambda c: (c.mean_pitch, c.score))
    else:
        chosen = max(candidates, key=lambda c: c.score)

    return chosen, sorted(candidates, key=lambda c: c.score, reverse=True)


def monophonize(notes: list[MidiNote]) -> list[MidiNote]:
    if not notes:
        return []

    groups: dict[int, list[MidiNote]] = {}
    for note in notes:
        groups.setdefault(note.start, []).append(note)

    chosen: list[MidiNote] = []

    for onset in sorted(groups):
        group = groups[onset]

        # Taqasim is expected to be melodic. If a simultaneous group exists,
        # keep the highest/strongest note consistently rather than preserving
        # a chord that Scaleify cannot model.
        best = max(
            group,
            key=lambda n: (
                n.pitch + n.bend_semitones,
                n.velocity,
                n.end - n.start,
            ),
        )

        chosen.append(
            MidiNote(
                start=best.start,
                end=best.end,
                pitch=best.pitch,
                velocity=best.velocity,
                bend_semitones=best.bend_semitones,
            )
        )

    out: list[MidiNote] = []

    for note in chosen:
        if out and note.start < out[-1].end:
            out[-1].end = max(out[-1].start + 1, note.start)

        if note.end <= note.start:
            continue

        out.append(note)

    return out


def tempo_events(midi: mido.MidiFile) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []

    for track in midi.tracks:
        absolute = 0
        for msg in track:
            absolute += int(msg.time)
            if msg.type == "set_tempo":
                events.append((absolute, int(msg.tempo)))

    events.sort(key=lambda item: item[0])

    # MIDI default tempo is 500000 us/beat (120 BPM).
    if not events or events[0][0] > 0:
        events.insert(0, (0, 500000))

    # If multiple tracks declare tempo at the same tick, keep the last one.
    collapsed: list[tuple[int, int]] = []
    for tick, tempo in events:
        if collapsed and collapsed[-1][0] == tick:
            collapsed[-1] = (tick, tempo)
        else:
            collapsed.append((tick, tempo))

    return collapsed


def ticks_to_seconds_map(
    ticks: set[int],
    ticks_per_beat: int,
    tempos: list[tuple[int, int]],
) -> dict[int, float]:
    targets = sorted(ticks)
    if not targets:
        return {}

    result: dict[int, float] = {}
    tempo_index = 0
    current_tempo = tempos[0][1]
    previous_tick = 0
    elapsed = 0.0

    for target in targets:
        while (
            tempo_index + 1 < len(tempos)
            and tempos[tempo_index + 1][0] <= target
        ):
            next_tick, next_tempo = tempos[tempo_index + 1]
            elapsed += mido.tick2second(
                next_tick - previous_tick,
                ticks_per_beat,
                current_tempo,
            )
            previous_tick = next_tick
            tempo_index += 1
            current_tempo = next_tempo

        result[target] = elapsed + mido.tick2second(
            target - previous_tick,
            ticks_per_beat,
            current_tempo,
        )

    return result


def midi_pitch_to_freq(pitch: float) -> float:
    return 440.0 * 2.0 ** ((float(pitch) - 69.0) / 12.0)


def synth_tone(
    midi_pitch: float,
    seconds: float,
    velocity: int,
) -> np.ndarray:
    n = max(1, int(round(seconds * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    freq = midi_pitch_to_freq(midi_pitch)
    phase = 2.0 * np.pi * freq * t

    # Same neutral family used by the other Scaleify corpus generators:
    # enough harmonics for stable F0/onset extraction without source timbre.
    y = (
        0.78 * np.sin(phase)
        + 0.15 * np.sin(2.0 * phase)
        + 0.05 * np.sin(3.0 * phase)
        + 0.02 * np.sin(4.0 * phase)
    )

    attack = min(n, max(1, int(round(0.006 * SR))))
    release = min(n, max(1, int(round(0.024 * SR))))

    env = np.ones(n, dtype=np.float64)

    if attack > 1:
        env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)

    if release > 1:
        env[-release:] *= np.linspace(1.0, 0.0, release)

    amp = 0.68 + 0.14 * min(
        1.0,
        max(0.0, float(velocity) / 127.0),
    )

    return (y * env * amp).astype(np.float32)


def note_seconds(
    midi: mido.MidiFile,
    notes: list[MidiNote],
    timing: str,
    fixed_bpm: float,
) -> list[tuple[MidiNote, float, float]]:
    first_tick = min(n.start for n in notes)

    shifted = [
        MidiNote(
            start=n.start - first_tick,
            end=n.end - first_tick,
            pitch=n.pitch,
            velocity=n.velocity,
            bend_semitones=n.bend_semitones,
        )
        for n in notes
    ]

    if timing == "fixed":
        seconds_per_tick = (
            60.0 / fixed_bpm
        ) / max(1, midi.ticks_per_beat)

        return [
            (
                note,
                note.start * seconds_per_tick,
                note.end * seconds_per_tick,
            )
            for note in shifted
        ]

    # Source tempo map uses original absolute ticks. Build the mapping before
    # subtracting the first-note time so tempo changes remain aligned.
    original_ticks = {
        tick
        for note in notes
        for tick in (note.start, note.end, first_tick)
    }
    mapping = ticks_to_seconds_map(
        ticks=original_ticks,
        ticks_per_beat=midi.ticks_per_beat,
        tempos=tempo_events(midi),
    )
    origin_s = mapping[first_tick]

    return [
        (
            shifted_note,
            mapping[original.start] - origin_s,
            mapping[original.end] - origin_s,
        )
        for shifted_note, original in zip(shifted, notes)
    ]


def render_melody(
    midi: mido.MidiFile,
    notes: list[MidiNote],
    timing: str,
    fixed_bpm: float,
    gap_ms: float,
) -> np.ndarray:
    if not notes:
        return np.zeros(1, dtype=np.float32)

    timed = note_seconds(
        midi=midi,
        notes=notes,
        timing=timing,
        fixed_bpm=fixed_bpm,
    )

    final_s = max(end_s for _, _, end_s in timed)
    audio = np.zeros(
        max(1, int(math.ceil((final_s + 0.10) * SR))),
        dtype=np.float32,
    )

    gap_s = max(0.0, gap_ms / 1000.0)

    for note, start_s, end_s in timed:
        duration_s = max(1.0 / SR, end_s - start_s)
        note_gap = min(gap_s, duration_s * 0.10)
        tone_s = max(0.016, duration_s - note_gap)

        tone = synth_tone(
            midi_pitch=note.pitch + note.bend_semitones,
            seconds=tone_s,
            velocity=note.velocity,
        )

        start = int(round(start_s * SR))
        end = min(len(audio), start + len(tone))

        if start < len(audio):
            audio[start:end] += tone[: end - start]

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio *= 0.88 / peak

    return audio.astype(np.float32)


def analyze_and_render(
    midi_path: Path,
    wav_path: Path,
    track_strategy: str,
    timing: str,
    fixed_bpm: float,
    gap_ms: float,
    pitch_bend_range: float,
) -> dict:
    midi = mido.MidiFile(midi_path)

    chosen, candidates = select_melody_track(
        midi=midi,
        strategy=track_strategy,
        default_pitch_bend_range=pitch_bend_range,
    )
    notes = monophonize(chosen.notes)

    if len(notes) < 6:
        raise RuntimeError(
            f"Selected melody track produced only {len(notes)} monophonic notes"
        )

    audio = render_melody(
        midi=midi,
        notes=notes,
        timing=timing,
        fixed_bpm=fixed_bpm,
        gap_ms=gap_ms,
    )

    sf.write(
        wav_path,
        audio,
        SR,
        subtype="PCM_16",
    )

    bends = np.asarray(
        [n.bend_semitones for n in notes],
        dtype=np.float64,
    )

    bent_mask = np.abs(bends) >= 0.01

    return {
        "duration_s": len(audio) / SR,
        "note_events": len(notes),
        "pitch_min_midi": min(
            n.pitch + n.bend_semitones
            for n in notes
        ),
        "pitch_max_midi": max(
            n.pitch + n.bend_semitones
            for n in notes
        ),
        "selected_track_index": chosen.index,
        "selected_track_name": chosen.name,
        "selected_track_score": chosen.score,
        "selected_track_monophony": chosen.monophony,
        "selected_track_mean_pitch": chosen.mean_pitch,
        "pitchwheel_messages": chosen.pitchwheel_messages,
        "bent_note_fraction": float(np.mean(bent_mask)),
        "max_abs_bend_semitones": float(
            np.max(np.abs(bends))
            if bends.size
            else 0.0
        ),
        "track_candidates": [
            {
                "index": c.index,
                "name": c.name,
                "notes": len(c.notes),
                "score": round(c.score, 4),
                "monophony": round(c.monophony, 4),
                "mean_pitch": round(c.mean_pitch, 3),
                "unique_pitches": c.unique_pitches,
                "pitchwheel_messages": c.pitchwheel_messages,
            }
            for c in candidates[:8]
        ],
    }


def analyze_existing(
    midi_path: Path,
    wav_path: Path,
    track_strategy: str,
    pitch_bend_range: float,
) -> dict:
    midi = mido.MidiFile(midi_path)

    chosen, candidates = select_melody_track(
        midi=midi,
        strategy=track_strategy,
        default_pitch_bend_range=pitch_bend_range,
    )
    notes = monophonize(chosen.notes)

    if not notes:
        raise RuntimeError("No usable melody notes")

    existing, existing_sr = sf.read(
        wav_path,
        dtype="float32",
        always_2d=False,
    )

    bends = np.asarray(
        [n.bend_semitones for n in notes],
        dtype=np.float64,
    )
    bent_mask = np.abs(bends) >= 0.01

    return {
        "duration_s": len(existing) / existing_sr,
        "note_events": len(notes),
        "pitch_min_midi": min(
            n.pitch + n.bend_semitones
            for n in notes
        ),
        "pitch_max_midi": max(
            n.pitch + n.bend_semitones
            for n in notes
        ),
        "selected_track_index": chosen.index,
        "selected_track_name": chosen.name,
        "selected_track_score": chosen.score,
        "selected_track_monophony": chosen.monophony,
        "selected_track_mean_pitch": chosen.mean_pitch,
        "pitchwheel_messages": chosen.pitchwheel_messages,
        "bent_note_fraction": float(np.mean(bent_mask)),
        "max_abs_bend_semitones": float(
            np.max(np.abs(bends))
            if bends.size
            else 0.0
        ),
        "track_candidates": [
            {
                "index": c.index,
                "name": c.name,
                "notes": len(c.notes),
                "score": round(c.score, 4),
                "monophony": round(c.monophony, 4),
                "mean_pitch": round(c.mean_pitch, 3),
                "unique_pitches": c.unique_pitches,
                "pitchwheel_messages": c.pitchwheel_messages,
            }
            for c in candidates[:8]
        ],
    }


def output_filename(item: SourceItem) -> str:
    return f"taqasim_{item.global_index:04d}.wav"


def print_source_summary(items: list[SourceItem]) -> None:
    counts = Counter(item.maqam for item in items)

    print("Detected source corpus")
    print("----------------------")
    print(f"Unique MIDI files: {len(items)}")

    for maqam in (*MAQAM_ORDER, "unknown"):
        count = counts.get(maqam, 0)
        if count:
            print(f"{MAQAM_DISPLAY[maqam]:10} {count:4d}")


def write_metadata(output: Path, rows: list[dict]) -> None:
    fields = [
        "filename",
        "id",
        "maqam",
        "maqam_display",
        "source_relative_path",
        "source_sha256",
        "duration_s",
        "note_events",
        "pitch_min_midi",
        "pitch_max_midi",
        "selected_track_index",
        "selected_track_name",
        "selected_track_monophony",
        "pitchwheel_messages",
        "bent_note_fraction",
        "max_abs_bend_semitones",
        "timing",
        "fixed_bpm",
        "source_repository",
        "paper_url",
        "citation",
        "rights",
    ]

    with (output / "metadata.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {key: row.get(key, "") for key in fields}
            )


def write_manifest(
    output: Path,
    archive_path: Path,
    all_items: list[SourceItem],
    rows: list[dict],
    failures: list[dict],
    timing: str,
    fixed_bpm: float,
    track_strategy: str,
    pitch_bend_range: float,
) -> None:
    source_counts = Counter(item.maqam for item in all_items)
    generated_counts = Counter(row["maqam"] for row in rows)

    payload = {
        "dataset": "arabic_taqasim",
        "tradition": "Arab maqam / taqasim",
        "source_repository": REPOSITORY_URL,
        "source_archive": ARCHIVE_URL,
        "source_archive_sha256": sha256_file(archive_path),
        "paper_url": PAPER_URL,
        "citation": CITATION,
        "rights_note": RIGHTS_NOTE,
        "paper_reported_statistics": {
            "improvisations": 717,
            "duration_hours": 22.09,
            "note_count": 631201,
            "mean_note_count": 880.34,
            "maqamat": [
                MAQAM_DISPLAY[m]
                for m in MAQAM_ORDER
            ],
        },
        "source_discovery": {
            "unique_midi_files": len(all_items),
            "maqam_counts": dict(source_counts),
            "label_detection": (
                "Maqam labels are inferred only from archive path/file naming; "
                "metadata labels are not used by Scaleify during audio training."
            ),
        },
        "rendering": {
            "sample_rate": SR,
            "timing": timing,
            "fixed_bpm": (
                fixed_bpm
                if timing == "fixed"
                else None
            ),
            "track_strategy": track_strategy,
            "timbre": "neutral_harmonic_synth",
            "source_instrument_program_used": False,
            "pitch_bend": (
                "Note-on pitch bend is preserved when present. Standard RPN "
                "pitch-bend sensitivity is parsed when available; otherwise "
                f"default range={pitch_bend_range} semitones."
            ),
        },
        "generated_count": len(rows),
        "generated_maqam_counts": dict(generated_counts),
        "songs": rows,
        "failures": failures,
    }

    (output / "manifest.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a neutral monophonic WAV corpus from the public "
            "TaqasimDataset MIDI archive."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory (default: datasets/arabic)",
    )
    parser.add_argument(
        "--maqam",
        choices=(*MAQAM_ORDER, "unknown"),
        default=None,
        help="Generate only one detected maqam subset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render at most N selected files (useful for smoke tests).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Download/extract if needed, then print detected maqam counts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download, re-extract, and re-render source data.",
    )
    parser.add_argument(
        "--track-strategy",
        choices=["auto", "first", "highest"],
        default="auto",
    )
    parser.add_argument(
        "--timing",
        choices=["source", "fixed"],
        default="source",
        help=(
            "source: preserve MIDI tempo map; "
            "fixed: preserve tick ratios at one neutral BPM."
        ),
    )
    parser.add_argument(
        "--render-bpm",
        type=float,
        default=DEFAULT_FIXED_BPM,
        help="BPM used only with --timing fixed.",
    )
    parser.add_argument(
        "--articulation-gap-ms",
        type=float,
        default=DEFAULT_GAP_MS,
    )
    parser.add_argument(
        "--pitch-bend-range",
        type=float,
        default=DEFAULT_PITCH_BEND_RANGE,
        help=(
            "Fallback pitch-wheel range in semitones when MIDI does not "
            "declare RPN pitch-bend sensitivity."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")

    if args.render_bpm <= 0:
        parser.error("--render-bpm must be > 0")

    if args.pitch_bend_range <= 0:
        parser.error("--pitch-bend-range must be > 0")

    output = args.output
    source_root = output / "_source"
    source_midi_dir = output / "_source_midi"
    archive_path = source_root / "All taqasim MIDI files.zip"

    output.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Scaleify research corpus generator/1.0 "
                "(TaqasimDataset local research builder)"
            )
        }
    )

    download_archive(
        session=session,
        archive_path=archive_path,
        timeout=args.timeout,
        force=args.force,
    )

    safe_extract_zip(
        archive_path=archive_path,
        destination=source_midi_dir,
        force=args.force,
    )

    all_items = discover_source_items(source_midi_dir)

    if not all_items:
        raise SystemExit(
            f"No MIDI files found after extraction: {source_midi_dir}"
        )

    print_source_summary(all_items)

    if args.list:
        return

    selected = all_items

    if args.maqam is not None:
        selected = [
            item
            for item in selected
            if item.maqam == args.maqam
        ]

    if args.limit is not None:
        selected = selected[: args.limit]

    if not selected:
        raise SystemExit("No source MIDI files match the requested selection.")

    if len(all_items) != 717:
        print()
        print(
            "WARNING: The paper reports 717 improvisations, but "
            f"{len(all_items)} unique MIDI files were discovered."
        )
        print(
            "         Check manifest/source layout before using this as a "
            "paper-reproduction corpus."
        )

    unknown_count = sum(
        item.maqam == "unknown"
        for item in all_items
    )

    if unknown_count:
        print()
        print(
            f"WARNING: {unknown_count} source file(s) could not be assigned "
            "a maqam label from archive naming."
        )
        print(
            "         They are still rendered unless --maqam filters them out."
        )

    print()
    print(
        f"Rendering {len(selected)} file(s) -> {output}"
    )

    rows: list[dict] = []
    failures: list[dict] = []

    for i, item in enumerate(selected, start=1):
        wav_name = output_filename(item)
        wav_path = output / wav_name

        print(
            f"[{i:03d}/{len(selected):03d}] "
            f"{MAQAM_DISPLAY[item.maqam]:9} "
            f"{item.relative_path}"
        )

        try:
            if args.force or not wav_path.exists():
                analysis = analyze_and_render(
                    midi_path=item.path,
                    wav_path=wav_path,
                    track_strategy=args.track_strategy,
                    timing=args.timing,
                    fixed_bpm=args.render_bpm,
                    gap_ms=args.articulation_gap_ms,
                    pitch_bend_range=args.pitch_bend_range,
                )
            else:
                analysis = analyze_existing(
                    midi_path=item.path,
                    wav_path=wav_path,
                    track_strategy=args.track_strategy,
                    pitch_bend_range=args.pitch_bend_range,
                )

            row = {
                "filename": wav_name,
                "id": f"taqasim_{item.global_index:04d}",
                "maqam": item.maqam,
                "maqam_display": MAQAM_DISPLAY[item.maqam],
                "source_relative_path": item.relative_path,
                "source_sha256": item.sha256,
                "duration_s": round(
                    float(analysis["duration_s"]),
                    3,
                ),
                "note_events": int(
                    analysis["note_events"]
                ),
                "pitch_min_midi": round(
                    float(analysis["pitch_min_midi"]),
                    3,
                ),
                "pitch_max_midi": round(
                    float(analysis["pitch_max_midi"]),
                    3,
                ),
                "selected_track_index": int(
                    analysis["selected_track_index"]
                ),
                "selected_track_name": (
                    analysis["selected_track_name"]
                ),
                "selected_track_monophony": round(
                    float(
                        analysis[
                            "selected_track_monophony"
                        ]
                    ),
                    4,
                ),
                "pitchwheel_messages": int(
                    analysis["pitchwheel_messages"]
                ),
                "bent_note_fraction": round(
                    float(
                        analysis["bent_note_fraction"]
                    ),
                    6,
                ),
                "max_abs_bend_semitones": round(
                    float(
                        analysis[
                            "max_abs_bend_semitones"
                        ]
                    ),
                    4,
                ),
                "timing": args.timing,
                "fixed_bpm": (
                    args.render_bpm
                    if args.timing == "fixed"
                    else ""
                ),
                "source_repository": REPOSITORY_URL,
                "paper_url": PAPER_URL,
                "citation": CITATION,
                "rights": RIGHTS_NOTE,
                "track_candidates": (
                    analysis["track_candidates"]
                ),
            }

            rows.append(row)

            print(
                f"    track={analysis['selected_track_index']} "
                f"mono={analysis['selected_track_monophony']:.3f} "
                f"notes={analysis['note_events']} "
                f"bend={analysis['bent_note_fraction']:.3f} "
                f"duration={analysis['duration_s']:.2f}s "
                f"-> {wav_name}"
            )

        except Exception as exc:
            failure = {
                "id": f"taqasim_{item.global_index:04d}",
                "maqam": item.maqam,
                "source_relative_path": item.relative_path,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(
                f"    [FAILED] {failure['error']}"
            )

        if args.request_delay > 0:
            time.sleep(args.request_delay)

    write_metadata(
        output=output,
        rows=rows,
    )

    write_manifest(
        output=output,
        archive_path=archive_path,
        all_items=all_items,
        rows=rows,
        failures=failures,
        timing=args.timing,
        fixed_bpm=args.render_bpm,
        track_strategy=args.track_strategy,
        pitch_bend_range=args.pitch_bend_range,
    )

    failure_path = output / "failures.json"

    if failures:
        failure_path.write_text(
            json.dumps(
                failures,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    elif failure_path.exists():
        failure_path.unlink()

    print()
    print("Generation complete")
    print("-------------------")
    print(f"Generated: {len(rows)}/{len(selected)}")
    print(f"Output:    {output}")
    print(f"Metadata:  {output / 'metadata.csv'}")
    print(f"Manifest:  {output / 'manifest.json'}")

    bent_files = sum(
        float(row["bent_note_fraction"]) > 0
        for row in rows
    )
    if bent_files:
        print(
            f"Pitch bend: detected in {bent_files}/{len(rows)} "
            "generated file(s)"
        )

    if failures:
        print(
            f"Failures:  {len(failures)} "
            f"-> {failure_path}"
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()