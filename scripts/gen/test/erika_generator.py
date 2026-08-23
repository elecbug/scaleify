from pathlib import Path

import numpy as np
import soundfile as sf


# ============================================================
# Erika test generator
#
# Composer: Herms Niel (1888-1954)
# Public-domain melody in life+70-or-less jurisdictions.
#
# This is a newly synthesized test rendering:
# - G major
# - 2/4
# - A-A-B-A form
# - no copied historical recording or commercial arrangement
#
# Default output is monophonic for scale-conversion testing.
# ============================================================

SR = 44100
BPM = 120
BEAT = 60.0 / BPM

OUTPUT = Path("test/erika_test.wav")

# Keep these False for clean --no-demucs scale tests.
ADD_STOMP = False
ADD_BASS = False

NOTE_GAP_S = 0.015


NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1,
    "D": 2, "D#": 3,
    "E": 4,
    "F": 5, "F#": 6,
    "G": 7, "G#": 8,
    "A": 9, "A#": 10,
    "B": 11,
}


# ============================================================
# Melody
#
# Source form: G major, 2/4.
#
# Event tuple:
#   (note, quarter-note beats, velocity)
#
# LilyPond values:
#   4. = dotted quarter = 1.5 beats
#   8  = eighth         = 0.5 beats
#   4  = quarter        = 1.0 beat
#   s  = rest/space
# ============================================================

A_SECTION = [
    # b4. c8 | d4 d
    ("B4", 1.5, 0.95), ("C5", 0.5, 0.85),
    ("D5", 1.0, 0.95), ("D5", 1.0, 0.90),

    # d4 g | g b
    ("D5", 1.0, 0.92), ("G5", 1.0, 1.00),
    ("G5", 1.0, 0.95), ("B5", 1.0, 1.00),

    # b4. a8 | g4 rest
    ("B5", 1.5, 1.00), ("A5", 0.5, 0.90),
    ("G5", 1.0, 1.00), ("R", 1.0, 0.0),

    # full rest bar
    ("R", 2.0, 0.0),

    # fis4 g | a4 rest
    ("F#5", 1.0, 0.90), ("G5", 1.0, 0.95),
    ("A5", 1.0, 1.00), ("R", 1.0, 0.0),

    # full rest bar
    ("R", 2.0, 0.0),

    # b4. a8 | g4 rest
    ("B5", 1.5, 1.00), ("A5", 0.5, 0.90),
    ("G5", 1.0, 1.00), ("R", 1.0, 0.0),

    # full rest bar
    ("R", 2.0, 0.0),
]


B_SECTION = [
    # d4. g8 | fis4 fis
    ("D5", 1.5, 0.90), ("G5", 0.5, 0.95),
    ("F#5", 1.0, 0.95), ("F#5", 1.0, 0.90),

    # fis4 fis | e4 fis
    ("F#5", 1.0, 0.92), ("F#5", 1.0, 0.90),
    ("E5", 1.0, 0.88), ("F#5", 1.0, 0.95),

    # g4 rest | full rest
    ("G5", 1.0, 1.00), ("R", 1.0, 0.0),
    ("R", 2.0, 0.0),

    # fis4. g8 | a4 a
    ("F#5", 1.5, 0.90), ("G5", 0.5, 0.95),
    ("A5", 1.0, 0.98), ("A5", 1.0, 0.95),

    # a4 a | d4. c8
    ("A5", 1.0, 0.95), ("A5", 1.0, 0.92),
    ("D6", 1.5, 1.00), ("C6", 0.5, 0.92),

    # b4 rest | full rest
    ("B5", 1.0, 1.00), ("R", 1.0, 0.0),
    ("R", 2.0, 0.0),
]


# Classical song form from the score: A, A, B, then return to A.
SEQUENCE = A_SECTION + A_SECTION + B_SECTION + A_SECTION


# ============================================================
# Helpers
# ============================================================

def note_to_midi(note: str) -> int:
    if note == "R":
        raise ValueError("Rest has no pitch")

    if len(note) >= 3 and note[1] == "#":
        name = note[:2]
        octave = int(note[2:])
    else:
        name = note[0]
        octave = int(note[1:])

    return 12 * (octave + 1) + NOTE_TO_SEMITONE[name]


def note_to_freq(note: str) -> float:
    midi = note_to_midi(note)
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(
        max(1, int(seconds * SR)),
        dtype=np.float32,
    )


# ============================================================
# Lead synthesis
# ============================================================

def synth_lead(
    freq: float,
    duration: float,
    velocity: float,
) -> np.ndarray:
    n = max(1, int(duration * SR))
    t = np.arange(n, dtype=np.float64) / SR

    # Clean brass/reed-like synthetic lead.
    # Fundamental remains dominant for YIN/pYIN.
    vibrato_cents = (
        4.0
        * np.sin(2.0 * np.pi * 5.2 * t)
    )

    inst_freq = (
        freq
        * 2.0 ** (vibrato_cents / 1200.0)
    )

    phase = (
        2.0
        * np.pi
        * np.cumsum(inst_freq)
        / SR
    )

    y = (
        0.70 * np.sin(phase)
        + 0.17 * np.sin(2.0 * phase)
        + 0.08 * np.sin(3.0 * phase)
        + 0.05 * np.sin(4.0 * phase)
    )

    attack = min(
        n,
        max(1, int(0.010 * SR)),
    )

    release = min(
        n,
        max(1, int(0.055 * SR)),
    )

    env = np.ones(n, dtype=np.float64)

    if attack > 1:
        env[:attack] = np.linspace(
            0.0,
            1.0,
            attack,
            endpoint=False,
        )

    env *= np.exp(
        -0.22 * t / max(duration, 1e-6)
    )

    if release > 1:
        env[-release:] *= np.linspace(
            1.0,
            0.0,
            release,
        )

    return (
        y
        * env
        * velocity
    ).astype(np.float32)


# ============================================================
# Optional simple bass
#
# This is an original test accompaniment, not the historical arrangement.
# ============================================================

BASS_PATTERN = [
    "G2", "D3",
    "G2", "D3",
    "C3", "G2",
    "D3", "D2",
]


def synth_bass(
    freq: float,
    duration: float,
) -> np.ndarray:
    n = max(1, int(duration * SR))
    t = np.arange(n, dtype=np.float64) / SR

    y = (
        0.84 * np.sin(2.0 * np.pi * freq * t)
        + 0.16 * np.sin(4.0 * np.pi * freq * t)
    )

    env = np.exp(
        -1.9 * t / max(duration, 1e-6)
    )

    attack = min(
        n,
        max(1, int(0.006 * SR)),
    )

    if attack > 1:
        env[:attack] *= np.linspace(
            0.0,
            1.0,
            attack,
        )

    return (
        y
        * env
        * 0.24
    ).astype(np.float32)


# ============================================================
# Optional neutral stomp/click
#
# Not intended to imitate any specific historical recording.
# ============================================================

RNG = np.random.default_rng(1479)


def synth_stomp(duration=0.07):
    n = int(duration * SR)
    t = np.arange(n, dtype=np.float64) / SR

    noise = RNG.normal(
        0.0,
        1.0,
        n,
    )

    low = np.sin(
        2.0
        * np.pi
        * 85.0
        * t
    )

    env = np.exp(-45.0 * t)

    return (
        (0.70 * low + 0.30 * noise)
        * env
        * 0.20
    ).astype(np.float32)


def add_hit(
    track: np.ndarray,
    time_s: float,
    hit: np.ndarray,
) -> None:
    start = int(time_s * SR)

    if start >= len(track):
        return

    end = min(
        len(track),
        start + len(hit),
    )

    track[start:end] += hit[:end - start]


# ============================================================
# Rendering
# ============================================================

def render_lead() -> np.ndarray:
    parts = []

    for note, beats, velocity in SEQUENCE:
        duration = beats * BEAT

        if note == "R":
            parts.append(
                silence(duration)
            )
            continue

        tone_duration = max(
            0.035,
            duration - NOTE_GAP_S,
        )

        parts.append(
            synth_lead(
                note_to_freq(note),
                tone_duration,
                velocity,
            )
        )

        parts.append(
            silence(
                min(
                    NOTE_GAP_S,
                    duration * 0.20,
                )
            )
        )

    return np.concatenate(parts)


def render_bass(duration_s: float) -> np.ndarray:
    track = np.zeros(
        int(duration_s * SR),
        dtype=np.float32,
    )

    # One bass note per quarter beat.
    t = 0.0
    i = 0

    while t < duration_s:
        note = BASS_PATTERN[
            i % len(BASS_PATTERN)
        ]

        add_hit(
            track,
            t,
            synth_bass(
                note_to_freq(note),
                BEAT * 0.82,
            ),
        )

        t += BEAT
        i += 1

    return track


def render_stomp(duration_s: float) -> np.ndarray:
    track = np.zeros(
        int(duration_s * SR),
        dtype=np.float32,
    )

    # Quarter-note pulse to make full-mix separation testable.
    t = 0.0

    while t < duration_s:
        add_hit(
            track,
            t,
            synth_stomp(),
        )
        t += BEAT

    return track


# ============================================================
# Main
# ============================================================

lead = render_lead()
audio = lead.copy()

duration_s = len(lead) / SR

if ADD_BASS:
    bass = render_bass(duration_s)
    n = min(len(audio), len(bass))
    audio = audio[:n] + bass[:n]

if ADD_STOMP:
    stomp = render_stomp(duration_s)
    n = min(len(audio), len(stomp))
    audio = audio[:n] + stomp[:n]


peak = float(np.max(np.abs(audio)))

if peak > 0:
    audio /= peak

audio *= 0.86


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

sf.write(
    OUTPUT,
    audio.astype(np.float32),
    SR,
    subtype="PCM_16",
)

print(OUTPUT)
print(
    f"Duration: {len(audio) / SR:.2f} s | "
    f"BPM: {BPM} | "
    f"Key: G major | "
    f"Form: A-A-B-A | "
    f"Bass: {ADD_BASS} | "
    f"Stomp: {ADD_STOMP}"
)