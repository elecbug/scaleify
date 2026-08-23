from pathlib import Path

import numpy as np
import soundfile as sf

# ============================================================
# Korobeiniki test generator
#
# Traditional Russian folk melody commonly associated with Tetris.
# This generator intentionally uses its own synthesis / accompaniment
# instead of copying a specific commercial game arrangement.
# ============================================================

SR = 44100
BPM = 152
BEAT = 60.0 / BPM

OUTPUT = Path("results/korobeiniki_test.wav")

# For pure scale-conversion testing:
ADD_BASS = False
ADD_DRUMS = False

# Turn these on for Demucs/full-mix testing.
# ADD_BASS = True
# ADD_DRUMS = True

REPEATS = 2
NOTE_GAP_S = 0.012


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
# Traditional Korobeiniki melody in A minor.
#
# Tuple:
#   (note, beat length, velocity)
#
# The rhythm/phrasing below is our own compact test rendering,
# not a transcription of a specific Tetris arrangement.
# ============================================================

A_SECTION = [
    ("E5", 1.0, 1.00),
    ("B4", 0.5, 0.85),
    ("C5", 0.5, 0.90),
    ("D5", 1.0, 0.95),
    ("C5", 0.5, 0.90),
    ("B4", 0.5, 0.85),

    ("A4", 1.0, 1.00),
    ("A4", 0.5, 0.75),
    ("C5", 0.5, 0.90),
    ("E5", 1.0, 1.00),
    ("D5", 0.5, 0.90),
    ("C5", 0.5, 0.85),

    ("B4", 1.0, 0.95),
    ("B4", 0.5, 0.75),
    ("C5", 0.5, 0.90),
    ("D5", 1.0, 0.95),
    ("E5", 1.0, 1.00),

    ("C5", 1.0, 0.90),
    ("A4", 1.0, 1.00),
    ("A4", 1.0, 0.80),
    ("R", 1.0, 0.0),
]


B_SECTION = [
    ("D5", 1.0, 0.95),
    ("F5", 0.5, 0.90),
    ("A5", 1.0, 1.00),
    ("G5", 0.5, 0.90),
    ("F5", 0.5, 0.85),
    ("E5", 0.5, 0.90),

    ("C5", 1.0, 0.90),
    ("E5", 0.5, 0.95),
    ("D5", 0.5, 0.90),
    ("C5", 0.5, 0.85),
    ("B4", 1.0, 0.90),
    ("B4", 0.5, 0.75),

    ("C5", 0.5, 0.90),
    ("D5", 1.0, 0.95),
    ("E5", 1.0, 1.00),
    ("C5", 1.0, 0.90),

    ("A4", 1.0, 1.00),
    ("A4", 1.0, 0.80),
    ("R", 1.0, 0.0),
]


# A higher-energy variation:
# same traditional melodic material, but with shorter repeated values.
C_SECTION = [
    ("E5", 0.5, 1.00),
    ("E5", 0.5, 0.82),
    ("B4", 0.5, 0.88),
    ("C5", 0.5, 0.92),

    ("D5", 0.5, 0.96),
    ("D5", 0.5, 0.82),
    ("C5", 0.5, 0.90),
    ("B4", 0.5, 0.88),

    ("A4", 0.5, 1.00),
    ("C5", 0.5, 0.90),
    ("E5", 0.5, 1.00),
    ("A5", 0.5, 1.00),

    ("G5", 0.5, 0.92),
    ("F5", 0.5, 0.90),
    ("E5", 0.5, 0.95),
    ("D5", 0.5, 0.90),

    ("C5", 0.5, 0.90),
    ("B4", 0.5, 0.88),
    ("C5", 0.5, 0.90),
    ("D5", 0.5, 0.95),

    ("E5", 0.5, 1.00),
    ("C5", 0.5, 0.90),
    ("A4", 1.0, 1.00),
    ("R", 1.0, 0.0),
]


SEQUENCE = (A_SECTION + B_SECTION + C_SECTION) * REPEATS


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
# Melody synthesizer
# ============================================================

def synth_lead(
    freq: float,
    duration: float,
    velocity: float,
) -> np.ndarray:
    n = max(1, int(SR * duration))
    t = np.arange(n, dtype=np.float64) / SR

    # Bright folk/game-like lead while keeping the fundamental strong
    # enough for YIN/pYIN tracking.
    vibrato_cents = 5.0 * np.sin(
        2.0 * np.pi * 5.5 * t
    )

    inst_freq = freq * (
        2.0 ** (vibrato_cents / 1200.0)
    )

    phase = (
        2.0
        * np.pi
        * np.cumsum(inst_freq)
        / SR
    )

    y = (
        0.68 * np.sin(phase)
        + 0.18 * np.sin(2.0 * phase)
        + 0.09 * np.sin(3.0 * phase)
        + 0.05 * np.sin(4.0 * phase)
    )

    attack = min(
        n,
        max(1, int(0.008 * SR)),
    )

    release = min(
        n,
        max(1, int(0.05 * SR)),
    )

    env = np.ones(n, dtype=np.float64)

    env[:attack] = np.linspace(
        0.0,
        1.0,
        attack,
        endpoint=False,
    )

    env *= np.exp(
        -0.30 * t / max(duration, 1e-6)
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
# Bass
# ============================================================

BASS_PATTERN = [
    "A2", "E3", "A2", "E3",
    "F2", "C3", "G2", "E3",
]


def synth_bass(
    freq: float,
    duration: float,
) -> np.ndarray:
    n = max(1, int(duration * SR))
    t = np.arange(n) / SR

    y = (
        0.82 * np.sin(2 * np.pi * freq * t)
        + 0.18 * np.sin(4 * np.pi * freq * t)
    )

    env = np.exp(-2.0 * t / max(duration, 1e-6))

    attack = min(
        n,
        int(0.008 * SR),
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
        * 0.30
    ).astype(np.float32)


# ============================================================
# Simple percussion
# ============================================================

RNG = np.random.default_rng(1479)


def kick(duration=0.12):
    n = int(duration * SR)
    t = np.arange(n) / SR

    freq = 125.0 * np.exp(-24.0 * t) + 42.0
    phase = 2 * np.pi * np.cumsum(freq) / SR

    return (
        np.sin(phase)
        * np.exp(-28.0 * t)
        * 0.62
    ).astype(np.float32)


def snare(duration=0.11):
    n = int(duration * SR)
    t = np.arange(n) / SR

    noise = RNG.normal(0, 1, n)
    env = np.exp(-30.0 * t)

    return (
        noise
        * env
        * 0.20
    ).astype(np.float32)


def hat(duration=0.035):
    n = int(duration * SR)
    t = np.arange(n) / SR

    noise = RNG.normal(0, 1, n)

    return (
        noise
        * np.exp(-85.0 * t)
        * 0.075
    ).astype(np.float32)


def add_hit(
    track: np.ndarray,
    time_s: float,
    hit: np.ndarray,
):
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

def render_lead():
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


def render_bass(duration_s: float):
    track = np.zeros(
        int(duration_s * SR),
        dtype=np.float32,
    )

    beat_index = 0
    t = 0.0

    while t < duration_s:
        note = BASS_PATTERN[
            beat_index % len(BASS_PATTERN)
        ]

        note_audio = synth_bass(
            note_to_freq(note),
            BEAT * 0.82,
        )

        add_hit(
            track,
            t,
            note_audio,
        )

        beat_index += 1
        t += BEAT

    return track


def render_drums(duration_s: float):
    track = np.zeros(
        int(duration_s * SR),
        dtype=np.float32,
    )

    total_eighths = int(
        np.ceil(duration_s / (BEAT / 2))
    )

    for i in range(total_eighths):
        t = i * BEAT / 2

        # eighth-note hats
        add_hit(
            track,
            t,
            hat(),
        )

        # quarter-note index
        if i % 2 == 0:
            quarter = i // 2
            beat_in_bar = quarter % 4

            if beat_in_bar in (0, 2):
                add_hit(
                    track,
                    t,
                    kick(),
                )

            if beat_in_bar in (1, 3):
                add_hit(
                    track,
                    t,
                    snare(),
                )

    return track


# ============================================================
# Main
# ============================================================

lead = render_lead()

audio = lead.copy()
duration_s = len(audio) / SR

if ADD_BASS:
    bass = render_bass(duration_s)
    n = min(len(audio), len(bass))
    audio = audio[:n] + bass[:n]

if ADD_DRUMS:
    drums = render_drums(duration_s)
    n = min(len(audio), len(drums))
    audio = audio[:n] + drums[:n]


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
    f"Bass: {ADD_BASS} | "
    f"Drums: {ADD_DRUMS}"
)