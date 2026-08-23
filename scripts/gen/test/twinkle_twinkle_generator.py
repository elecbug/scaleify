from pathlib import Path

import numpy as np
import soundfile as sf

sr = 44100
bpm = 132
beat = 60.0 / bpm

sequence = [
    ("C4", 1), ("C4", 1), ("G4", 1), ("G4", 1),
    ("A4", 1), ("A4", 1), ("G4", 2),

    ("F4", 1), ("F4", 1), ("E4", 1), ("E4", 1),
    ("D4", 1), ("D4", 1), ("C4", 2),

    ("G4", 1), ("G4", 1), ("F4", 1), ("F4", 1),
    ("E4", 1), ("E4", 1), ("D4", 2),

    ("G4", 1), ("G4", 1), ("F4", 1), ("F4", 1),
    ("E4", 1), ("E4", 1), ("D4", 2),

    ("C4", 1), ("C4", 1), ("G4", 1), ("G4", 1),
    ("A4", 1), ("A4", 1), ("G4", 2),

    ("F4", 1), ("F4", 1), ("E4", 1), ("E4", 1),
    ("D4", 1), ("D4", 1), ("C4", 2),
]

NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "D": 2, "D#": 3,
    "E": 4, "F": 5, "F#": 6, "G": 7,
    "G#": 8, "A": 9, "A#": 10, "B": 11,
}

def note_to_freq(note):
    if len(note) >= 3 and note[1] == "#":
        name = note[:2]
        octave = int(note[2:])
    else:
        name = note[0]
        octave = int(note[1:])

    midi = 12 * (octave + 1) + NOTE_TO_SEMITONE[name]
    return 440.0 * 2 ** ((midi - 69) / 12)

def synth_note(freq, duration):
    n = max(1, int(sr * duration))
    t = np.arange(n, dtype=np.float64) / sr

    phase = 2 * np.pi * freq * t
    y = (
        0.72 * np.sin(phase)
        + 0.18 * np.sin(2 * phase)
        + 0.07 * np.sin(3 * phase)
        + 0.03 * np.sin(4 * phase)
    )

    attack = min(n, int(0.01 * sr))
    release = min(n, int(0.06 * sr))

    env = np.ones(n, dtype=np.float64)

    if attack > 1:
        env[:attack] = np.linspace(0, 1, attack, endpoint=False)

    env *= np.exp(-0.35 * t / max(duration, 1e-6))

    if release > 1:
        env[-release:] *= np.linspace(1, 0, release)

    return (y * env).astype(np.float32)

gap_s = 0.02
parts = []

for note, beats in sequence:
    duration = beats * beat
    tone_duration = max(0.05, duration - gap_s)

    parts.append(
        synth_note(
            note_to_freq(note),
            tone_duration,
        )
    )

    parts.append(
        np.zeros(
            int(sr * gap_s),
            dtype=np.float32,
        )
    )

audio = np.concatenate(parts)

audio /= max(
    float(np.max(np.abs(audio))),
    1e-9,
)

audio *= 0.82

out = Path("results/twinkle_twinkle_test.wav")
out.parent.mkdir(parents=True, exist_ok=True)

sf.write(
    out,
    audio,
    sr,
    subtype="PCM_16",
)

print(out)
print(
    f"Duration: {len(audio) / sr:.2f} s | "
    f"BPM: {bpm} | mono PCM16"
)