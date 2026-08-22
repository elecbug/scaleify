from pathlib import Path

import numpy as np
import soundfile as sf

sr = 44100
bpm = 124
beat = 60 / bpm

# Deliberately varied monophonic melody:
# - wide range C4~C6
# - all 12 pitch classes appear
# - semitone motion, whole-tone motion, thirds, fourths, fifths
# - ascending/descending and repeated motifs
sequence = [
    # Phrase 1: diatonic with wider contour
    ("C4", 0.5), ("E4", 0.5), ("G4", 0.5), ("B4", 0.5),
    ("A4", 0.5), ("F4", 0.5), ("D4", 0.5), ("G4", 0.5),
    ("C5", 1.0), ("B4", 0.5), ("A4", 0.5), ("G4", 1.0),

    # Phrase 2: chromatic ascent/descent
    ("C4", 0.5), ("C#4", 0.5), ("D4", 0.5), ("D#4", 0.5),
    ("E4", 0.5), ("F4", 0.5), ("F#4", 0.5), ("G4", 0.5),
    ("G#4", 0.5), ("A4", 0.5), ("A#4", 0.5), ("B4", 0.5),
    ("C5", 1.0),
    ("B4", 0.5), ("A#4", 0.5), ("A4", 0.5), ("G#4", 0.5),
    ("G4", 0.5), ("F#4", 0.5), ("F4", 0.5), ("E4", 0.5),

    # Phrase 3: interval stress test
    ("C4", 0.5), ("G4", 0.5), ("D5", 0.5), ("A4", 0.5),
    ("E5", 0.5), ("B4", 0.5), ("F5", 0.5), ("C5", 0.5),
    ("G5", 1.0), ("D5", 0.5), ("A4", 0.5), ("E5", 1.0),

    # Phrase 4: semitone-heavy expressive line
    ("E4", 0.75), ("F4", 0.25), ("F#4", 0.75), ("G4", 0.25),
    ("G#4", 0.75), ("A4", 0.25), ("A#4", 0.75), ("B4", 0.25),
    ("C5", 0.5), ("D#5", 0.5), ("F#5", 0.5), ("A5", 0.5),
    ("C6", 1.5),

    # Phrase 5: closing motif, mixed steps
    ("A5", 0.5), ("F5", 0.5), ("D#5", 0.5), ("B4", 0.5),
    ("G#4", 0.5), ("F4", 0.5), ("D4", 0.5), ("C#4", 0.5),
    ("C4", 2.0),
]

NOTE_TO_SEMITONE = {
    "C":0, "C#":1, "D":2, "D#":3, "E":4, "F":5,
    "F#":6, "G":7, "G#":8, "A":9, "A#":10, "B":11
}

def note_to_freq(note):
    if len(note) == 2:
        name, octave = note[0], int(note[1])
    else:
        name, octave = note[:2], int(note[2])
    midi = 12 * (octave + 1) + NOTE_TO_SEMITONE[name]
    return 440.0 * 2 ** ((midi - 69) / 12)

def synth_note(freq, duration):
    n = int(sr * duration)
    t = np.arange(n) / sr

    # richer reed/string-like timbre, still clean enough for pYIN
    y = (
        0.68 * np.sin(2*np.pi*freq*t)
        + 0.20 * np.sin(2*np.pi*2*freq*t)
        + 0.08 * np.sin(2*np.pi*3*freq*t)
        + 0.04 * np.sin(2*np.pi*4*freq*t)
    )

    attack = max(1, int(0.012 * sr))
    release = max(1, int(0.06 * sr))
    env = np.ones(n, dtype=np.float32)
    env[:attack] = np.linspace(0, 1, attack, endpoint=False)
    env *= np.exp(-0.55 * t / max(duration, 1e-6))
    if release < n:
        env[-release:] *= np.linspace(1, 0, release)

    # very slight vibrato only after attack
    vib = 0.003 * np.sin(2*np.pi*5.2*t)
    y = (
        0.68 * np.sin(2*np.pi*freq*t + vib)
        + 0.20 * np.sin(2*np.pi*2*freq*t + 0.5*vib)
        + 0.08 * np.sin(2*np.pi*3*freq*t)
        + 0.04 * np.sin(2*np.pi*4*freq*t)
    )
    return y * env

parts = []
gap_s = 0.018

for note, beats in sequence:
    dur = beat * beats
    tone_dur = max(0.04, dur - gap_s)
    parts.append(synth_note(note_to_freq(note), tone_dur))
    parts.append(np.zeros(int(sr * gap_s), dtype=np.float32))

audio = np.concatenate(parts)
audio /= max(np.max(np.abs(audio)), 1e-9)
audio *= 0.82

out = Path("test/test_audio.wav")
sf.write(out, audio.astype(np.float32), sr, subtype="PCM_16")

print(out)
print(f"Duration: {len(audio)/sr:.2f} s")
