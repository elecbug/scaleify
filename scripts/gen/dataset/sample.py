#!/usr/bin/env python3
"""
public_domain_corpus_generator.py

Generate a small seed corpus of newly synthesized monophonic WAV files from
old/traditional melodies whose underlying compositions are in the public domain
or are anonymous traditional tunes.

IMPORTANT COPYRIGHT NOTE
------------------------
This script does NOT copy commercial recordings or modern arrangements.
It encodes only simple monophonic melody transcriptions and synthesizes a new
neutral performance.

A public-domain underlying composition does not automatically make every modern
score edition, arrangement, engraving, recording, or performance public domain.
The source references below are used to identify/transcribe the old melody; this
generator intentionally omits modern accompaniment/arrangement details.

Purpose
-------
The output is designed for Scaleify corpus tuning:
- monophonic
- neutral timbre
- explicit attacks/re-attacks
- no accompaniment
- no expressive style simulation that would bias the learned profile

Usage
-----
List melodies:
    python public_domain_corpus_generator.py --list

Generate all:
    python public_domain_corpus_generator.py --all --output corpus_seed

Generate one:
    python public_domain_corpus_generator.py --song sakura_sakura --output corpus_seed

Dependencies:
    numpy
    soundfile
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


SR = 44100
NOTE_GAP_S = 0.025


@dataclass(frozen=True)
class Tune:
    id: str
    title: str
    country: str
    tradition: str
    style_hint: str
    bpm: float
    root: str
    meter: str
    source_url: str
    rights_note: str
    transcription_note: str
    events: tuple[tuple[str, float], ...]


def E(note: str, beats: float) -> tuple[str, float]:
    return (note, float(beats))


# ---------------------------------------------------------------------------
# 1) Japan — Sakura Sakura
# ABC source: anonymous/traditional, Am, common time, L=1/4.
# The source identifies it as an anonymous Japanese traditional melody.
# ---------------------------------------------------------------------------
SAKURA = (
    E("A4",1), E("A4",1), E("B4",2),
    E("A4",1), E("A4",1), E("B4",2),
    E("A4",1), E("B4",1), E("C5",1), E("B4",1),
    E("A4",1), E("B4",0.5), E("A4",0.5), E("F4",2),

    E("E4",1), E("C4",1), E("E4",1), E("F4",1),
    E("E4",1), E("E4",0.5), E("C4",0.5), E("B3",2),
    E("A4",1), E("B4",1), E("C5",1), E("B4",1),
    E("A4",1), E("B4",0.5), E("A4",0.5), E("F4",2),

    E("E4",1), E("C4",1), E("E4",1), E("F4",1),
    E("E4",1), E("E4",0.5), E("C4",0.5), E("B3",2),
    E("A4",1), E("A4",1), E("B4",2),
    E("A4",1), E("A4",1), E("B4",2),

    E("E4",1), E("F4",1), E("B4",0.5), E("A4",0.5), E("F4",1),
    E("E4",2), E("R",2),
)


# ---------------------------------------------------------------------------
# 2) China — Mo Li Hua
# Melody follows the widely published C-major LilyPond melody.
# Durations are represented in quarter-note units.
# ---------------------------------------------------------------------------
MOLIHUA_OPENING = (
    E("E4",1), E("E4",0.5), E("G4",0.5),
    E("A4",0.5), E("C5",0.5), E("C5",0.5), E("A4",0.5),
    E("G4",1), E("G4",0.5), E("A4",0.5), E("G4",1), E("R",1),
)
MOLIHUA = (
    *MOLIHUA_OPENING,
    *MOLIHUA_OPENING,

    E("G4",1), E("G4",1), E("G4",1), E("E4",0.5), E("G4",0.5),
    E("A4",1), E("A4",1), E("G4",2),

    E("E4",1), E("D4",0.5), E("E4",0.5),
    E("G4",1), E("E4",0.5), E("D4",0.5),
    E("C4",1), E("C4",0.5), E("D4",0.5), E("C4",2),

    E("E4",0.5), E("D4",0.5), E("C4",0.5), E("E4",0.5),
    E("D4",1.5), E("E4",0.5),
    E("G4",1), E("A4",0.5), E("C5",0.5), E("G4",2),

    E("D4",1), E("E4",0.5), E("G4",0.5),
    E("D4",0.5), E("E4",0.5), E("C4",0.5), E("A3",0.5),
    E("G3",2), E("A3",1), E("C4",1),

    E("D4",1.5), E("E4",0.5),
    E("C4",0.5), E("D4",0.5), E("C4",0.5), E("A3",0.5),
    E("G3",2), E("R",2),
)


# ---------------------------------------------------------------------------
# 3) Korea — Arirang
# Based on a traditional G-major ABC transcription, 3/4, L=1/8.
# Triplets are expanded into equal thirds of a quarter note.
# ---------------------------------------------------------------------------
T = 1.0 / 3.0
ARIRANG_A = (
    E("D4",1.5), E("E4",0.5), E("D4",0.5), E("E4",0.5),
    E("G4",1.5), E("A4",0.5), E("G4",0.5), E("A4",0.5),
    E("B4",1), E("A4",T), E("B4",T), E("A4",T), E("G4",0.5), E("E4",0.5),
    E("D4",1.5), E("E4",0.5), E("D4",0.5), E("E4",0.5),

    E("G4",1.5), E("A4",0.5), E("G4",0.5), E("A4",0.5),
    E("B4",0.5), E("A4",0.5), E("G4",0.5), E("E4",0.5), E("D4",0.5), E("E4",0.5),
    E("G4",1.5), E("A4",0.5), E("G4",1),
    E("G4",3),
)
ARIRANG_B = (
    E("D5",2), E("D5",1),
    E("D5",1), E("B4",1), E("A4",1),
    E("B4",1), E("A4",T), E("B4",T), E("A4",T), E("G4",0.5), E("E4",0.5),
    E("D4",1.5), E("E4",0.5), E("D4",0.5), E("E4",0.5),

    E("G4",1.5), E("A4",0.5), E("G4",0.5), E("A4",0.5),
    E("B4",0.5), E("A4",0.5), E("G4",0.5), E("E4",0.5), E("D4",0.5), E("E4",0.5),
    E("G4",1.5), E("A4",0.5), E("G4",1),
    E("G4",3),
)
ARIRANG = (*ARIRANG_A, *ARIRANG_B)


# ---------------------------------------------------------------------------
# 4) Ireland — Londonderry Air
# Follows the anonymous melody printed in George Petrie's 1855 collection,
# transcribed here as a neutral Eb-major monophonic line.
# ---------------------------------------------------------------------------
LONDONDERRY = (
    # pickup
    E("D4",0.5), E("Eb4",0.5), E("F4",0.5),

    E("G4",1.5), E("F4",0.5), E("G4",0.5), E("C5",0.5), E("Bb4",0.5), E("G4",0.5),
    E("F4",0.5), E("Eb4",0.5), E("C4",1), E("R",0.5), E("Eb4",0.5), E("G4",0.5), E("Ab4",0.5),
    E("Bb4",1.5), E("C5",0.5), E("Bb4",0.5), E("G4",0.5), E("Eb4",0.5), E("G4",0.5),
    E("F4",2), E("R",0.5), E("D4",0.5), E("Eb4",0.5), E("F4",0.5),

    E("G4",1.5), E("F4",0.5), E("G4",0.5), E("C5",0.5), E("Bb4",0.5), E("G4",0.5),
    E("F4",0.5), E("Eb4",0.5), E("C4",0.5), E("B3",0.5),
    E("C4",0.5), E("D4",0.5), E("Eb4",0.5), E("F4",0.5),
    E("G4",1.5), E("Ab4",0.5), E("G4",0.5), E("F4",0.5), E("Eb4",0.5), E("F4",0.5),
    E("Eb4",2), E("R",0.5), E("Bb4",0.5), E("C5",0.5), E("D5",0.5),

    E("Eb5",1.5), E("D5",0.5), E("D5",0.5), E("C5",0.5), E("Bb4",0.5), E("G4",0.5),
    E("Bb4",0.5), E("G4",0.5), E("Eb4",1), E("R",0.5), E("Bb4",0.5), E("C5",0.5), E("D5",0.5),
    E("Eb5",1.5), E("D5",0.5), E("D5",0.5), E("C5",0.5), E("Bb4",0.5), E("G4",0.5),
    E("F4",2), E("R",0.5), E("Bb4",0.5), E("Bb4",0.5), E("Bb4",0.5),

    E("G5",1.5), E("F5",0.5), E("F5",0.5), E("Eb5",0.5), E("C5",0.5), E("Eb5",0.5),
    E("Bb4",0.5), E("G4",0.5), E("Eb4",1), E("R",0.5), E("D4",0.5), E("Eb4",0.5), E("F4",0.5),
    E("G4",0.5), E("C5",0.5), E("Bb4",0.5), E("G4",0.5),
    E("F4",0.5), E("Eb4",0.5), E("C4",0.5), E("D4",0.5),
    E("Eb4",2.5), E("R",1.5),
)


# ---------------------------------------------------------------------------
# 5) Sweden — Hårgalåten
# Anonymous/traditional Swedish hambo. We use the A and B sections from an
# Am ABC version. Broken rhythms e>f are encoded as 0.75 + 0.25 beats.
# ---------------------------------------------------------------------------
def horga_a(last_rest: bool = False):
    end = (E("A4",1), E("R",2)) if last_rest else (
        E("A4",0.75), E("G#4",0.25), E("A4",0.5), E("B4",0.5), E("C5",0.5), E("D5",0.5)
    )
    return (
        E("E5",1), E("E5",0.75), E("F5",0.25), E("E5",0.75), E("D5",0.25),
        E("D5",0.75), E("C5",0.25), E("C5",1), E("A4",0.75), E("C5",0.25),
        E("C5",0.75), E("B4",0.25), E("B4",0.5), E("E4",0.5), E("G#4",0.5), E("B4",0.5),
        *end,
    )

HORGA_B = (
    E("C5",0.75), E("D5",0.25), E("E5",1), E("C5",1),
    E("D5",0.75), E("E5",0.25), E("F5",1), E("D5",1),
    E("G5",1), E("G5",0.75), E("A5",0.25), E("G5",0.75), E("F5",0.25),
    E("F5",0.75), E("E5",0.25), E("E5",2),
)
HORGALATEN = (
    *horga_a(False),
    *horga_a(True),
    *HORGA_B,
    *HORGA_B,
    E("R",1),
)


# ---------------------------------------------------------------------------
# 6) Hungary — Tavaszi szél vizet áraszt
# Traditional Hungarian children's/folk song, D minor ABC transcription.
# ---------------------------------------------------------------------------
TAVASZI = (
    E("F4",1), E("G4",1), E("A4",1), E("A4",1),
    E("G4",1), E("G4",0.5), E("A4",0.5), E("F4",1), E("G4",1),
    E("A4",0.5), E("A4",1.5), E("G4",1), E("G4",0.5), E("A4",0.5),
    E("F4",2), E("C4",2),

    E("F4",1), E("G4",1), E("A4",0.5), E("A4",1.5),
    E("G4",1), E("G4",0.5), E("A4",0.5), E("F4",0.5), E("E4",0.5), E("D4",1),
    E("G4",0.5), E("G4",1), E("A4",0.5), E("F4",1), E("E4",1),
    E("D4",2), E("D4",2),
)


# ---------------------------------------------------------------------------
# 7) Spain — El Vito
# Traditional Andalusian melody. This is deliberately a simplified monophonic
# pitch-skeleton rendering, not a copy of a modern piano/guitar arrangement.
# It is included as a seed only; use multiple independent traditional sources
# before treating it as a statistical corpus.
# ---------------------------------------------------------------------------
EL_VITO_A = (
    E("E5",0.5), E("E5",0.5), E("E5",0.5), E("E5",0.5),
    E("E5",0.5), E("D5",0.5), E("D5",0.5), E("C5",0.5),
    E("B4",0.5), E("C5",0.5), E("E5",1),
    E("B4",0.5), E("C5",0.5), E("A4",0.5), E("G#4",0.5), E("E4",1.5),
    E("R",0.5),
)
EL_VITO_B = (
    E("E4",0.5), E("G#4",0.5), E("B4",1), E("G#4",0.5),
    E("A4",0.5), E("B4",0.5), E("A4",0.5), E("G#4",0.5),
    E("A4",0.5), E("B4",0.5), E("C5",1), E("A4",0.5),
    E("B4",0.5), E("G4",0.5), E("F4",0.5), E("E4",1.5),
    E("R",0.5),
)
EL_VITO = (*EL_VITO_A, *EL_VITO_A, *EL_VITO_B, *EL_VITO_B)


TUNES: dict[str, Tune] = {
    "sakura_sakura": Tune(
        id="sakura_sakura",
        title="Sakura Sakura",
        country="Japan",
        tradition="Traditional Japanese / Edo-period urban folk melody",
        style_hint="japanese_in",
        bpm=72,
        root="A",
        meter="4/4",
        source_url="https://abcnotation.com/tunePage?a=trillian.mit.edu/~jc/music/abc/Japan/Sakura_Am.utf8/0000",
        rights_note="Anonymous/traditional underlying melody; public-domain composition.",
        transcription_note="Monophonic ABC-derived melody; no modern accompaniment.",
        events=SAKURA,
    ),
    "mo_li_hua": Tune(
        id="mo_li_hua",
        title="Mo Li Hua (Jasmine Flower)",
        country="China",
        tradition="Traditional Chinese xiaodiao folk song",
        style_hint="chinese_gong",
        bpm=106,
        root="C",
        meter="2/2",
        source_url="https://en.wikipedia.org/wiki/Mo_Li_Hua",
        rights_note="Traditional melody documented from the 18th century and earlier textual tradition.",
        transcription_note="Monophonic melody from published LilyPond notation.",
        events=MOLIHUA,
    ),
    "arirang": Tune(
        id="arirang",
        title="Arirang",
        country="Korea",
        tradition="Traditional Korean folk song",
        style_hint="korean_traditional_candidate",
        bpm=92,
        root="G",
        meter="3/4",
        source_url="https://abcnotation.com/tunePage?a=trillian.mit.edu/~jc/music/abc/Korea/Arirang_G/0000",
        rights_note="Traditional folk melody transmitted collectively across generations.",
        transcription_note="Monophonic traditional ABC version in G; treat as a Korean traditional candidate, not proof of one specific mode.",
        events=ARIRANG,
    ),
    "londonderry_air": Tune(
        id="londonderry_air",
        title="Londonderry Air",
        country="Ireland",
        tradition="Traditional Irish air",
        style_hint="irish_air_not_dorian",
        bpm=70,
        root="Eb",
        meter="4/4",
        source_url="https://en.wikipedia.org/wiki/Londonderry_Air",
        rights_note="Anonymous air published in George Petrie's 1855 collection.",
        transcription_note="Monophonic transcription of the 1855 printed melody; Irish-air seed only, not a direct Irish-Dorian training example.",
        events=LONDONDERRY,
    ),
    "horgalaten": Tune(
        id="horgalaten",
        title="Hårgalåten",
        country="Sweden",
        tradition="Traditional Swedish hambo/polska-family tune",
        style_hint="swedish_hambo_not_dorian",
        bpm=138,
        root="A",
        meter="3/4",
        source_url="https://abcnotation.com/tunePage?a=trillian.mit.edu/~jc/music/abc/Sweden/hambo/HaargaLaaten_Am/0000",
        rights_note="Anonymous/traditional Swedish tune; several traditional variants exist.",
        transcription_note="A/B sections from an Am ABC variant; Swedish hambo seed, but this variant is not Dorian and should not directly tune a Dorian profile.",
        events=HORGALATEN,
    ),
    "tavaszi_szel": Tune(
        id="tavaszi_szel",
        title="Tavaszi szél vizet áraszt",
        country="Hungary",
        tradition="Traditional Hungarian folk/children's song",
        style_hint="hungarian_folk_not_hungarian_minor",
        bpm=110,
        root="D",
        meter="4/4",
        source_url="https://abcnotation.com/tunePage?a=www.campin.me.uk/Music/Chalumeau/0553",
        rights_note="Traditional Hungarian folk song.",
        transcription_note="Simple D-minor ABC version; Hungarian folk seed only and NOT Hungarian-minor scale evidence.",
        events=TAVASZI,
    ),
    "el_vito": Tune(
        id="el_vito",
        title="El Vito",
        country="Spain",
        tradition="Traditional Andalusian folk song/dance",
        style_hint="spanish_flamenco_candidate",
        bpm=144,
        root="E",
        meter="3/8",
        source_url="https://en.wikipedia.org/wiki/El_vito",
        rights_note="Traditional Andalusian underlying melody; modern arrangements can carry separate copyright.",
        transcription_note="Simplified independent monophonic pitch skeleton; plausible flamenco seed, but use multiple independent traditional sources before tuning.",
        events=EL_VITO,
    ),
}


NOTE_PC = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}


def note_to_midi(note: str) -> int:
    if note == "R":
        raise ValueError("Rest has no MIDI pitch")

    if len(note) >= 3 and note[1] in "#b":
        name = note[:2]
        octave = int(note[2:])
    else:
        name = note[0]
        octave = int(note[1:])

    return 12 * (octave + 1) + NOTE_PC[name]


def note_to_freq(note: str) -> float:
    midi = note_to_midi(note)
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def render_tone(freq: float, seconds: float, velocity: float = 0.85) -> np.ndarray:
    n = max(1, int(round(seconds * SR)))
    t = np.arange(n, dtype=np.float64) / SR

    # Neutral reed-like synthetic timbre.
    # The fundamental is dominant so YIN/pYIN remains stable.
    phase = 2.0 * np.pi * freq * t
    y = (
        0.78 * np.sin(phase)
        + 0.15 * np.sin(2.0 * phase)
        + 0.05 * np.sin(3.0 * phase)
        + 0.02 * np.sin(4.0 * phase)
    )

    attack_n = min(n, max(1, int(0.008 * SR)))
    release_n = min(n, max(1, int(0.035 * SR)))

    env = np.ones(n, dtype=np.float64)
    if attack_n > 1:
        env[:attack_n] = np.linspace(0.0, 1.0, attack_n, endpoint=False)
    if release_n > 1:
        env[-release_n:] *= np.linspace(1.0, 0.0, release_n)

    # Very mild decay; avoid culture-specific articulation.
    env *= np.exp(-0.10 * t / max(seconds, 1e-6))

    return (y * env * velocity).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(max(1, int(round(seconds * SR))), dtype=np.float32)


def render_tune(tune: Tune) -> np.ndarray:
    quarter_s = 60.0 / tune.bpm
    chunks: list[np.ndarray] = []

    for note, beats in tune.events:
        duration_s = max(0.0, beats * quarter_s)

        if note == "R":
            chunks.append(silence(duration_s))
            continue

        # Always leave a short explicit inter-note gap. This makes repeated
        # same-pitch notes recoverable by both onset and F0 segmentation.
        gap = min(NOTE_GAP_S, duration_s * 0.18)
        tone_s = max(0.020, duration_s - gap)

        chunks.append(render_tone(note_to_freq(note), tone_s))
        chunks.append(silence(gap))

    if not chunks:
        return np.zeros(1, dtype=np.float32)

    y = np.concatenate(chunks)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak * 0.88
    return y.astype(np.float32)


def tune_output_path(root: Path, tune: Tune) -> Path:
    country_dir = tune.country.lower().replace(" ", "_")
    return root / country_dir / f"{tune.id}.wav"


def write_metadata(root: Path, generated: list[tuple[Tune, Path, float]]) -> None:
    path = root / "metadata.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "id",
            "title",
            "country",
            "tradition",
            "style_hint",
            "root",
            "meter",
            "bpm",
            "duration_s",
            "source_url",
            "rights_note",
            "transcription_note",
        ])
        for tune, wav_path, duration_s in generated:
            writer.writerow([
                str(wav_path.relative_to(root)),
                tune.id,
                tune.title,
                tune.country,
                tune.tradition,
                tune.style_hint,
                tune.root,
                tune.meter,
                tune.bpm,
                f"{duration_s:.3f}",
                tune.source_url,
                tune.rights_note,
                tune.transcription_note,
            ])


def list_tunes() -> None:
    print(f"{'ID':22} {'COUNTRY':10} {'STYLE HINT':24} TITLE")
    print("-" * 86)
    for tune in TUNES.values():
        print(
            f"{tune.id:22} "
            f"{tune.country:10} "
            f"{tune.style_hint:24} "
            f"{tune.title}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate neutral monophonic WAVs from old/traditional melodies."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--song", choices=sorted(TUNES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/public_domain_seed_corpus"),
    )
    args = parser.parse_args()

    if args.list:
        list_tunes()
        return

    selected = list(TUNES.values()) if args.all else [TUNES[args.song]]

    args.output.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[Tune, Path, float]] = []

    for tune in selected:
        y = render_tune(tune)
        path = tune_output_path(args.output, tune)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, y, SR, subtype="PCM_16")

        duration_s = len(y) / SR
        generated.append((tune, path, duration_s))
        print(
            f"{tune.id:22} -> {path} "
            f"({duration_s:.2f}s, {tune.root}, {tune.bpm:g} BPM)"
        )

    write_metadata(args.output, generated)
    print(f"Metadata -> {args.output / 'metadata.csv'}")


if __name__ == "__main__":
    main()