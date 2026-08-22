#!/usr/bin/env python3
"""
japan_1892_dataset_generator.py

Build a 28-song monophonic WAV seed corpus from the 1892 edition of:

    Y. Nagai & K. Kobatake,
    "Japanese Popular Music:
     A Collection of the Popular Music of Japan Rendered into the Staff Notation"
    Osaka, S. Miki & Co., 1892.

Source transcriptions
---------------------
Daisyfield Archive of Japanese Traditional Music:
https://www.daisyfield.com/music/htm/-genres/japan.htm

Daisyfield states that the original song publications are out of copyright and
that Tom Potter donated his transcriptions to the public domain.

The archive table mixes in two songs from other editions:
- Miyasan      : 1891 edition
- Takai-Yama   : 1895 edition

Those two are deliberately EXCLUDED here, leaving the exact 28 songs from 1892.

What this generator does
------------------------
1. Downloads the public-domain MusicXML transcription for each of the 28 songs.
2. Parses the melody directly from MusicXML.
3. Reduces rare two-note chords to one monophonic pitch using pitch continuity.
4. Synthesizes every tune with the SAME neutral reed-like oscillator.
5. Writes WAV files to dataset/japan by default.
6. Writes metadata.csv and manifest.json.
7. Caches the downloaded source MusicXML under dataset/japan/_source_musicxml.

The goal is NOT to imitate Japanese instruments or performance practice.
The goal is to provide pitch/rhythm material to the Scaleify corpus trainer
without adding timbral/style bias from the synthesizer itself.

Dependencies
------------
    numpy
    requests
    soundfile

Usage
-----
Generate all 28:
    python japan_1892_dataset_generator.py

Choose another output directory:
    python japan_1892_dataset_generator.py --output my_dataset/japan

List songs:
    python japan_1892_dataset_generator.py --list

Generate one song:
    python japan_1892_dataset_generator.py --song echigo_jishi

Force re-download and re-render:
    python japan_1892_dataset_generator.py --force

Notes
-----
- This is a seed corpus from ONE historical collection. It is much better than
  one or two hand-picked melodies, but it should not be treated as a complete
  statistical model of all Japanese traditional music.
- The collection itself contains multiple traditional genres and modes.
- A later step can cluster the 28 pieces by learned pitch-class/mode statistics
  before tuning a specific profile such as japanese_in.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import soundfile as sf


SR = 44100
DEFAULT_OUTPUT = Path("dataset/japan")
ARCHIVE_URL = "https://www.daisyfield.com/music/htm/-genres/japan.htm"
COLLECTION_TITLE = "Japanese Popular Music"
COLLECTION_YEAR = 1892
RIGHTS_NOTE = (
    "Underlying 1892 publication is public domain; Daisyfield states that "
    "Tom Potter donated these transcriptions to the public domain."
)


@dataclass(frozen=True)
class Song:
    id: str
    title: str
    source_stem: str

    @property
    def xml_url(self) -> str:
        return (
            "https://www.daisyfield.com/music/jpm/xml/"
            f"{self.source_stem}.xml"
        )


# Exact 28 from the 1892 section.
# Miyasan (1891) and Takai-Yama (1895) are intentionally omitted.
SONGS: tuple[Song, ...] = (
    Song("asaku_tomo", "Asaku-Tomo", "JPM069-Asaku-Tomo"),
    Song("dodoitsu", "Dodoitsu", "JPM093-Dodoitsu"),
    Song("doteo_toruwa", "Doteo-Toruwa", "JPM061-Doteo-Toruwa"),
    Song("echigo_jishi", "Echigo-Jishi", "JPM096-Echigo-Jishi"),
    Song("fuku_ju_so", "Fuku-Ju-So", "JPM101-Fuku-Ju-So"),
    Song("gonbe_ga_tanemaku", "Gonbe ga Tanemaku", "JPM001-GonbeGaTanemaku"),
    Song("gosho_no_oniwa", "Gosho no Oniwa", "JPM080-GoshoNoOniwa"),
    Song("horete_kayouni", "Horete-Kayouni", "JPM056-Horete-Kayouni"),
    Song("inshu_inaba", "Inshu-Inaba", "JPM021-Inshu-Inaba"),
    Song("iyo_bushi", "Iyo-Bushi", "JPM072-Iyo-Bushi"),
    Song("kappore", "Kappore", "JPM024-Kappore"),
    Song("kappore_honen", "Kappore-Honen", "JPM032-K-Honen"),
    Song("kayo_kami", "Kayo-kami", "JPM053-Kayo-Kami"),
    Song("kosunoto", "Kosunoto", "JPM048-Kosunoto"),
    Song("kuro_kami", "Kuro-Kami", "JPM040-Kuro-Kami"),
    Song("murasaki", "Murasaki", "JPM085-Murasaki"),
    Song("na_no_ha", "Na no Ha", "JPM088-NaNoHa"),
    Song("o_edo_nihonbashi", "O Edo-Nihonbashi", "JPM007-O-Edo-Nihonbashi"),
    Song("oki_no_taisen", "Oki no Taisen", "JPM077-OkiNoTaisen"),
    Song("otsue_bushi", "Otsue-Bushi", "JPM064-Otsue-Bushi"),
    Song("riukiu_bushi", "Riukiu-Bushi", "JPM013-Riukiu-Bushi"),
    Song("sakura_miyotote", "Sakura-Miyotote", "JPM037-Sakura-Miyotote"),
    Song("sedo_no_danbatake", "Sedo no Danbatake", "JPM016-SedoNoDanbatake"),
    Song("suiryo_bushi", "Suiryo-Bushi", "JPM009-Suiryo-Bushi"),
    Song("toka_ebisu", "Toka-Ebisu", "JPM004-Toka-Ebisu"),
    Song("ukiyo_bushi", "Ukiyo-Bushi", "JPM105-Ukiyo-Bushi"),
    Song("waga_koiwa", "Waga-Koiwa", "JPM045-Waga-Koiwa"),
    Song("yube_yonda", "Yube-Yonda", "JPM029-YubeYonda"),
)


@dataclass
class XmlNote:
    onset_q: float
    duration_q: float
    midi: int | None
    tie_start: bool = False
    tie_stop: bool = False


@dataclass
class MelodyEvent:
    onset_q: float
    duration_q: float
    midi: int | None


NOTE_PC = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def child(element: ET.Element, name: str) -> ET.Element | None:
    for item in element:
        if local_name(item.tag) == name:
            return item
    return None


def children(element: ET.Element, name: str):
    for item in element:
        if local_name(item.tag) == name:
            yield item


def text_of(element: ET.Element | None, default: str = "") -> str:
    if element is None or element.text is None:
        return default
    return element.text.strip()


def parse_pitch(note: ET.Element) -> int | None:
    if child(note, "rest") is not None:
        return None

    pitch = child(note, "pitch")
    if pitch is None:
        # Unpitched/percussion is not useful for this corpus.
        return None

    step = text_of(child(pitch, "step"))
    octave_text = text_of(child(pitch, "octave"))
    alter_text = text_of(child(pitch, "alter"), "0")

    if step not in NOTE_PC or not octave_text:
        return None

    octave = int(octave_text)
    alter = int(round(float(alter_text)))
    return 12 * (octave + 1) + NOTE_PC[step] + alter


def find_first_tempo(root: ET.Element, fallback: float) -> float:
    # MusicXML can represent tempo as <sound tempo="..."> or
    # <metronome><per-minute>...</per-minute>.
    for element in root.iter():
        name = local_name(element.tag)

        if name == "sound":
            value = element.attrib.get("tempo")
            if value:
                try:
                    tempo = float(value)
                    if tempo > 0:
                        return tempo
                except ValueError:
                    pass

        elif name == "per-minute":
            try:
                tempo = float(text_of(element))
                if tempo > 0:
                    return tempo
            except ValueError:
                pass

    return fallback


def first_part(root: ET.Element) -> ET.Element:
    for element in root:
        if local_name(element.tag) == "part":
            return element
    raise ValueError("MusicXML contains no <part>")


def parse_musicxml(xml_bytes: bytes, fallback_bpm: float) -> tuple[list[MelodyEvent], float]:
    """
    Parse a Finale-style score-partwise MusicXML file without music21.

    The 1892 source is essentially single-staff melody. Rare simultaneous
    notes are collapsed to one pitch later.
    """
    root = ET.fromstring(xml_bytes)
    tempo = find_first_tempo(root, fallback_bpm)
    part = first_part(root)

    divisions = 1.0
    measure_base = 0.0
    all_notes: list[XmlNote] = []

    for measure in children(part, "measure"):
        local_cursor = 0.0
        local_extent = 0.0
        previous_note_onset = 0.0

        for item in measure:
            name = local_name(item.tag)

            if name == "attributes":
                div = child(item, "divisions")
                if div is not None:
                    try:
                        value = float(text_of(div))
                        if value > 0:
                            divisions = value
                    except ValueError:
                        pass
                continue

            if name == "backup":
                dur = child(item, "duration")
                if dur is not None:
                    local_cursor -= float(text_of(dur, "0")) / divisions
                continue

            if name == "forward":
                dur = child(item, "duration")
                if dur is not None:
                    local_cursor += float(text_of(dur, "0")) / divisions
                    local_extent = max(local_extent, local_cursor)
                continue

            if name != "note":
                continue

            is_chord = child(item, "chord") is not None
            is_grace = child(item, "grace") is not None

            duration_node = child(item, "duration")
            if duration_node is not None:
                duration_q = float(text_of(duration_node, "0")) / divisions
            elif is_grace:
                # Grace note without notated duration. Give it a short
                # symbolic duration; it will not materially dominate training.
                duration_q = 0.125
            else:
                duration_q = 0.0

            if is_chord:
                onset_local = previous_note_onset
            else:
                onset_local = local_cursor
                previous_note_onset = onset_local

            tie_types = {
                node.attrib.get("type", "")
                for node in children(item, "tie")
            }

            all_notes.append(
                XmlNote(
                    onset_q=measure_base + onset_local,
                    duration_q=max(0.0, duration_q),
                    midi=parse_pitch(item),
                    tie_start="start" in tie_types,
                    tie_stop="stop" in tie_types,
                )
            )

            if not is_chord:
                # Grace notes normally do not advance the MusicXML cursor.
                if not is_grace:
                    local_cursor += duration_q
                else:
                    # Keep grace note immediately before/at the main note.
                    pass

            local_extent = max(
                local_extent,
                onset_local + (0.0 if is_grace else duration_q),
            )

        # For multi-voice measures, local cursor may have been backed up.
        # Advance by the maximum occupied extent instead.
        measure_base += max(local_extent, local_cursor, 0.0)

    return collapse_to_monophonic(all_notes), tempo


def collapse_to_monophonic(notes: list[XmlNote]) -> list[MelodyEvent]:
    """
    Group simultaneous notes and choose a single melodic pitch.

    Selection rule:
    - one pitch: use it
    - chord: choose the pitch nearest the previously selected melody pitch
    - first chord: choose the highest pitch

    Rests are reconstructed from timeline gaps, not selected as chord notes.
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: (n.onset_q, n.midi is None, n.midi or -999))

    groups: list[list[XmlNote]] = []
    current: list[XmlNote] = []
    current_onset: float | None = None

    for note in notes:
        if current_onset is None or abs(note.onset_q - current_onset) <= 1e-7:
            current.append(note)
            current_onset = note.onset_q if current_onset is None else current_onset
        else:
            groups.append(current)
            current = [note]
            current_onset = note.onset_q

    if current:
        groups.append(current)

    selected: list[MelodyEvent] = []
    prev_pitch: int | None = None

    for group in groups:
        pitched = [n for n in group if n.midi is not None]
        if not pitched:
            # Explicit rests are not emitted here; silence appears as gaps
            # between pitched events.
            continue

        if len(pitched) == 1:
            chosen = pitched[0]
        elif prev_pitch is None:
            chosen = max(pitched, key=lambda n: int(n.midi))
        else:
            chosen = min(
                pitched,
                key=lambda n: (abs(int(n.midi) - prev_pitch), -int(n.midi)),
            )

        selected.append(
            MelodyEvent(
                onset_q=float(chosen.onset_q),
                duration_q=max(0.05, float(chosen.duration_q)),
                midi=int(chosen.midi),
            )
        )
        prev_pitch = int(chosen.midi)

    if not selected:
        return []

    # Merge tied/continuous same-pitch events when there is no positive gap.
    # We intentionally do NOT merge ordinary repeated notes separated by a
    # notated re-attack; their onsets stay distinct.
    merged: list[MelodyEvent] = []
    for event in selected:
        if (
            merged
            and event.midi == merged[-1].midi
            and event.onset_q < merged[-1].onset_q + merged[-1].duration_q - 1e-5
        ):
            prev = merged[-1]
            new_end = max(
                prev.onset_q + prev.duration_q,
                event.onset_q + event.duration_q,
            )
            prev.duration_q = new_end - prev.onset_q
        else:
            merged.append(event)

    return merged


def midi_to_frequency(midi: int) -> float:
    return 440.0 * 2.0 ** ((float(midi) - 69.0) / 12.0)


def synth_tone(freq: float, seconds: float) -> np.ndarray:
    n = max(1, int(round(seconds * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    phase = 2.0 * np.pi * freq * t

    # Same neutral timbre for all 28 pieces.
    y = (
        0.78 * np.sin(phase)
        + 0.15 * np.sin(2.0 * phase)
        + 0.05 * np.sin(3.0 * phase)
        + 0.02 * np.sin(4.0 * phase)
    )

    attack = min(n, max(1, int(round(0.008 * SR))))
    release = min(n, max(1, int(round(0.030 * SR))))

    env = np.ones(n, dtype=np.float64)
    if attack > 1:
        env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
    if release > 1:
        env[-release:] *= np.linspace(1.0, 0.0, release)

    return (y * env * 0.82).astype(np.float32)


def render_events(
    events: list[MelodyEvent],
    bpm: float,
    articulation_gap_ms: float,
) -> np.ndarray:
    if not events:
        return np.zeros(1, dtype=np.float32)

    quarter_s = 60.0 / max(1e-6, bpm)
    end_q = max(e.onset_q + e.duration_q for e in events)
    total_s = end_q * quarter_s + 0.10
    y = np.zeros(max(1, int(math.ceil(total_s * SR))), dtype=np.float32)

    gap_s = max(0.0, articulation_gap_ms / 1000.0)

    for event in events:
        if event.midi is None:
            continue

        start_s = event.onset_q * quarter_s
        duration_s = event.duration_q * quarter_s

        # Never make an artificial gap longer than 12% of the note.
        note_gap = min(gap_s, duration_s * 0.12)
        tone_s = max(0.018, duration_s - note_gap)

        tone = synth_tone(midi_to_frequency(event.midi), tone_s)
        start = int(round(start_s * SR))
        end = min(len(y), start + len(tone))

        if start < len(y):
            y[start:end] += tone[: end - start]

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y *= 0.88 / peak

    return y.astype(np.float32)


def download_xml(
    session: requests.Session,
    song: Song,
    cache_path: Path,
    force: bool,
    timeout: float,
) -> bytes:
    if cache_path.exists() and not force:
        return cache_path.read_bytes()

    response = session.get(song.xml_url, timeout=timeout)
    response.raise_for_status()

    data = response.content
    # Parse once now so an HTML error page never gets cached as XML.
    ET.fromstring(data)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


def sanitize_score_bpm(value: float, fallback: float) -> float:
    if not np.isfinite(value) or value < 30 or value > 240:
        return fallback
    return float(value)


def write_metadata(
    output: Path,
    rows: list[dict],
) -> None:
    fields = [
        "filename",
        "id",
        "title",
        "country",
        "collection",
        "collection_year",
        "bpm",
        "duration_s",
        "note_events",
        "pitch_min_midi",
        "pitch_max_midi",
        "source_musicxml",
        "archive_url",
        "rights",
    ]

    with (output / "metadata.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(output: Path) -> None:
    payload = {
        "dataset": "japan_1892_nagai_kobatake",
        "country": "Japan",
        "collection": COLLECTION_TITLE,
        "collection_year": COLLECTION_YEAR,
        "archive_url": ARCHIVE_URL,
        "rights": RIGHTS_NOTE,
        "song_count": len(SONGS),
        "excluded_other_editions": [
            {
                "title": "Miyasan",
                "edition": 1891,
                "reason": "Not part of the exact 1892 28-song set",
            },
            {
                "title": "Takai-Yama",
                "edition": 1895,
                "reason": "Not part of the exact 1892 28-song set",
            },
        ],
        "songs": [
            {
                "id": s.id,
                "title": s.title,
                "source_musicxml": s.xml_url,
            }
            for s in SONGS
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def list_songs() -> None:
    print(f"{'#':>2}  {'ID':24} TITLE")
    print("-" * 62)
    for i, song in enumerate(SONGS, start=1):
        print(f"{i:2d}  {song.id:24} {song.title}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the exact 28-song Nagai/Kobatake 1892 Japanese "
            "traditional-melody WAV seed corpus."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory (default: dataset/japan)",
    )
    parser.add_argument(
        "--song",
        choices=[s.id for s in SONGS],
        default=None,
        help="Generate only one song. Default: all 28.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download MusicXML and overwrite existing WAVs.",
    )
    parser.add_argument(
        "--fallback-bpm",
        type=float,
        default=100.0,
        help="Tempo when MusicXML has no usable tempo marking.",
    )
    parser.add_argument(
        "--articulation-gap-ms",
        type=float,
        default=18.0,
        help=(
            "Neutral release gap inserted inside notes. "
            "Small by design; note onset timing remains score-derived."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.08,
        help="Polite delay between source downloads.",
    )
    args = parser.parse_args()

    if args.list:
        list_songs()
        return

    if args.fallback_bpm <= 0:
        parser.error("--fallback-bpm must be > 0")

    output = args.output
    source_dir = output / "_source_musicxml"
    output.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        [next(s for s in SONGS if s.id == args.song)]
        if args.song
        else list(SONGS)
    )

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Scaleify research corpus generator/1.0 "
            "(public-domain MusicXML downloader)"
        )
    })

    rows: list[dict] = []
    failures: list[dict] = []

    for index, song in enumerate(selected, start=1):
        wav_path = output / f"{song.id}.wav"
        xml_path = source_dir / f"{song.source_stem}.xml"

        print(f"[{index:02d}/{len(selected):02d}] {song.title}")

        try:
            xml_bytes = download_xml(
                session=session,
                song=song,
                cache_path=xml_path,
                force=args.force,
                timeout=args.timeout,
            )

            events, score_bpm = parse_musicxml(
                xml_bytes,
                fallback_bpm=args.fallback_bpm,
            )
            bpm = sanitize_score_bpm(score_bpm, args.fallback_bpm)

            if not events:
                raise RuntimeError("No pitched melody events parsed")

            if args.force or not wav_path.exists():
                audio = render_events(
                    events=events,
                    bpm=bpm,
                    articulation_gap_ms=args.articulation_gap_ms,
                )
                sf.write(
                    wav_path,
                    audio,
                    SR,
                    subtype="PCM_16",
                )
            else:
                audio, existing_sr = sf.read(
                    wav_path,
                    dtype="float32",
                    always_2d=False,
                )
                if existing_sr != SR:
                    print(
                        f"    [warn] existing WAV sample rate={existing_sr}, "
                        f"expected={SR}"
                    )

            duration_s = len(audio) / SR
            pitches = [int(e.midi) for e in events if e.midi is not None]

            rows.append({
                "filename": wav_path.name,
                "id": song.id,
                "title": song.title,
                "country": "Japan",
                "collection": COLLECTION_TITLE,
                "collection_year": COLLECTION_YEAR,
                "bpm": round(bpm, 3),
                "duration_s": round(duration_s, 3),
                "note_events": len(events),
                "pitch_min_midi": min(pitches),
                "pitch_max_midi": max(pitches),
                "source_musicxml": song.xml_url,
                "archive_url": ARCHIVE_URL,
                "rights": RIGHTS_NOTE,
            })

            print(
                f"    notes={len(events)} "
                f"bpm={bpm:.1f} "
                f"duration={duration_s:.2f}s "
                f"-> {wav_path}"
            )

        except Exception as exc:
            failures.append({
                "id": song.id,
                "title": song.title,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"    [FAILED] {type(exc).__name__}: {exc}")

        if args.request_delay > 0:
            time.sleep(args.request_delay)

    if rows:
        write_metadata(output, rows)
    write_manifest(output)

    if failures:
        (output / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        failure_path = output / "failures.json"
        if failure_path.exists():
            failure_path.unlink()

    print()
    print(f"Generated: {len(rows)}/{len(selected)}")
    print(f"Output:    {output}")
    print(f"Metadata:  {output / 'metadata.csv'}")
    print(f"Manifest:  {output / 'manifest.json'}")

    if failures:
        print(f"Failures:  {len(failures)} -> {output / 'failures.json'}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()