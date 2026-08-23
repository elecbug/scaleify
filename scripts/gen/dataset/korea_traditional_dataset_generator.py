#!/usr/bin/env python3
"""
korea_traditional_dataset_generator.py

Build a neutral monophonic Korean traditional-song WAV corpus.

Primary source
--------------
Music Laboratory / World Traditional Songs:
https://www.mu-tech.org/WorldTrad/index_korean_song.html

The source site provides Korean traditional-song score pages and downloadable
MIDI files. Its terms state that most free folk/traditional-song data is based
on anonymous or copyright-expired works, and that traditional/arranged MIDI,
MP3 and MP4 files may be used freely for non-commercial purposes.

Terms:
https://mu-tech.org/Traditional/TermsOfService.html

IMPORTANT
---------
This generator does NOT render the source accompaniment/arrangement.

The site's downloadable MIDI often contains automatically generated
accompaniment. To avoid teaching Scaleify the arranger's harmony/timbre rather
than Korean melodic grammar, this script:

    source MIDI
        -> choose likely melody track
        -> reduce that track to one monophonic pitch at a time
        -> discard programs/instruments/accompaniment
        -> render all songs with the SAME neutral synthetic timbre
        -> dataset/korea/*.wav

The generated WAV corpus is therefore intended for non-commercial research and
experimentation unless you separately verify broader rights for your use case.

Default output
--------------
dataset/korea/

Dependencies
------------
numpy
requests
soundfile
mido

Usage
-----
Generate the curated corpus:

    python korea_traditional_dataset_generator.py

List included songs:

    python korea_traditional_dataset_generator.py --list

Generate one song:

    python korea_traditional_dataset_generator.py --song jindo_arirang

Force source MIDI re-download and WAV regeneration:

    python korea_traditional_dataset_generator.py --force

Notes
-----
- Modern composed songs, national anthems and most children's songs listed on
  the source index are deliberately excluded.
- Two regional "Gaegori Taryeong" entries are retained independently.
- The source index/transliterations contain occasional romanization/metadata
  errors. The original source URL is stored for every output.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import mido
import numpy as np
import requests
import soundfile as sf


SR = 44100
DEFAULT_OUTPUT = Path("datasets/korea")
INDEX_URL = "https://www.mu-tech.org/WorldTrad/index_korean_song.html"
TERMS_URL = "https://mu-tech.org/Traditional/TermsOfService.html"
DEFAULT_RENDER_BPM = 100.0
DEFAULT_GAP_MS = 18.0

RIGHTS_NOTE = (
    "Traditional/folk source. Music Laboratory terms describe most free "
    "traditional-song material as anonymous/copyright-expired and permit "
    "non-commercial use of traditional/arranged MIDI data. Verify separately "
    "before commercial use or redistribution."
)


@dataclass(frozen=True)
class SongSpec:
    id: str
    anchor: str
    title: str
    title_ko: str
    region: str
    occurrence: int = 0
    note: str = ""


# Curated traditional/folk subset from the source site's Korean index.
# Modern named-composer songs, national anthems, obvious imported songs and
# most nursery-rhyme material are intentionally omitted.
SONGS: tuple[SongSpec, ...] = (
    SongSpec("doraji", "Dolaji", "Doraji Taryeong", "도라지타령", "Korea"),
    SongSpec("gyeonggi_arirang", "Gyoenggido Arirang", "Gyeonggi Arirang", "경기도 아리랑", "Gyeonggi"),
    SongSpec("gin_arirang", "Gin a ri rang", "Gin Arirang", "긴 아리랑", "Gyeonggi"),
    SongSpec("jindo_arirang", "Jindo Arirang", "Jindo Arirang", "진도 아리랑", "Jeolla"),
    SongSpec("gangwon_arirang", "Gang won do arirang", "Gangwon-do Arirang", "강원도 아리랑", "Gangwon"),
    SongSpec("yeongnam_arirang", "Yeungnam arirang", "Yeongnam Arirang", "영남 아리랑", "Yeongnam"),
    SongSpec("miryang_arirang", "Miryan arirang", "Miryang Arirang", "밀양 아리랑", "Gyeongsang"),
    SongSpec("seodo_arirang", "SuhDo Arirang", "Seodo Arirang", "서도 아리랑", "Seodo"),

    SongSpec("gaetaryeong", "Gae ta ryeong", "Gaetaryeong", "개타령", "Gyeonggi"),
    SongSpec(
        "gaegori_taryeong_jeju", "Gae go ri ta ryeong",
        "Gaegori Taryeong (Jeju)", "개고리타령", "Jeju", occurrence=0
    ),
    SongSpec(
        "gaegori_taryeong_jeolla", "Gae go ri ta ryeong",
        "Gaegori Taryeong (Jeolla)", "개고리타령", "Jeolla", occurrence=1
    ),
    SongSpec("gakseori_taryeong", "Gak seor i ta ryeong", "Gakseori Taryeong", "각설이타령", "Korea"),
    SongSpec("gayageum_taryeong", "Gayaeum ta ryeong", "Gayageum Taryeong", "가야금타령", "Jeolla"),
    SongSpec("geondeureong_taryeong", "Geon deu reong ta ryeong", "Geondeureong Taryeong", "건드렁타령", "Gyeonggi"),
    SongSpec("gunbam_taryeong", "Gun bamta ryeong", "Gunbam Taryeong", "군밤타령", "Gyeonggi"),
    SongSpec("gyeongbokgung_taryeong", "Gyeong bok gung ta ryeong", "Gyeongbokgung Taryeong", "경복궁타령", "Gyeonggi"),
    SongSpec("nampo_taryeong", "Nam po ta ryeong", "Nampo Taryeong", "남포타령", "Jeolla"),
    SongSpec("kkaturi_taryeong", "Kkaturi Taryeong", "Kkaturi Taryeong", "까투리타령", "Jeolla"),

    SongSpec("ganggangsullae", "Gang gang su wol rae", "Ganggangsullae", "강강수월래", "Jeolla"),
    SongSpec("ulsan_agassi", "Ulsan agasi", "Ulsan Agassi", "울산아가씨", "Gyeongsang"),
    SongSpec("singosan_taryeong", "Shinkosan Taryon", "Singosan Taryeong", "신고산타령", "Hamgyeong"),
    SongSpec("cheonan_samgeori", "Chonan samugori", "Cheonan Samgeori", "천안삼거리", "Chungcheong"),
    SongSpec("geomjilmaegi", "Geom jil mae gi", "Geomjilmaegi", "검질매기", "Jeju"),
    SongSpec("geomundo_baetnorae", "Geo mun do baet no rae", "Geomundo Baetnorae", "거문도뱃노래", "Jeolla"),
    SongSpec("gim_gyehwa", "Gim gye hwa", "Gim Gyehwa", "김계화", "Jeju"),
    SongSpec("gungcho_daenggi", "Gung cho daeng gi", "Gungcho Daenggi", "궁초댕기", "Hamgyeong"),
    SongSpec("gwanghallu", "Gwang han ru", "Gwanghallu", "광한루", "Jeolla"),

    # The source site romanizes this unusually as "Iuccia Paighi" and shows
    # the characters 六字伯. It is retained under the conventional Korean
    # research label Yukjabaegi, while preserving the source URL in metadata.
    SongSpec(
        "yukjabaegi", "Iuccia Paighi", "Yukjabaegi", "육자배기", "Jeolla",
        note="Source site uses the romanization 'Iuccia Paighi'."
    ),
    SongSpec("nongbuga", "Nongbuga", "Nongbuga", "농부가", "Jeolla"),
    SongSpec("batgari_taryeong", "Batgal-italyeong", "Batgari Taryeong", "밭갈이타령", "Korean Peninsula"),
    SongSpec("saeya_saeya_parangsaeya", "Saeyasaeya palangsaeya", "Saeya Saeya Parangsaeya", "새야새야 파랑새야", "Korea"),
    SongSpec("yangyang_palgyeong", "Yang-yang palgyeong", "Yangyang Palgyeong", "양양팔경", "Gangwon"),
    SongSpec("geumgangsan_taryeong", "Geum gang san ta ryeong", "Geumgangsan Taryeong", "금강산타령", "Gangwon"),
    SongSpec("yangsando", "Yangsando", "Yangsando", "양산도", "Korean Peninsula"),
)


@dataclass
class Anchor:
    text: str
    href: str


@dataclass
class MidiNote:
    start: int
    end: int
    pitch: int
    velocity: int = 80


@dataclass
class TrackCandidate:
    index: int
    name: str
    notes: list[MidiNote]
    score: float
    monophony: float
    mean_pitch: float
    unique_pitches: int


class AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[Anchor] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        attrs = dict(attrs)
        self._href = attrs.get("href")
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        text = " ".join("".join(self._parts).split())
        self.anchors.append(Anchor(text=text, href=self._href))
        self._href = None
        self._parts = []


def normalize_anchor(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isascii() and ch.isalnum())


def decode_page(content: bytes) -> str:
    """
    Preserve all ASCII URLs/romanized link text even if an old page's declared
    character set is inconsistent.
    """
    for encoding in ("utf-8", "shift_jis", "euc_jp", "cp949"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            pass
    return content.decode("latin-1", errors="replace")


def fetch(
    session: requests.Session,
    url: str,
    timeout: float,
) -> bytes:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def discover_index_pages(
    session: requests.Session,
    timeout: float,
) -> dict[str, list[str]]:
    content = fetch(session, INDEX_URL, timeout)
    parser = AnchorParser()
    parser.feed(decode_page(content))

    found: dict[str, list[str]] = {}
    for anchor in parser.anchors:
        href = anchor.href.strip()
        if not href:
            continue

        full = urljoin(INDEX_URL, href)
        parsed = urlparse(full)
        if "mu-tech.org" not in parsed.netloc:
            continue
        if not parsed.path.lower().endswith(".html"):
            continue
        if "index_" in Path(parsed.path).name.lower():
            continue

        key = normalize_anchor(anchor.text)
        if not key:
            continue
        found.setdefault(key, [])
        if full not in found[key]:
            found[key].append(full)

    return found


def resolve_song_page(
    spec: SongSpec,
    discovered: dict[str, list[str]],
) -> str:
    key = normalize_anchor(spec.anchor)
    matches = discovered.get(key, [])

    if len(matches) <= spec.occurrence:
        # Loose substring fallback for harmless spacing/romanization changes.
        candidates = []
        for discovered_key, urls in discovered.items():
            if key in discovered_key or discovered_key in key:
                candidates.extend(urls)
        matches = list(dict.fromkeys(candidates))

    if len(matches) <= spec.occurrence:
        raise RuntimeError(
            f"Could not find source page for {spec.title!r} "
            f"(anchor={spec.anchor!r}, occurrence={spec.occurrence})"
        )

    return matches[spec.occurrence]


def extract_midi_links(page_url: str, page_content: bytes) -> list[str]:
    parser = AnchorParser()
    parser.feed(decode_page(page_content))

    links = []
    for anchor in parser.anchors:
        href = html.unescape(anchor.href.strip())
        if ".mid" not in href.lower():
            continue
        full = urljoin(page_url, href)
        if full not in links:
            links.append(full)
    return links


def choose_source_midi_url(page_url: str, page_content: bytes) -> str:
    links = extract_midi_links(page_url, page_content)

    # Prefer the site's common "Unplugged" style merely as a stable download
    # source. Its accompaniment is discarded later.
    for url in links:
        if "_unplugged.mid" in url.lower():
            return url

    if links:
        return links[0]

    # Some old pages construct download controls dynamically. The source site
    # uses this stable naming pattern for many WorldTrad pages.
    stem = Path(urlparse(page_url).path).stem
    return (
        "https://www.mu-tech.co.jp/midi/traditional/fsout/"
        f"{stem}_Unplugged.mid"
    )


def download_source_midi(
    session: requests.Session,
    url: str,
    path: Path,
    timeout: float,
    force: bool,
) -> None:
    if path.exists() and not force:
        return

    data = fetch(session, url, timeout)
    if len(data) < 14 or data[:4] != b"MThd":
        raise RuntimeError(
            f"Downloaded file is not a Standard MIDI file: {url}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def track_notes(track: mido.MidiTrack) -> tuple[str, list[MidiNote]]:
    absolute = 0
    name = ""
    active: dict[tuple[int, int], list[tuple[int, int]]] = {}
    notes: list[MidiNote] = []

    for msg in track:
        absolute += int(msg.time)

        if msg.type == "track_name" and not name:
            name = str(msg.name)
            continue

        if not hasattr(msg, "channel"):
            continue
        if int(msg.channel) == 9:  # General MIDI drums
            continue

        if msg.type == "note_on" and int(msg.velocity) > 0:
            key = (int(msg.channel), int(msg.note))
            active.setdefault(key, []).append((absolute, int(msg.velocity)))
            continue

        if msg.type in ("note_off", "note_on"):
            # note_on velocity=0 is note-off.
            key = (int(msg.channel), int(msg.note))
            stack = active.get(key)
            if not stack:
                continue
            start, velocity = stack.pop(0)
            if absolute > start:
                notes.append(
                    MidiNote(
                        start=start,
                        end=absolute,
                        pitch=int(msg.note),
                        velocity=velocity,
                    )
                )

    return name, notes


def overlap_monophony(notes: list[MidiNote]) -> float:
    if len(notes) <= 1:
        return 1.0

    notes = sorted(notes, key=lambda n: (n.start, n.end, n.pitch))
    overlap_count = 0
    current_end = notes[0].end
    for note in notes[1:]:
        if note.start < current_end:
            overlap_count += 1
        current_end = max(current_end, note.end)

    return max(0.0, 1.0 - overlap_count / max(1, len(notes) - 1))


def score_track(index: int, name: str, notes: list[MidiNote]) -> TrackCandidate | None:
    if len(notes) < 6:
        return None

    pitches = np.asarray([n.pitch for n in notes], dtype=np.float64)
    mean_pitch = float(np.mean(pitches))
    unique = len(set(int(x) for x in pitches))
    mono = overlap_monophony(notes)

    name_l = name.lower()
    name_bonus = 0.0
    if any(token in name_l for token in ("melody", "vocal", "lead", "メロ", "歌")):
        name_bonus += 220.0
    if any(token in name_l for token in ("bass", "chord", "drum", "perc", "伴奏")):
        name_bonus -= 180.0

    # Auto-arrangers commonly put the melody in an early track. Reward that
    # gently, but let monophony/pitch/count dominate when metadata differs.
    early_bonus = max(0.0, 55.0 - 10.0 * index)

    # Penalize bass-like tracks.
    low_penalty = max(0.0, 52.0 - mean_pitch) * 5.0

    score = (
        2.8 * len(notes) * mono
        + 2.0 * unique
        + 1.2 * mean_pitch
        + 120.0 * mono
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
    )


def select_melody_track(
    midi: mido.MidiFile,
    strategy: str,
) -> tuple[TrackCandidate, list[TrackCandidate]]:
    candidates: list[TrackCandidate] = []

    for index, track in enumerate(midi.tracks):
        name, notes = track_notes(track)
        candidate = score_track(index, name, notes)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No pitched MIDI track with enough note events")

    if strategy == "first":
        chosen = min(candidates, key=lambda c: c.index)
    elif strategy == "highest":
        # Among reasonably monophonic candidates, take highest mean register.
        mono = [c for c in candidates if c.monophony >= 0.80]
        chosen = max(mono or candidates, key=lambda c: (c.mean_pitch, c.score))
    else:
        chosen = max(candidates, key=lambda c: c.score)

    return chosen, sorted(candidates, key=lambda c: c.score, reverse=True)


def monophonize(notes: list[MidiNote]) -> list[MidiNote]:
    """
    Collapse simultaneous pitches to one note and prevent overlapping output.

    Highest pitch wins inside an exact-onset chord, which is a good fallback for
    melody-track doublings. Ordinary repeated same-pitch attacks are retained.
    """
    if not notes:
        return []

    groups: dict[int, list[MidiNote]] = {}
    for note in notes:
        groups.setdefault(note.start, []).append(note)

    chosen: list[MidiNote] = []
    for onset in sorted(groups):
        group = groups[onset]
        best = max(group, key=lambda n: (n.pitch, n.velocity, n.end - n.start))
        chosen.append(
            MidiNote(
                start=best.start,
                end=best.end,
                pitch=best.pitch,
                velocity=best.velocity,
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


def synth_tone(freq: float, seconds: float, velocity: int) -> np.ndarray:
    n = max(1, int(round(seconds * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    phase = 2.0 * np.pi * freq * t

    # Same neutral reed-like timbre used for every corpus song.
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

    amp = 0.68 + 0.14 * min(1.0, max(0.0, velocity / 127.0))
    return (y * env * amp).astype(np.float32)


def midi_to_freq(midi_pitch: int) -> float:
    return 440.0 * 2.0 ** ((float(midi_pitch) - 69.0) / 12.0)


def render_melody(
    notes: list[MidiNote],
    ticks_per_beat: int,
    bpm: float,
    gap_ms: float,
) -> np.ndarray:
    if not notes:
        return np.zeros(1, dtype=np.float32)

    first_tick = min(n.start for n in notes)
    shifted = [
        MidiNote(
            start=n.start - first_tick,
            end=n.end - first_tick,
            pitch=n.pitch,
            velocity=n.velocity,
        )
        for n in notes
    ]

    seconds_per_tick = (60.0 / bpm) / max(1, ticks_per_beat)
    final_tick = max(n.end for n in shifted)
    total_s = final_tick * seconds_per_tick + 0.10
    audio = np.zeros(max(1, int(math.ceil(total_s * SR))), dtype=np.float32)

    gap_s = max(0.0, gap_ms / 1000.0)

    for note in shifted:
        start_s = note.start * seconds_per_tick
        dur_s = max(seconds_per_tick, (note.end - note.start) * seconds_per_tick)

        # Explicit but small release gap improves repeated-note segmentation.
        note_gap = min(gap_s, dur_s * 0.12)
        tone_s = max(0.018, dur_s - note_gap)

        tone = synth_tone(midi_to_freq(note.pitch), tone_s, note.velocity)
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
    strategy: str,
    bpm: float,
    gap_ms: float,
) -> dict:
    midi = mido.MidiFile(midi_path)
    chosen, candidates = select_melody_track(midi, strategy)
    notes = monophonize(chosen.notes)

    if len(notes) < 6:
        raise RuntimeError(
            f"Selected melody track produced only {len(notes)} monophonic notes"
        )

    audio = render_melody(
        notes=notes,
        ticks_per_beat=midi.ticks_per_beat,
        bpm=bpm,
        gap_ms=gap_ms,
    )
    sf.write(wav_path, audio, SR, subtype="PCM_16")

    return {
        "duration_s": len(audio) / SR,
        "note_events": len(notes),
        "pitch_min_midi": min(n.pitch for n in notes),
        "pitch_max_midi": max(n.pitch for n in notes),
        "selected_track_index": chosen.index,
        "selected_track_name": chosen.name,
        "selected_track_score": chosen.score,
        "selected_track_monophony": chosen.monophony,
        "selected_track_mean_pitch": chosen.mean_pitch,
        "track_candidates": [
            {
                "index": c.index,
                "name": c.name,
                "notes": len(c.notes),
                "score": round(c.score, 4),
                "monophony": round(c.monophony, 4),
                "mean_pitch": round(c.mean_pitch, 3),
                "unique_pitches": c.unique_pitches,
            }
            for c in candidates[:8]
        ],
    }


def write_metadata(output: Path, rows: list[dict]) -> None:
    fields = [
        "filename",
        "id",
        "title",
        "title_ko",
        "region",
        "duration_s",
        "note_events",
        "pitch_min_midi",
        "pitch_max_midi",
        "selected_track_index",
        "selected_track_name",
        "selected_track_monophony",
        "render_bpm",
        "source_page",
        "source_midi",
        "terms_url",
        "rights",
        "note",
    ]

    with (output / "metadata.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def list_songs() -> None:
    print(f"{'#':>2}  {'ID':28} {'REGION':18} TITLE")
    print("-" * 88)
    for i, song in enumerate(SONGS, start=1):
        print(f"{i:2d}  {song.id:28} {song.region:18} {song.title} / {song.title_ko}")


def write_manifest(
    output: Path,
    rows: list[dict],
    failures: list[dict],
    render_bpm: float,
    strategy: str,
) -> None:
    payload = {
        "dataset": "korea_traditional_mu_tech",
        "country": "Korea",
        "source_index": INDEX_URL,
        "terms_url": TERMS_URL,
        "rights_note": RIGHTS_NOTE,
        "selection": (
            "Curated traditional/folk subset. Modern named-composer songs, "
            "anthems, obvious imported songs and most nursery rhymes excluded."
        ),
        "rendering": {
            "sample_rate": SR,
            "render_bpm": render_bpm,
            "track_strategy": strategy,
            "timbre": "neutral_reed_like",
            "source_accompaniment_used": False,
        },
        "song_count_requested": len(SONGS),
        "song_count_generated": len(rows),
        "songs": rows,
        "failures": failures,
    }

    (output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a neutral monophonic Korean traditional-song WAV corpus "
            "from the Music Laboratory WorldTrad MIDI collection."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory (default: dataset/korea)",
    )
    parser.add_argument(
        "--song",
        choices=[s.id for s in SONGS],
        default=None,
        help="Generate only one song. Default: full curated corpus.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--track-strategy",
        choices=["auto", "first", "highest"],
        default="auto",
        help="How to choose the likely melody MIDI track.",
    )
    parser.add_argument(
        "--render-bpm",
        type=float,
        default=DEFAULT_RENDER_BPM,
        help=(
            "Neutral fixed render tempo. MIDI rhythm ratios are preserved while "
            "source arrangement tempo is ignored."
        ),
    )
    parser.add_argument(
        "--articulation-gap-ms",
        type=float,
        default=DEFAULT_GAP_MS,
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--request-delay", type=float, default=0.10)
    args = parser.parse_args()

    if args.list:
        list_songs()
        return

    if args.render_bpm <= 0:
        parser.error("--render-bpm must be > 0")

    output = args.output
    source_dir = output / "_source_midi"
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
            "(non-commercial traditional-song dataset builder)"
        )
    })

    print("Discovering source pages...")
    discovered = discover_index_pages(session, args.timeout)
    print(f"Indexed anchor keys: {len(discovered)}")
    print()

    rows: list[dict] = []
    failures: list[dict] = []

    for i, spec in enumerate(selected, start=1):
        print(f"[{i:02d}/{len(selected):02d}] {spec.title} / {spec.title_ko}")

        try:
            page_url = resolve_song_page(spec, discovered)
            page_content = fetch(session, page_url, args.timeout)
            midi_url = choose_source_midi_url(page_url, page_content)

            midi_path = source_dir / f"{spec.id}.mid"
            wav_path = output / f"{spec.id}.wav"

            download_source_midi(
                session=session,
                url=midi_url,
                path=midi_path,
                timeout=args.timeout,
                force=args.force,
            )

            if args.force or not wav_path.exists():
                analysis = analyze_and_render(
                    midi_path=midi_path,
                    wav_path=wav_path,
                    strategy=args.track_strategy,
                    bpm=args.render_bpm,
                    gap_ms=args.articulation_gap_ms,
                )
            else:
                # Re-analyze MIDI for metadata, but do not rewrite audio.
                midi = mido.MidiFile(midi_path)
                chosen, candidates = select_melody_track(
                    midi, args.track_strategy
                )
                notes = monophonize(chosen.notes)
                existing, existing_sr = sf.read(
                    wav_path, dtype="float32", always_2d=False
                )
                analysis = {
                    "duration_s": len(existing) / existing_sr,
                    "note_events": len(notes),
                    "pitch_min_midi": min(n.pitch for n in notes),
                    "pitch_max_midi": max(n.pitch for n in notes),
                    "selected_track_index": chosen.index,
                    "selected_track_name": chosen.name,
                    "selected_track_score": chosen.score,
                    "selected_track_monophony": chosen.monophony,
                    "selected_track_mean_pitch": chosen.mean_pitch,
                    "track_candidates": [
                        {
                            "index": c.index,
                            "name": c.name,
                            "notes": len(c.notes),
                            "score": round(c.score, 4),
                            "monophony": round(c.monophony, 4),
                            "mean_pitch": round(c.mean_pitch, 3),
                            "unique_pitches": c.unique_pitches,
                        }
                        for c in candidates[:8]
                    ],
                }

            row = {
                "filename": wav_path.name,
                "id": spec.id,
                "title": spec.title,
                "title_ko": spec.title_ko,
                "region": spec.region,
                "duration_s": round(float(analysis["duration_s"]), 3),
                "note_events": int(analysis["note_events"]),
                "pitch_min_midi": int(analysis["pitch_min_midi"]),
                "pitch_max_midi": int(analysis["pitch_max_midi"]),
                "selected_track_index": int(analysis["selected_track_index"]),
                "selected_track_name": analysis["selected_track_name"],
                "selected_track_monophony": round(
                    float(analysis["selected_track_monophony"]), 4
                ),
                "render_bpm": args.render_bpm,
                "source_page": page_url,
                "source_midi": midi_url,
                "terms_url": TERMS_URL,
                "rights": RIGHTS_NOTE,
                "note": spec.note,
                "track_candidates": analysis["track_candidates"],
            }
            rows.append(row)

            print(
                f"    track={analysis['selected_track_index']} "
                f"mono={analysis['selected_track_monophony']:.3f} "
                f"notes={analysis['note_events']} "
                f"duration={analysis['duration_s']:.2f}s "
                f"-> {wav_path}"
            )

        except Exception as exc:
            failure = {
                "id": spec.id,
                "title": spec.title,
                "title_ko": spec.title_ko,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"    [FAILED] {failure['error']}")

        if args.request_delay > 0:
            time.sleep(args.request_delay)

    write_metadata(output, rows)
    write_manifest(
        output=output,
        rows=rows,
        failures=failures,
        render_bpm=args.render_bpm,
        strategy=args.track_strategy,
    )

    if failures:
        (output / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        fail_path = output / "failures.json"
        if fail_path.exists():
            fail_path.unlink()

    print()
    print("Generation complete")
    print("-------------------")
    print(f"Generated: {len(rows)}/{len(selected)}")
    print(f"Output:    {output}")
    print(f"Metadata:  {output / 'metadata.csv'}")
    print(f"Manifest:  {output / 'manifest.json'}")

    if failures:
        print(f"Failures:  {len(failures)} -> {output / 'failures.json'}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()