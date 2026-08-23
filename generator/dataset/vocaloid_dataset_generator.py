#!/usr/bin/env python3
"""
vocaloid_dataset_generator.py

Build a Scaleify-compatible monophonic Vocaloid / singing-synth melody corpus.

Supported source formats
------------------------
Native:
  - .vsqx  VOCALOID3 / VOCALOID4
  - .vsq   VOCALOID2 (SMF-based VSQ; parsed as MIDI notes)
  - .ustx  OpenUtau
  - .ust   UTAU
  - .mid / .midi  standard or VOCALOID-exported MIDI

For .vpr (VOCALOID5), .ppsf, .svp, etc., convert to .vsqx or .ustx first with
UtaFormatix, then feed the converted files to this tool.

Official-source presets
-----------------------
The generator can also discover creator-distributed symbolic assets:

  --preset otoiro-deco27
      DECO*27 official MIDI assets from OTOIRO SPECIAL.

  --preset crusher
      Original Crusher songs with creator-distributed VSQ/VSQX/MIDI/UST assets.

  --preset official
      Both of the above.

Preset downloads require --accept-source-terms. Use --discover-only to inspect
what would be downloaded without fetching the project files.

Output
------
dataset/vocaloid/
  <song>.wav                # directly consumable by Scaleify v10.1 train_style.py
  metadata.csv
  manifest.json
  _symbolic/<song>.csv      # exact extracted note events for audit / future trainer
  _downloads/               # optional manifest-driven downloads
  _source/                  # downloaded/extracted project files

Why WAV?
--------
The current Scaleify v10.1 trainer recursively reads monophonic WAV files and
performs onset-aware F0 extraction. This generator renders every symbolic vocal
track using the same neutral synthetic timbre, fixed BPM, and a small articulation
gap. Therefore the existing trainer can be used unchanged.

The exact symbolic note sequence is also written to _symbolic/*.csv so a future
symbolic-native trainer can skip F0 extraction entirely.

Copyright / corpus policy
-------------------------
A downloadable VSQX/USTX file and the underlying musical composition can have
different rights. This tool does NOT ship or scrape copyrighted song projects.
For reproducible research, use files you are permitted to use and record source,
creator, and license in sources.csv.

sources.csv columns:
  id,title,creator,url,license,source_page,notes,enabled

The tool can download direct project files or ZIP archives listed there. It does
not decide whether a license is sufficient for your research; that provenance is
preserved in metadata/manifest for you to audit.

Examples
--------
Local projects:
  python vocaloid_dataset_generator.py ./vocaloid_sources

Official creator-distributed corpus:
  python vocaloid_dataset_generator.py ./vocaloid_sources \
      --preset official --accept-source-terms

Inspect official preset discoveries without downloading:
  python vocaloid_dataset_generator.py ./vocaloid_sources \
      --preset official --discover-only

Manifest + local projects:
  python vocaloid_dataset_generator.py ./vocaloid_sources \
      --sources-csv sources.csv

List parsed projects and selected tracks:
  python vocaloid_dataset_generator.py ./vocaloid_sources --list

Generate only one project:
  python vocaloid_dataset_generator.py ./vocaloid_sources --song my_song

Then train with the existing trainer:
  python train_style.py dataset/vocaloid --region Vocaloid
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
import time
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, urlunparse
import xml.etree.ElementTree as ET

import mido
import numpy as np
import requests
import soundfile as sf
import yaml
from bs4 import BeautifulSoup


SUPPORTED_EXTS = {".vsqx", ".vsq", ".ustx", ".ust", ".mid", ".midi"}
CONVERT_WITH_UTAFORMATIX = {
    ".vpr", ".svp", ".s5p", ".ccs", ".dv", ".ppsf", ".tssln", ".ufdata"
}

OTOIRO_SPECIAL_URL = "https://otoiro.co.jp/special/"
OTOIRO_TERMS_URL = "https://otoiro.co.jp/s_terms/"
CRUSHER_RESOURCES_URL = "https://ccrusherr.com/resources"
CRUSHER_USAGE_URL = "https://ccrusherr.com/usage"

# Current non-DECO*27 titles sharing OTOIRO's SPECIAL page. Keeping this as a
# deny-list is safer than silently mixing a second producer into a DECO*27 corpus.
OTOIRO_OTHER_CREATOR_TITLES = {
    "おなかすいた", "キュンする。", "釈迦ラカ", "アンハッピージャム",
    "恋愛工場", "ドラキュラブ", "GABI", "吐いちゃうぞ",
}

# Only original Crusher works. Covers/remixes such as BIG BROTHER, LOVE IS WAR,
# ALIEN ALIEN, etc. are deliberately excluded from the built-in preset.
CRUSHER_ORIGINAL_TITLES = {
    "ECHO", "WICKED", "AGAIN", "PROPAGANDA!", "SLEEPLESS NIGHTS",
    "いつまでも (ITSUMADEMO)",
}
DEFAULT_SR = 44100
DEFAULT_BPM = 120.0
DEFAULT_GAP_MS = 18.0
DEFAULT_TICKS_PER_BEAT = 480


@dataclass
class Note:
    start: int
    duration: int
    pitch: int
    lyric: str = ""

    @property
    def end(self) -> int:
        return self.start + self.duration


@dataclass
class Track:
    index: int
    name: str
    notes: list[Note] = field(default_factory=list)


@dataclass
class Project:
    path: Path
    format: str
    resolution: int
    tracks: list[Track]
    source_meta: dict = field(default_factory=dict)


@dataclass
class TrackScore:
    track: Track
    score: float
    monophony: float
    mean_pitch: float
    unique_pitches: int
    total_duration: int


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    out = []
    prev_us = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        elif ch in " -_.()[]{}":
            if out and not prev_us:
                out.append("_")
                prev_us = True
    value = "".join(out).strip("_")
    return value or "song"


def localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def child_text(elem: ET.Element, names: Iterable[str], default: str | None = None) -> str | None:
    wanted = set(names)
    for child in list(elem):
        if localname(child.tag) in wanted:
            return (child.text or "").strip()
    return default


def int_child(elem: ET.Element, names: Iterable[str], default: int = 0) -> int:
    text = child_text(elem, names)
    if text is None or text == "":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def parse_vsqx(path: Path) -> Project:
    root = ET.parse(path).getroot()

    resolution = DEFAULT_TICKS_PER_BEAT
    for elem in root.iter():
        if localname(elem.tag) == "resolution":
            try:
                resolution = max(1, int(float((elem.text or "").strip())))
            except Exception:
                pass
            break

    tracks: list[Track] = []
    track_elems = [e for e in list(root) if localname(e.tag) == "vsTrack"]
    if not track_elems:
        track_elems = [e for e in root.iter() if localname(e.tag) == "vsTrack"]

    for fallback_idx, tr in enumerate(track_elems):
        idx = int_child(tr, ("tNo", "trackNo"), fallback_idx)
        name = child_text(tr, ("name",), f"Track {idx}") or f"Track {idx}"
        notes: list[Note] = []

        parts = [e for e in list(tr) if localname(e.tag) == "vsPart"]
        if not parts:
            parts = [e for e in tr.iter() if localname(e.tag) == "vsPart"]

        for part in parts:
            part_start = int_child(part, ("t", "posTick"), 0)
            for note_elem in [e for e in list(part) if localname(e.tag) == "note"]:
                rel_start = int_child(note_elem, ("t", "posTick"), 0)
                duration = int_child(note_elem, ("dur", "durTick"), 0)
                pitch = int_child(note_elem, ("n", "noteNum"), -1)
                lyric = child_text(note_elem, ("y", "lyric"), "") or ""
                if duration <= 0 or not 0 <= pitch <= 127:
                    continue
                notes.append(Note(part_start + rel_start, duration, pitch, lyric))

        tracks.append(Track(idx, name, sorted(notes, key=lambda n: (n.start, n.pitch))))

    return Project(path=path, format="vsqx", resolution=resolution, tracks=tracks)


def parse_ustx(path: Path) -> Project:
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("USTX root is not a mapping")

    resolution = int(data.get("resolution") or DEFAULT_TICKS_PER_BEAT)
    track_defs = data.get("tracks") or []
    parts = data.get("voice_parts") or []

    by_track: dict[int, list[Note]] = {}
    for part in parts:
        if not isinstance(part, dict):
            continue
        track_no = int(part.get("track_no", 0))
        part_pos = int(part.get("position", 0) or 0)
        for item in part.get("notes") or []:
            if not isinstance(item, dict):
                continue
            start = part_pos + int(item.get("position", 0) or 0)
            duration = int(item.get("duration", 0) or 0)
            pitch = int(item.get("tone", -1) or -1)
            lyric = str(item.get("lyric", "") or "")
            if duration <= 0 or not 0 <= pitch <= 127:
                continue
            by_track.setdefault(track_no, []).append(Note(start, duration, pitch, lyric))

    all_indexes = sorted(set(by_track) | set(range(len(track_defs))))
    tracks: list[Track] = []
    for idx in all_indexes:
        name = f"Track {idx}"
        if idx < len(track_defs) and isinstance(track_defs[idx], dict):
            name = str(track_defs[idx].get("track_name") or name)
        notes = sorted(by_track.get(idx, []), key=lambda n: (n.start, n.pitch))
        tracks.append(Track(idx, name, notes))

    return Project(path=path, format="ustx", resolution=max(1, resolution), tracks=tracks)


def decode_ust(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp932", "shift_jis", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def parse_ust(path: Path) -> Project:
    text = decode_ust(path)
    section_re = re.compile(r"(?m)^\[(#[^\]]+)\]\s*$")
    matches = list(section_re.finditer(text))
    notes: list[Note] = []
    cursor = 0
    tempo = None

    for i, match in enumerate(matches):
        section = match.group(1).upper()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]

        kv = {}
        for line in body.splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

        if "Tempo" in kv:
            try:
                tempo = float(kv["Tempo"])
            except Exception:
                pass

        if not re.fullmatch(r"#\d{4}", section):
            continue

        try:
            length = int(float(kv.get("Length", "0")))
        except ValueError:
            length = 0
        if length <= 0:
            continue

        lyric = kv.get("Lyric", "")
        try:
            pitch = int(float(kv.get("NoteNum", "-1")))
        except ValueError:
            pitch = -1

        is_rest = lyric.strip().lower() in {"r", "rest", "pau", "sil"}
        if not is_rest and 0 <= pitch <= 127:
            notes.append(Note(cursor, length, pitch, lyric))
        cursor += length

    meta = {"tempo": tempo} if tempo else {}
    return Project(
        path=path,
        format="ust",
        resolution=DEFAULT_TICKS_PER_BEAT,
        tracks=[Track(0, path.stem, notes)],
        source_meta=meta,
    )


def midi_track_notes(track: mido.MidiTrack) -> tuple[str, list[Note]]:
    abs_tick = 0
    name = ""
    active: dict[tuple[int, int], list[tuple[int, int]]] = {}
    notes: list[Note] = []

    for msg in track:
        abs_tick += int(msg.time)

        if msg.type == "track_name" and not name:
            name = str(msg.name)
            continue

        if not hasattr(msg, "channel"):
            continue
        if int(msg.channel) == 9:
            continue

        key = None
        if msg.type == "note_on" and int(msg.velocity) > 0:
            key = (int(msg.channel), int(msg.note))
            active.setdefault(key, []).append((abs_tick, int(msg.velocity)))
        elif msg.type in ("note_off", "note_on"):
            key = (int(msg.channel), int(msg.note))
            stack = active.get(key)
            if stack:
                start, _velocity = stack.pop(0)
                if abs_tick > start:
                    notes.append(Note(start, abs_tick - start, int(msg.note), ""))

    return name, sorted(notes, key=lambda n: (n.start, n.pitch))


def parse_midi(path: Path) -> Project:
    midi = mido.MidiFile(path)
    tracks = []
    for i, tr in enumerate(midi.tracks):
        name, notes = midi_track_notes(tr)
        if notes:
            tracks.append(Track(i, name or f"Track {i}", notes))
    return Project(
        path=path,
        format="midi",
        resolution=max(1, int(midi.ticks_per_beat)),
        tracks=tracks,
    )


def parse_project(path: Path) -> Project:
    ext = path.suffix.lower()
    if ext == ".vsqx":
        return parse_vsqx(path)
    if ext == ".ustx":
        return parse_ustx(path)
    if ext == ".ust":
        return parse_ust(path)
    if ext in {".mid", ".midi", ".vsq"}:
        project = parse_midi(path)
        if ext == ".vsq":
            project.format = "vsq"
        return project
    if ext in CONVERT_WITH_UTAFORMATIX:
        raise ValueError(
            f"{ext} is not parsed natively. Convert it to VSQX/USTX with UtaFormatix first."
        )
    raise ValueError(f"Unsupported project format: {ext}")


def monophony_score(notes: list[Note]) -> float:
    if len(notes) <= 1:
        return 1.0
    notes = sorted(notes, key=lambda n: (n.start, n.end))
    overlap = 0
    current_end = notes[0].end
    for note in notes[1:]:
        if note.start < current_end:
            overlap += 1
        current_end = max(current_end, note.end)
    return max(0.0, 1.0 - overlap / max(1, len(notes) - 1))


def score_track(track: Track) -> TrackScore | None:
    if not track.notes:
        return None
    pitches = np.asarray([n.pitch for n in track.notes], dtype=np.float64)
    mono = monophony_score(track.notes)
    mean_pitch = float(np.mean(pitches))
    unique = len(set(int(x) for x in pitches))
    total_dur = int(sum(n.duration for n in track.notes))
    name = track.name.lower()

    name_bonus = 0.0
    if any(x in name for x in (
        "main", "melody", "lead", "vocal", "voice",
        "miku", "rin", "len", "kaito", "meiko", "gumi",
        "メロ", "メロディ", "ボーカル", "ヴォーカル", "ミク", "歌",
    )):
        name_bonus += 45.0
    if any(x in name for x in (
        "harm", "harmony", "chorus", "back", "backing", "sub", "guide",
        "bass", "chord", "accomp", "伴奏", "コード", "ベース", "ドラム",
    )):
        name_bonus -= 35.0

    # Main singing lines should be close to monophonic. Increase this term for
    # full-arrangement official MIDIs where accompaniment tracks can be denser.
    early_bonus = 20.0 / (1.0 + max(0, track.index))
    register_bonus = max(0.0, min(12.0, (mean_pitch - 48.0) * 0.35))
    score = (
        1.55 * len(track.notes)
        + 75.0 * mono
        + 0.35 * unique
        + register_bonus
        + name_bonus
        + early_bonus
    )
    return TrackScore(track, score, mono, mean_pitch, unique, total_dur)


def select_track(project: Project, strategy: str, forced_index: int | None) -> tuple[Track, list[TrackScore]]:
    scored = [s for s in (score_track(t) for t in project.tracks) if s is not None]
    if not scored:
        raise ValueError("No note-bearing vocal/melody track found")

    if forced_index is not None:
        for item in scored:
            if item.track.index == forced_index:
                return item.track, sorted(scored, key=lambda s: s.score, reverse=True)
        raise ValueError(f"Requested track index {forced_index} not found")

    if strategy == "first":
        chosen = min(scored, key=lambda s: s.track.index)
    elif strategy == "most-notes":
        chosen = max(scored, key=lambda s: (len(s.track.notes), s.monophony, -s.track.index))
    elif strategy == "highest":
        chosen = max(scored, key=lambda s: (s.mean_pitch, s.monophony, s.score))
    else:
        chosen = max(scored, key=lambda s: s.score)

    return chosen.track, sorted(scored, key=lambda s: s.score, reverse=True)


def monophonize(notes: list[Note]) -> list[Note]:
    """
    Reduce accidental polyphony/harmony inside the selected track to one line.

    - same-onset chord: keep highest pitch (vocal melody bias)
    - overlap: truncate previous note at next onset
    - repeated identical onsets are retained as distinct events if sequential
    """
    groups: dict[int, list[Note]] = {}
    for note in notes:
        groups.setdefault(note.start, []).append(note)

    selected = []
    for onset in sorted(groups):
        group = groups[onset]
        best = max(group, key=lambda n: (n.pitch, n.duration))
        selected.append(Note(best.start, best.duration, best.pitch, best.lyric))

    out: list[Note] = []
    for note in selected:
        if out and note.start < out[-1].end:
            out[-1].duration = max(1, note.start - out[-1].start)
        if note.duration > 0:
            out.append(note)
    return out


def normalize_notes(notes: list[Note]) -> list[Note]:
    if not notes:
        return []
    first = min(n.start for n in notes)
    return [Note(n.start - first, n.duration, n.pitch, n.lyric) for n in notes]


def midi_to_hz(midi: int) -> float:
    return 440.0 * 2.0 ** ((float(midi) - 69.0) / 12.0)


def synth_tone(freq: float, seconds: float, sr: int) -> np.ndarray:
    n = max(1, int(round(seconds * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    phase = 2.0 * np.pi * freq * t

    # Same neutral reed-like spectrum used by the other Scaleify corpus builders.
    y = (
        0.78 * np.sin(phase)
        + 0.15 * np.sin(2.0 * phase)
        + 0.05 * np.sin(3.0 * phase)
        + 0.02 * np.sin(4.0 * phase)
    )

    attack = min(n, max(1, int(round(0.004 * sr))))
    release = min(n, max(1, int(round(0.004 * sr))))
    env = np.ones(n, dtype=np.float64)

    if attack > 1:
        env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
    if release > 1:
        env[-release:] *= np.linspace(1.0, 0.0, release)

    return (0.78 * y * env).astype(np.float32)


def render_notes(
    notes: list[Note],
    resolution: int,
    sr: int,
    bpm: float,
    gap_ms: float,
) -> np.ndarray:
    if not notes:
        return np.zeros(1, dtype=np.float32)

    sec_per_tick = (60.0 / bpm) / max(1, resolution)
    end_tick = max(n.end for n in notes)
    total_s = end_tick * sec_per_tick + 0.08
    audio = np.zeros(max(1, int(math.ceil(total_s * sr))), dtype=np.float32)
    gap_s = max(0.0, gap_ms / 1000.0)

    for note in notes:
        start_s = note.start * sec_per_tick
        dur_s = max(sec_per_tick, note.duration * sec_per_tick)

        # Critical for Scaleify's current onset-aware WAV trainer:
        # preserve a small gap so C4,C4 re-attacks remain two events.
        gap = min(gap_s, dur_s * 0.12)
        tone_s = max(0.018, dur_s - gap)

        tone = synth_tone(midi_to_hz(note.pitch), tone_s, sr)
        start = int(round(start_s * sr))
        end = min(len(audio), start + len(tone))
        if end > start:
            audio[start:end] += tone[: end - start]

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio *= 0.88 / peak
    return audio.astype(np.float32)


def write_symbolic_csv(path: Path, notes: list[Note], resolution: int, bpm: float) -> None:
    sec_per_tick = (60.0 / bpm) / max(1, resolution)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "start_tick", "duration_tick", "pitch_midi", "lyric",
            "start_seconds", "duration_seconds",
        ])
        for i, n in enumerate(notes):
            writer.writerow([
                i,
                n.start,
                n.duration,
                n.pitch,
                n.lyric,
                round(n.start * sec_per_tick, 6),
                round(n.duration * sec_per_tick, 6),
            ])


def safe_extract_zip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            out = target / member
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)



def _asset_url(tag, base_url: str) -> str | None:
    """Extract a download URL from href/data attributes/very small onclick wrappers."""
    for key in ("href", "data-url", "data-href", "data-download", "data-download-url"):
        value = tag.get(key)
        if value and str(value).strip() and not str(value).strip().startswith("#"):
            return urljoin(base_url, str(value).strip())

    onclick = str(tag.get("onclick") or "")
    m = re.search(r"""['"](https?://[^'"]+)['"]""", onclick)
    if m:
        return m.group(1)
    return None


def _otoiro_title_for_midi_anchor(anchor) -> str:
    """
    OTOIRO download cards contain the literal TITLE followed by the song title.
    Search upward for the smallest card that contains TITLE and MIDI.
    """
    node = anchor
    for _ in range(8):
        node = getattr(node, "parent", None)
        if node is None:
            break
        lines = [str(x).strip() for x in node.stripped_strings if str(x).strip()]
        normalized = [x.upper() for x in lines]
        if "TITLE" in normalized and any(x == "MIDI" for x in normalized):
            idx = normalized.index("TITLE")
            for candidate in lines[idx + 1:]:
                uc = candidate.upper()
                if uc not in {
                    "THUMBNAIL", "DOWNLOAD", "MIDI", "INSTRUMENTAL",
                    "LYRIC VIDEO（MP4）", "LYRIC VIDEO(MP4)",
                } and not uc.startswith("MOVIE PJF") and "3D DANCE" not in uc:
                    return candidate
    return ""


def discover_otoiro_deco27_from_html(html_text: str) -> list[dict]:
    soup = BeautifulSoup(html_text, "html.parser")
    records: list[dict] = []
    seen_urls: set[str] = set()

    for anchor in soup.find_all(["a", "button"]):
        label = " ".join(anchor.stripped_strings).strip()
        if label.upper() != "MIDI":
            continue

        url = _asset_url(anchor, OTOIRO_SPECIAL_URL)
        if not url or url in seen_urls:
            continue

        title = _otoiro_title_for_midi_anchor(anchor)
        if title in OTOIRO_OTHER_CREATOR_TITLES:
            continue

        seen_urls.add(url)
        ordinal = len(records) + 1
        stable_id = f"deco27_{ordinal:03d}"
        if title:
            stable_id += "_" + slugify(title)

        records.append({
            "id": stable_id,
            "title": title or f"OTOIRO DECO*27 MIDI {ordinal}",
            "creator": "DECO*27 / OTOIRO",
            "url": url,
            "filename": f"{stable_id}.mid",
            "license": "Official creator asset; use subject to OTOIRO SPECIAL Terms",
            "source_page": OTOIRO_SPECIAL_URL,
            "terms_url": OTOIRO_TERMS_URL,
            "notes": (
                "Official MIDI distributed on OTOIRO SPECIAL. "
                "Keep source asset local unless the applicable terms permit redistribution."
            ),
            "preset": "otoiro-deco27",
        })

    return records


def discover_otoiro_deco27(session: requests.Session, timeout: float) -> list[dict]:
    r = session.get(OTOIRO_SPECIAL_URL, timeout=timeout)
    r.raise_for_status()
    return discover_otoiro_deco27_from_html(r.text)


def _nearest_asset_context(anchor) -> tuple[str, str]:
    """
    Return (asset_text, detected_extension) from the smallest nearby Crusher
    asset block. Download buttons are generic, so the surrounding label is used.
    """
    node = anchor
    for _ in range(7):
        node = getattr(node, "parent", None)
        if node is None:
            break
        txt = " ".join(node.stripped_strings)
        m = re.search(r"\b(VSQX|VSQ|USTX|UST|MIDI|MID)\b", txt, re.I)
        if m:
            raw = m.group(1).lower()
            ext = ".mid" if raw == "midi" else "." + raw
            return txt, ext
    return "", ""


def discover_crusher_from_html(html_text: str) -> list[dict]:
    soup = BeautifulSoup(html_text, "html.parser")
    candidates: dict[str, list[dict]] = {}

    for anchor in soup.find_all(["a", "button"]):
        label = " ".join(anchor.stripped_strings).strip().lower()
        if "download" not in label:
            continue

        url = _asset_url(anchor, CRUSHER_RESOURCES_URL)
        if not url:
            continue

        heading = anchor.find_previous(["h2", "h3"])
        if heading is None:
            continue
        title = " ".join(heading.stripped_strings).strip()
        if title not in CRUSHER_ORIGINAL_TITLES:
            continue

        asset_text, ext = _nearest_asset_context(anchor)
        if not ext:
            # URL filename can still expose the symbolic extension.
            url_ext = Path(urlparse(url).path).suffix.lower()
            if url_ext in SUPPORTED_EXTS:
                ext = url_ext
        if ext not in SUPPORTED_EXTS:
            continue

        sid = "crusher_" + slugify(title)
        candidates.setdefault(title, []).append({
            "id": sid,
            "title": title,
            "creator": "Crusher",
            "url": url,
            "filename": f"{sid}{ext}",
            "license": "Official creator asset; use subject to Crusher Usage Guidelines",
            "source_page": CRUSHER_RESOURCES_URL,
            "terms_url": CRUSHER_USAGE_URL,
            "notes": f"Official original-song symbolic asset. Asset context: {asset_text[:300]}",
            "preset": "crusher",
            "_ext": ext,
        })

    # Avoid counting MIDI + VSQX of the same song as independent training songs.
    preference = {".vsqx": 6, ".vsq": 5, ".mid": 4, ".midi": 4, ".ustx": 3, ".ust": 2}
    records = []
    for title in sorted(candidates):
        best = max(candidates[title], key=lambda x: preference.get(x["_ext"], 0))
        best.pop("_ext", None)
        records.append(best)
    return records


def discover_crusher(session: requests.Session, timeout: float) -> list[dict]:
    r = session.get(CRUSHER_RESOURCES_URL, timeout=timeout)
    r.raise_for_status()
    return discover_crusher_from_html(r.text)


def discover_preset_sources(presets: list[str], timeout: float) -> list[dict]:
    expanded = []
    for preset in presets:
        if preset == "official":
            expanded.extend(["otoiro-deco27", "crusher"])
        else:
            expanded.append(preset)
    # Stable de-duplication while preserving requested order.
    expanded = list(dict.fromkeys(expanded))

    session = requests.Session()
    session.headers.update({"User-Agent": "Scaleify-Vocaloid-Corpus-Builder/2.0"})

    records: list[dict] = []
    for preset in expanded:
        if preset == "otoiro-deco27":
            found = discover_otoiro_deco27(session, timeout)
        elif preset == "crusher":
            found = discover_crusher(session, timeout)
        else:
            raise ValueError(f"Unknown preset: {preset}")
        print(f"Preset discovery: {preset}: {len(found)} symbolic asset(s)")
        records.extend(found)

    # URL is the strongest duplicate key; ID is fallback.
    unique = {}
    for rec in records:
        unique[(rec.get("url") or "", rec.get("id") or "")] = rec
    return list(unique.values())


def _normalized_cloud_url(url: str) -> str:
    """
    Normalize simple direct-download hosts. Google Drive is handled separately
    by gdown because creator links are often /file/d/.../view links.
    """
    parsed = urlparse(url)
    if "dropbox.com" in parsed.netloc:
        q = parse_qs(parsed.query)
        q["dl"] = ["1"]
        return urlunparse(parsed._replace(query=urlencode(q, doseq=True)))
    return url


def _download_asset(
    session: requests.Session,
    url: str,
    path: Path,
    timeout: float,
) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    if "drive.google.com" in url or "docs.google.com" in url:
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "Google Drive asset detected. Install requirements-vocaloid-dataset.txt "
                "(gdown is required)."
            ) from exc
        result = gdown.download(url=url, output=str(tmp), quiet=False, fuzzy=True)
        if not result or not tmp.exists():
            raise RuntimeError(f"gdown failed for {url}")
        tmp.replace(path)
        return

    direct = _normalized_cloud_url(url)
    with session.get(direct, timeout=timeout, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(path)


def provenance_for_path(path: Path, records: list[dict]) -> dict:
    resolved = path.resolve()
    for rec in records:
        root_text = rec.get("extracted_to")
        if not root_text:
            continue
        root = Path(root_text).resolve()
        try:
            resolved.relative_to(root)
            return rec
        except ValueError:
            pass
    return {}

def read_sources_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        enabled = str(row.get("enabled", "1")).strip().lower()
        if enabled in {"0", "false", "no", "n", "off"}:
            continue
        url = str(row.get("url", "")).strip()
        if not url:
            continue
        out.append({k: (v or "").strip() for k, v in row.items()})
    return out


def download_manifest_sources(
    records: list[dict],
    source_dir: Path,
    timeout: float,
    force: bool,
) -> list[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Scaleify-Vocaloid-Corpus-Builder/1.0"})
    downloads = source_dir / "_downloads"
    extracted = source_dir / "_source"
    downloads.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)

    provenance = []

    for i, rec in enumerate(records, start=1):
        url = rec["url"]
        parsed = urlparse(url)
        sid = slugify(rec.get("id") or Path(parsed.path).stem or f"source_{i}")
        candidate = Path(str(rec.get("filename") or "")).name
        if not candidate:
            candidate = Path(parsed.path).name or f"source_{i}"

        ext = Path(candidate).suffix.lower()
        if ext not in SUPPORTED_EXTS and ext != ".zip" and ext not in CONVERT_WITH_UTAFORMATIX:
            candidate = f"{sid}{ext or '.bin'}"

        local = downloads / candidate
        if force or not local.exists():
            print(f"Downloading [{i}/{len(records)}] {rec.get('title') or sid}")
            _download_asset(session, url, local, timeout)

        dest_root = extracted / sid
        dest_root.mkdir(parents=True, exist_ok=True)

        if zipfile.is_zipfile(local):
            if force:
                for p in dest_root.iterdir():
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
            safe_extract_zip(local, dest_root)
        else:
            out = dest_root / local.name
            if force or not out.exists():
                shutil.copy2(local, out)

        provenance.append({
            **rec,
            "downloaded_file": str(local),
            "extracted_to": str(dest_root),
        })

    return provenance


def discover_projects(source_dir: Path) -> tuple[list[Path], list[Path]]:
    supported = []
    convertable = []
    for p in source_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in SUPPORTED_EXTS:
            supported.append(p)
        elif ext in CONVERT_WITH_UTAFORMATIX:
            convertable.append(p)
    return sorted(supported), sorted(convertable)


def make_unique_id(path: Path, source_dir: Path, used: set[str]) -> str:
    base = slugify(path.stem)
    value = base
    n = 2
    while value in used:
        value = f"{base}_{n}"
        n += 1
    used.add(value)
    return value


def metadata_row_for(
    song_id: str,
    project: Project,
    chosen: Track,
    scores: list[TrackScore],
    notes: list[Note],
    wav_path: Path,
    bpm: float,
    provenance: dict | None = None,
) -> dict:
    score_map = {s.track.index: s for s in scores}
    s = score_map[chosen.index]
    total_tick = max(n.end for n in notes) if notes else 0
    duration_s = total_tick * (60.0 / bpm) / max(1, project.resolution)

    provenance = provenance or {}
    return {
        "id": song_id,
        "filename": wav_path.name,
        "source_file": str(project.path),
        "source_title": provenance.get("title", ""),
        "source_creator": provenance.get("creator", ""),
        "source_license": provenance.get("license", ""),
        "source_page": provenance.get("source_page", ""),
        "source_url": provenance.get("url", ""),
        "terms_url": provenance.get("terms_url", ""),
        "preset": provenance.get("preset", ""),
        "format": project.format,
        "resolution": project.resolution,
        "track_index": chosen.index,
        "track_name": chosen.name,
        "track_score": round(s.score, 4),
        "track_monophony": round(s.monophony, 4),
        "track_count": len(project.tracks),
        "note_events": len(notes),
        "pitch_min_midi": min(n.pitch for n in notes),
        "pitch_max_midi": max(n.pitch for n in notes),
        "duration_seconds": round(duration_s, 4),
        "render_bpm": bpm,
        "candidate_tracks": [
            {
                "index": x.track.index,
                "name": x.track.name,
                "notes": len(x.track.notes),
                "score": round(x.score, 4),
                "monophony": round(x.monophony, 4),
                "mean_pitch": round(x.mean_pitch, 3),
            }
            for x in scores
        ],
    }


def write_metadata_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "id", "filename", "source_file",
        "source_title", "source_creator", "source_license", "source_page",
        "source_url", "terms_url", "preset",
        "format", "resolution",
        "track_index", "track_name", "track_monophony", "track_count",
        "note_events", "pitch_min_midi", "pitch_max_midi",
        "duration_seconds", "render_bpm",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Vocaloid/singing-synth symbolic projects into a Scaleify-compatible WAV corpus."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        type=Path,
        default=Path("vocaloid_sources"),
        help="Folder containing VSQX/USTX/UST/MIDI files.",
    )
    parser.add_argument("--output", type=Path, default=Path("dataset/vocaloid"))
    parser.add_argument(
        "--sources-csv",
        type=Path,
        default=None,
        help="Optional manifest of directly downloadable/open project files.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=["otoiro-deco27", "crusher", "official"],
        default=[],
        help=(
            "Discover official creator-distributed symbolic assets. Repeatable. "
            "'official' expands to OTOIRO DECO*27 + Crusher."
        ),
    )
    parser.add_argument(
        "--accept-source-terms",
        action="store_true",
        help=(
            "Required before preset asset downloads. Confirms that you reviewed "
            "the source-site terms; it does not change the underlying license."
        ),
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Discover preset assets and print provenance without downloading them.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--song", default=None, help="Generate only this normalized song id.")
    parser.add_argument(
        "--track-strategy",
        choices=["auto", "first", "most-notes", "highest"],
        default="auto",
    )
    parser.add_argument("--track-index", type=int, default=None)
    parser.add_argument("--min-notes", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    parser.add_argument("--render-bpm", type=float, default=DEFAULT_BPM)
    parser.add_argument("--articulation-gap-ms", type=float, default=DEFAULT_GAP_MS)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.min_notes < 1:
        parser.error("--min-notes must be >= 1")
    if args.sample_rate < 8000:
        parser.error("--sample-rate must be >= 8000")
    if args.render_bpm <= 0:
        parser.error("--render-bpm must be > 0")
    if args.articulation_gap_ms < 0:
        parser.error("--articulation-gap-ms must be >= 0")

    source_dir = args.source_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    symbolic_dir = output / "_symbolic"
    symbolic_dir.mkdir(parents=True, exist_ok=True)

    downloaded_provenance = []
    source_records: list[dict] = []

    if args.preset:
        preset_records = discover_preset_sources(args.preset, args.timeout)
        if args.discover_only:
            print()
            print("Discovered official symbolic assets")
            print("-----------------------------------")
            for rec in preset_records:
                print(
                    f"{rec.get('preset',''):15} "
                    f"{rec.get('id',''):36} "
                    f"{rec.get('title','')}"
                )
                print(f"    source: {rec.get('source_page','')}")
                print(f"    terms:  {rec.get('terms_url','')}")
                print(f"    asset:  {rec.get('url','')}")
            print()
            print(f"Total: {len(preset_records)}")
            return

        if not args.accept_source_terms:
            raise SystemExit(
                "Preset download requires --accept-source-terms. Review:\n"
                f"  OTOIRO:  {OTOIRO_TERMS_URL}\n"
                f"  Crusher: {CRUSHER_USAGE_URL}\n"
                "Use --discover-only to inspect assets without downloading."
            )
        source_records.extend(preset_records)

    if args.sources_csv is not None:
        source_records.extend(read_sources_csv(args.sources_csv))

    if source_records:
        # Stable de-duplication by URL + requested filename.
        deduped = {}
        for rec in source_records:
            deduped[(rec.get("url", ""), rec.get("filename", ""))] = rec
        source_records = list(deduped.values())
        downloaded_provenance = download_manifest_sources(
            source_records, source_dir, args.timeout, args.force
        )

    project_paths, convertable = discover_projects(source_dir)
    if not project_paths:
        raise SystemExit(
            f"No supported project files found under {source_dir}. "
            "Supported: VSQX, VSQ, USTX, UST, MID/MIDI."
        )

    used: set[str] = set()
    indexed: list[tuple[str, Path]] = []
    for project_path in project_paths:
        prov = provenance_for_path(project_path, downloaded_provenance)
        preferred = slugify(prov.get("id", "")) if prov.get("id") else ""
        if preferred:
            song_id = preferred
            candidate = song_id
            suffix = 2
            while candidate in used:
                candidate = f"{song_id}_{suffix}"
                suffix += 1
            used.add(candidate)
            song_id = candidate
        else:
            song_id = make_unique_id(project_path, source_dir, used)
        indexed.append((song_id, project_path))

    if args.song:
        target = slugify(args.song)
        indexed = [(sid, p) for sid, p in indexed if sid == target]
        if not indexed:
            raise SystemExit(f"Song id not found: {target}")

    rows = []
    failures = []

    for ordinal, (song_id, path) in enumerate(indexed, start=1):
        try:
            project = parse_project(path)
            track, scores = select_track(project, args.track_strategy, args.track_index)
            notes = normalize_notes(monophonize(track.notes))

            if len(notes) < args.min_notes:
                raise ValueError(
                    f"Only {len(notes)} usable notes after monophonic reduction "
                    f"(< --min-notes {args.min_notes})"
                )

            if args.list:
                print(
                    f"{song_id:32} {project.format:6} "
                    f"tracks={len(project.tracks):2d} "
                    f"selected={track.index}:{track.name!r} notes={len(notes)}"
                )
                for s in scores[:5]:
                    print(
                        f"    track {s.track.index:2d} "
                        f"notes={len(s.track.notes):4d} mono={s.monophony:.3f} "
                        f"score={s.score:.2f} name={s.track.name!r}"
                    )
                continue

            print(f"[{ordinal:03d}/{len(indexed):03d}] {song_id} <- {path.name}")

            wav_path = output / f"{song_id}.wav"
            symbolic_path = symbolic_dir / f"{song_id}.csv"

            if args.force or not wav_path.exists():
                audio = render_notes(
                    notes,
                    resolution=project.resolution,
                    sr=args.sample_rate,
                    bpm=args.render_bpm,
                    gap_ms=args.articulation_gap_ms,
                )
                sf.write(wav_path, audio, args.sample_rate, subtype="PCM_16")

            write_symbolic_csv(symbolic_path, notes, project.resolution, args.render_bpm)

            prov = provenance_for_path(path, downloaded_provenance)
            row = metadata_row_for(
                song_id, project, track, scores, notes, wav_path, args.render_bpm,
                provenance=prov,
            )
            rows.append(row)

            print(
                f"    {project.format} track={track.index}:{track.name!r} "
                f"notes={len(notes)} mono={row['track_monophony']:.3f} "
                f"range={row['pitch_min_midi']}-{row['pitch_max_midi']} "
                f"-> {wav_path.name}"
            )

        except Exception as exc:
            failures.append({
                "id": song_id,
                "source_file": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"    [FAILED] {path.name}: {type(exc).__name__}: {exc}")

    if args.list:
        if convertable:
            print("\nFiles requiring UtaFormatix conversion:")
            for p in convertable:
                print(f"  {p}")
        return

    write_metadata_csv(output / "metadata.csv", rows)

    manifest = {
        "dataset": "Scaleify Vocaloid / singing-synth melody corpus",
        "generator_version": "2.0",
        "scope": (
            "Monophonic vocal melody extracted from symbolic singing-synth project files. "
            "No lyrics, timbre, tuning curves, accompaniment, or singer identity are used by Scaleify."
        ),
        "supported_native_formats": sorted(SUPPORTED_EXTS),
        "formats_requiring_utaformatix_conversion": sorted(CONVERT_WITH_UTAFORMATIX),
        "rendering": {
            "sample_rate": args.sample_rate,
            "render_bpm": args.render_bpm,
            "articulation_gap_ms": args.articulation_gap_ms,
            "timbre": "Scaleify neutral reed-like synth",
            "purpose": "Compatibility with current Scaleify v10.1 WAV trainer",
        },
        "copyright_note": (
            "Project-file availability does not imply that the underlying composition is freely "
            "redistributable. Keep source/license provenance and use only material you are permitted to use."
        ),
        "official_presets_requested": args.preset,
        "source_terms_acknowledged": bool(args.accept_source_terms),
        "official_source_pages": {
            "otoiro_deco27": OTOIRO_SPECIAL_URL,
            "otoiro_terms": OTOIRO_TERMS_URL,
            "crusher_resources": CRUSHER_RESOURCES_URL,
            "crusher_usage": CRUSHER_USAGE_URL,
        },
        "downloaded_sources": downloaded_provenance,
        "generated_count": len(rows),
        "failed_count": len(failures),
        "songs": rows,
        "failures": failures,
        "conversion_needed": [str(p) for p in convertable],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if failures:
        (output / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if convertable:
        (output / "needs_utaformatix.txt").write_text(
            "\n".join(str(p) for p in convertable) + "\n",
            encoding="utf-8",
        )

    print()
    print("Generation complete")
    print("-------------------")
    print(f"Generated: {len(rows)}")
    print(f"Failed:    {len(failures)}")
    print(f"Output:    {output}")
    print(f"Train:     python train_style.py {output} --region Vocaloid")
    if convertable:
        print(f"Convert with UtaFormatix first: {len(convertable)} file(s)")

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
