#!/usr/bin/env python3
"""
vocaloid_dataset_generator.py

Build a Scaleify-compatible monophonic Vocaloid / singing-synth melody corpus.

Supported source formats
------------------------
Native:
  - .vsqx  VOCALOID3 / VOCALOID4
  - .vsq   VOCALOID2
  - .ustx  OpenUtau
  - .ust   UTAU
  - .mid / .midi  standard or VOCALOID-exported MIDI

For .vpr (VOCALOID5), .ppsf, .svp, etc., convert to .vsqx or .ustx first with
UtaFormatix, then feed the converted files to this tool.

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
import base64
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
from collections import Counter
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

FORMAT_PRIORITY = {
    ".vsqx": 60,
    ".vsq": 50,
    ".mid": 40,
    ".midi": 40,
    ".ustx": 30,
    ".ust": 20,
}

INTERNAL_SOURCE_DIRS = {"_downloads", "_source"}

OTOIRO_SPECIAL_URL = "https://otoiro.co.jp/special/"
OTOIRO_TERMS_URL = "https://otoiro.co.jp/s_terms/"
CRUSHER_RESOURCES_URL = "https://ccrusherr.com/resources"
CRUSHER_USAGE_URL = "https://ccrusherr.com/usage"

OTOIRO_OTHER_CREATOR_TITLES = {
    "おなかすいた", "キュンする。", "釈迦ラカ", "アンハッピージャム",
    "恋愛工場", "ドラキュラブ", "GABI", "吐いちゃうぞ",
}

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


def _is_macos_metadata_path(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._") or path.name == ".DS_Store"


def safe_extract_zip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            if _is_macos_metadata_path(member):
                continue
            out = target / member
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalized_title_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold().strip()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[‐‑‒–—―ー_\-・･:：!！?？'\"“”‘’()（）\[\]【】]", "", value)
    return value


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


def _asset_url(tag, base_url: str) -> str | None:
    node = tag
    for _ in range(5):
        if node is None:
            break
        for key in (
            "href", "data-url", "data-href", "data-download",
            "data-download-url", "data-file", "data-src",
        ):
            value = node.get(key) if hasattr(node, "get") else None
            if value:
                value = str(value).strip()
                if value and not value.startswith("#") and not value.lower().startswith("javascript:"):
                    return urljoin(base_url, value)

        onclick = str(node.get("onclick") or "") if hasattr(node, "get") else ""
        for match in re.finditer(r'''["']([^"']+)["']''', onclick):
            value = match.group(1).strip()
            if not value or value.startswith("#"):
                continue
            if value.startswith(("http://", "https://", "/", "./", "../")):
                return urljoin(base_url, value)

        node = getattr(node, "parent", None)
    return None

def _otoiro_title_for_midi_anchor(anchor) -> str:
    node = anchor
    for _ in range(9):
        node = getattr(node, "parent", None)
        if node is None:
            break
        lines = [str(x).strip() for x in node.stripped_strings if str(x).strip()]
        upper = [x.upper() for x in lines]
        if "TITLE" in upper and "MIDI" in upper:
            idx = upper.index("TITLE")
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
    candidates: list[dict] = []

    for anchor in soup.find_all(["a", "button"]):
        if " ".join(anchor.stripped_strings).strip().upper() != "MIDI":
            continue
        url = _asset_url(anchor, OTOIRO_SPECIAL_URL)
        if not url:
            continue
        title = _otoiro_title_for_midi_anchor(anchor)
        if title in OTOIRO_OTHER_CREATOR_TITLES:
            continue
        candidates.append({
            "title": title or "Untitled OTOIRO MIDI",
            "creator": "DECO*27 / OTOIRO",
            "url": url,
            "license": "Official creator asset; use subject to OTOIRO SPECIAL Terms",
            "source_page": OTOIRO_SPECIAL_URL,
            "terms_url": OTOIRO_TERMS_URL,
            "notes": "Official MIDI distributed on OTOIRO SPECIAL.",
            "preset": "otoiro-deco27",
        })

    # One visible composition title = one training source.
    by_title: dict[str, dict] = {}
    for rec in candidates:
        key = normalized_title_key(rec["title"]) or rec["url"]
        by_title.setdefault(key, rec)

    records = []
    for ordinal, rec in enumerate(by_title.values(), start=1):
        sid = f"deco27_{ordinal:03d}_{slugify(rec['title'])}"
        records.append({**rec, "id": sid, "filename": f"{sid}.mid"})
    return records


def discover_otoiro_deco27(session: requests.Session, timeout: float) -> list[dict]:
    r = session.get(OTOIRO_SPECIAL_URL, timeout=timeout)
    r.raise_for_status()
    return discover_otoiro_deco27_from_html(r.text)



def _decode_possible_base64_path(value: str) -> str | None:
    if not value:
        return None

    raw = value.strip()
    parsed = urlparse(raw)
    token = parsed.path.rstrip("/").split("/")[-1] if parsed.path else raw

    if "." in token or len(token) < 12:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_\-+/=]+", token):
        return None

    for candidate in (token, token.replace("-", "+").replace("_", "/")):
        try:
            padded = candidate + "=" * ((4 - len(candidate) % 4) % 4)
            decoded = base64.b64decode(padded, validate=False).decode("utf-8")
        except Exception:
            continue

        decoded = decoded.strip()
        if decoded and any(
            decoded.lower().endswith(ext)
            for ext in (
                ".wav", ".mp3", ".flac", ".jpg", ".jpeg", ".png", ".txt",
                ".vsqx", ".vsq", ".ustx", ".ust", ".mid", ".midi", ".svp",
            )
        ):
            return decoded

    return None


def _crusher_direct_asset_info(control) -> tuple[str | None, str | None, str]:
    context_parts = []
    node = control

    for _depth in range(3):
        if node is None:
            break

        if hasattr(node, "stripped_strings"):
            context_parts.extend(
                str(x).strip() for x in node.stripped_strings if str(x).strip()
            )

        for key in (
            "href", "data-url", "data-href", "data-download",
            "data-download-url", "data-file", "data-src",
        ):
            value = node.get(key) if hasattr(node, "get") else None
            if not value:
                continue

            raw = str(value).strip()
            if not raw or raw.startswith("#") or raw.lower().startswith("javascript:"):
                continue

            decoded = _decode_possible_base64_path(raw)
            if decoded:
                ext = Path(decoded).suffix.lower()
                if ext not in SUPPORTED_EXTS:
                    return None, None, " ".join(dict.fromkeys(context_parts))
                # Symbolic token, but still browser-only. Do not fake a direct URL.
                return None, ext, " ".join(dict.fromkeys(context_parts))

            direct = urljoin(CRUSHER_RESOURCES_URL, raw)
            ext = Path(urlparse(direct).path).suffix.lower()
            if ext in SUPPORTED_EXTS:
                return direct, ext, " ".join(dict.fromkeys(context_parts))

        node = getattr(node, "parent", None)

    return None, None, " ".join(dict.fromkeys(context_parts))


def _nearest_asset_context(anchor) -> tuple[str, str]:
    node = anchor
    for _ in range(8):
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


def _add_crusher_candidate(candidates, title: str, url: str, ext: str, context: str) -> None:
    if title not in CRUSHER_ORIGINAL_TITLES or ext not in SUPPORTED_EXTS:
        return
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
        "notes": f"Official original-song symbolic asset. {context[:300]}",
        "preset": "crusher",
        "_ext": ext,
    })



def discover_crusher_from_html(html_text: str) -> list[dict]:
    soup = BeautifulSoup(html_text, "html.parser")
    candidates: dict[str, list[dict]] = {}
    opaque_symbolic_seen: list[tuple[str, str]] = []

    for control in soup.find_all(["a", "button"]):
        heading = control.find_previous(["h2", "h3"])
        if heading is None:
            continue

        title = " ".join(heading.stripped_strings).strip()
        if title not in CRUSHER_ORIGINAL_TITLES:
            continue

        direct_url, ext, context = _crusher_direct_asset_info(control)

        if direct_url and ext in SUPPORTED_EXTS:
            _add_crusher_candidate(candidates, title, direct_url, ext, context)
            continue

        if ext in SUPPORTED_EXTS and not direct_url:
            opaque_symbolic_seen.append((title, ext))

    # Strict raw-HTML fallback:
    # only URLs that themselves end with a supported symbolic extension.
    raw = html_text.replace("\\/", "/").replace("\\u0026", "&")
    url_pattern = re.compile(r"https?://[^\"'<>\\\s]+", re.I)

    for title in CRUSHER_ORIGINAL_TITLES:
        for tm in re.finditer(re.escape(title), raw, flags=re.I):
            section = raw[tm.start():min(len(raw), tm.start() + 12000)]

            for um in url_pattern.finditer(section):
                value = um.group(0).rstrip(",)]}")
                ext = Path(urlparse(value).path).suffix.lower()
                if ext not in SUPPORTED_EXTS:
                    continue

                nearby = section[
                    max(0, um.start() - 300):
                    min(len(section), um.end() + 300)
                ]
                _add_crusher_candidate(
                    candidates,
                    title,
                    value,
                    ext,
                    re.sub(
                        r"\s+",
                        " ",
                        BeautifulSoup(nearby, "html.parser").get_text(" ", strip=True),
                    ),
                )

    records = []
    for title in sorted(candidates):
        unique = {}
        for item in candidates[title]:
            unique[(item["url"], item["_ext"])] = item

        best = max(
            unique.values(),
            key=lambda x: FORMAT_PRIORITY.get(x["_ext"], 0),
        )
        best = dict(best)
        best.pop("_ext", None)
        records.append(best)

    if opaque_symbolic_seen and not records:
        print(
            "WARNING: Crusher symbolic assets were found, but the current site "
            "exposes them through browser-only opaque download tokens. "
            "They were skipped instead of constructing invalid 404 URLs.",
            file=sys.stderr,
        )
        for title, ext in sorted(set(opaque_symbolic_seen)):
            print(
                f"    Crusher browser-only asset: {title} ({ext})",
                file=sys.stderr,
            )

    return records


def discover_crusher(session: requests.Session, timeout: float) -> list[dict]:
    r = session.get(CRUSHER_RESOURCES_URL, timeout=timeout)
    r.raise_for_status()
    return discover_crusher_from_html(r.text)


def discover_preset_sources(presets: list[str], timeout: float) -> list[dict]:
    expanded: list[str] = []
    for preset in presets:
        expanded.extend(["otoiro-deco27", "crusher"] if preset == "official" else [preset])
    expanded = list(dict.fromkeys(expanded))

    session = requests.Session()
    session.headers.update({"User-Agent": "Scaleify-Vocaloid-Corpus-Builder/4.1"})
    records: list[dict] = []
    for preset in expanded:
        if preset == "otoiro-deco27":
            found = discover_otoiro_deco27(session, timeout)
        elif preset == "crusher":
            found = discover_crusher(session, timeout)
        else:
            raise ValueError(f"Unknown preset: {preset}")
        print(f"Preset discovery: {preset}: {len(found)} symbolic asset(s)")
        if preset == "crusher" and not found:
            print(
                "WARNING: Crusher preset returned 0 assets. "
                "The resources page layout may have changed; Crusher will not be silently assumed present.",
                file=sys.stderr,
            )
        records.extend(found)

    unique: dict[tuple[str, str], dict] = {}
    for rec in records:
        unique[(rec.get("url", ""), rec.get("id", ""))] = rec
    return list(unique.values())


def _normalized_cloud_url(url: str) -> str:
    parsed = urlparse(url)
    if "dropbox.com" in parsed.netloc:
        q = parse_qs(parsed.query)
        q["dl"] = ["1"]
        return urlunparse(parsed._replace(query=urlencode(q, doseq=True)))
    return url


def _download_asset(session: requests.Session, url: str, path: Path, timeout: float) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    if "drive.google.com" in url or "docs.google.com" in url:
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "Google Drive asset detected. Install requirements-vocaloid-dataset.txt."
            ) from exc
        result = gdown.download(url=url, output=str(tmp), quiet=False, fuzzy=True)
        if not result or not tmp.exists():
            raise RuntimeError(f"gdown failed for {url}")
        tmp.replace(path)
        return

    with session.get(
        _normalized_cloud_url(url),
        timeout=timeout,
        stream=True,
        allow_redirects=True,
    ) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(path)


def download_manifest_sources(
    records: list[dict],
    source_dir: Path,
    timeout: float,
    force: bool,
) -> list[dict]:
    """
    `_downloads` is cache only. Canonical project files are exposed under
    `_source/<source-id>` and only that tree is used for training.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Scaleify-Vocaloid-Corpus-Builder/4.1"})
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
            candidate = Path(parsed.path).name or f"{sid}.bin"

        ext = Path(candidate).suffix.lower()
        if ext not in SUPPORTED_EXTS and ext != ".zip" and ext not in CONVERT_WITH_UTAFORMATIX:
            candidate = f"{sid}{ext or '.bin'}"

        local = downloads / candidate
        if force or not local.exists():
            print(f"Downloading [{i}/{len(records)}] {rec.get('title') or sid}")
            try:
                _download_asset(session, url, local, timeout)
            except requests.RequestException as exc:
                print(
                    f"WARNING: source download skipped: {rec.get('title') or sid}: {exc}",
                    file=sys.stderr,
                )
                provenance.append({
                    **rec,
                    "download_failed": True,
                    "download_error": f"{type(exc).__name__}: {exc}",
                })
                continue

        dest_root = extracted / sid
        if force and dest_root.exists():
            shutil.rmtree(dest_root)
        dest_root.mkdir(parents=True, exist_ok=True)

        if zipfile.is_zipfile(local):
            if force or not any(dest_root.iterdir()):
                safe_extract_zip(local, dest_root)
        else:
            out = dest_root / candidate
            if force or not out.exists():
                shutil.copy2(local, out)

        provenance.append({
            **rec,
            "downloaded_file": str(local),
            "download_sha256": sha256_file(local),
            "extracted_to": str(dest_root),
        })
    return provenance


def _iter_local_projects(source_dir: Path):
    for p in source_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(source_dir)
        if any(part in INTERNAL_SOURCE_DIRS for part in rel.parts):
            continue
        yield p


def _project_files_under(root: Path) -> tuple[list[Path], list[Path]]:
    supported: list[Path] = []
    convertable: list[Path] = []
    if not root.exists():
        return supported, convertable
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p
        if _is_macos_metadata_path(rel):
            continue
        ext = p.suffix.lower()
        if ext in SUPPORTED_EXTS:
            supported.append(p)
        elif ext in CONVERT_WITH_UTAFORMATIX:
            convertable.append(p)
    return supported, convertable


def discover_projects(
    source_dir: Path,
    downloaded_provenance: list[dict] | None = None,
) -> tuple[list[Path], list[Path], list[dict]]:
    """
    Canonical discovery:
      1) local files excluding `_downloads` and `_source`;
      2) downloaded records only from `_source/<id>`;
      3) max one symbolic file per downloaded source record;
      4) exact SHA-256 de-duplication.
    """
    supported: list[Path] = []
    convertable: list[Path] = []
    duplicate_report: list[dict] = []

    for p in _iter_local_projects(source_dir):
        ext = p.suffix.lower()
        if ext in SUPPORTED_EXTS:
            supported.append(p)
        elif ext in CONVERT_WITH_UTAFORMATIX:
            convertable.append(p)

    for rec in downloaded_provenance or []:
        root_text = rec.get("extracted_to")
        if not root_text:
            continue
        found, needs_conversion = _project_files_under(Path(root_text))
        convertable.extend(needs_conversion)
        if not found:
            continue

        found = sorted(
            found,
            key=lambda p: (
                -FORMAT_PRIORITY.get(p.suffix.lower(), 0),
                -p.stat().st_size,
                str(p),
            ),
        )
        chosen = found[0]
        supported.append(chosen)

        for skipped in found[1:]:
            duplicate_report.append({
                "reason": "multiple_symbolic_files_in_one_source",
                "kept": str(chosen),
                "skipped": str(skipped),
                "source_id": rec.get("id", ""),
                "title": rec.get("title", ""),
            })

    by_hash: dict[str, Path] = {}
    unique: list[Path] = []
    for p in sorted(supported):
        digest = sha256_file(p)
        if digest in by_hash:
            duplicate_report.append({
                "reason": "identical_sha256",
                "kept": str(by_hash[digest]),
                "skipped": str(p),
                "sha256": digest,
            })
            continue
        by_hash[digest] = p
        unique.append(p)

    convertable = sorted(dict.fromkeys(p.resolve() for p in convertable))
    return unique, convertable, duplicate_report



def make_unique_id(path: Path, source_dir: Path, used: set[str]) -> str:
    base = slugify(path.stem)
    value = base
    n = 2
    while value in used:
        value = f"{base}_{n}"
        n += 1
    used.add(value)
    return value


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


def clean_generated_output(output: Path) -> None:
    # Remove generated artifacts so old `_2`/`_3` WAVs cannot survive an upgrade.
    for p in output.glob("*.wav"):
        p.unlink(missing_ok=True)
    for name in ("metadata.csv", "manifest.json", "failures.json", "needs_utaformatix.txt"):
        (output / name).unlink(missing_ok=True)
    symbolic = output / "_symbolic"
    if symbolic.exists():
        for p in symbolic.glob("*.csv"):
            p.unlink(missing_ok=True)


@dataclass
class PreparedSong:
    song_id: str
    path: Path
    project: Project
    track: Track
    scores: list[TrackScore]
    notes: list[Note]
    provenance: dict
    marker_removed: dict[int, int] = field(default_factory=dict)


def detect_common_marker_pitches(
    songs: list[PreparedSong],
    min_song_fraction: float = 0.85,
    max_median_event_share: float = 0.02,
    min_median_extreme_gap: float = 9.0,
) -> list[dict]:
    if len(songs) < 4:
        return []

    reports = []
    total = len(songs)
    for pitch in range(128):
        containing = []
        shares = []
        gaps = []
        extreme_hits = 0
        event_total = 0

        for song in songs:
            count = sum(1 for n in song.notes if n.pitch == pitch)
            if count == 0:
                continue
            containing.append(song.song_id)
            shares.append(count / max(1, len(song.notes)))
            event_total += count

            unique = sorted(set(n.pitch for n in song.notes))
            if pitch == unique[0]:
                extreme_hits += 1
                if len(unique) > 1:
                    gaps.append(unique[1] - pitch)
            elif pitch == unique[-1]:
                extreme_hits += 1
                if len(unique) > 1:
                    gaps.append(pitch - unique[-2])

        if not containing:
            continue

        song_fraction = len(containing) / total
        extreme_fraction = extreme_hits / len(containing)
        median_share = float(np.median(shares)) if shares else 1.0
        median_gap = float(np.median(gaps)) if gaps else 0.0

        if (
            song_fraction >= min_song_fraction
            and extreme_fraction >= 0.90
            and median_share <= max_median_event_share
            and median_gap >= min_median_extreme_gap
        ):
            reports.append({
                "pitch_midi": pitch,
                "songs_containing": len(containing),
                "song_fraction": round(song_fraction, 6),
                "extreme_fraction": round(extreme_fraction, 6),
                "median_event_share": round(median_share, 6),
                "median_extreme_gap_semitones": round(median_gap, 3),
                "events_total": event_total,
                "song_ids": containing,
            })
    return reports


def remove_marker_pitches(song: PreparedSong, pitches: set[int]) -> None:
    removed = Counter(n.pitch for n in song.notes if n.pitch in pitches)
    if not removed:
        return
    song.notes = normalize_notes([n for n in song.notes if n.pitch not in pitches])
    song.marker_removed = dict(sorted(removed.items()))


def metadata_row_for(
    song_id: str,
    project: Project,
    chosen: Track,
    scores: list[TrackScore],
    notes: list[Note],
    wav_path: Path,
    bpm: float,
    provenance: dict | None = None,
    marker_removed: dict[int, int] | None = None,
) -> dict:
    score_map = {s.track.index: s for s in scores}
    s = score_map[chosen.index]
    total_tick = max(n.end for n in notes) if notes else 0
    duration_s = total_tick * (60.0 / bpm) / max(1, project.resolution)

    provenance = provenance or {}
    marker_removed = marker_removed or {}
    return {
        "id": song_id,
        "filename": wav_path.name,
        "source_file": str(project.path),
        "source_sha256": sha256_file(project.path),
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
        "marker_notes_removed": {str(k): v for k, v in marker_removed.items()},
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
        "id", "filename", "source_file", "source_sha256",
        "source_title", "source_creator", "source_license",
        "source_page", "source_url", "terms_url", "preset",
        "format", "resolution",
        "track_index", "track_name", "track_monophony", "track_count",
        "note_events", "pitch_min_midi", "pitch_max_midi",
        "duration_seconds", "render_bpm", "marker_notes_removed",
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
        help="'official' = OTOIRO DECO*27 + Crusher.",
    )
    parser.add_argument(
        "--accept-source-terms",
        action="store_true",
        help="Required before downloading preset assets.",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="List preset assets without downloading.",
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
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove previously generated WAV/metadata artifacts before rendering.",
    )
    parser.add_argument(
        "--marker-filter",
        choices=["auto", "off"],
        default="auto",
        help="Conservatively detect and remove corpus-wide sentinel/key-switch notes.",
    )
    parser.add_argument(
        "--marker-song-fraction",
        type=float,
        default=0.85,
        help="Minimum song fraction for auto marker detection (default: 0.85).",
    )
    args = parser.parse_args()

    if args.min_notes < 1:
        parser.error("--min-notes must be >= 1")
    if args.sample_rate < 8000:
        parser.error("--sample-rate must be >= 8000")
    if args.render_bpm <= 0:
        parser.error("--render-bpm must be > 0")
    if args.articulation_gap_ms < 0:
        parser.error("--articulation-gap-ms must be >= 0")
    if not 0.5 <= args.marker_song_fraction <= 1.0:
        parser.error("--marker-song-fraction must be between 0.5 and 1.0")

    source_dir = args.source_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    symbolic_dir = output / "_symbolic"
    symbolic_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        clean_generated_output(output)

    downloaded_provenance: list[dict] = []
    source_records: list[dict] = []

    if args.preset:
        preset_records = discover_preset_sources(args.preset, args.timeout)
        if args.discover_only:
            print()
            print("Discovered official symbolic assets")
            print("-----------------------------------")
            for rec in preset_records:
                print(f"{rec.get('preset',''):15} {rec.get('title','')}")
                print(f"    id:     {rec.get('id','')}")
                print(f"    source: {rec.get('source_page','')}")
                print(f"    terms:  {rec.get('terms_url','')}")
                print(f"    asset:  {rec.get('url','')}")
            print(f"Total: {len(preset_records)}")
            return

        if not args.accept_source_terms:
            raise SystemExit(
                "Preset download requires --accept-source-terms. Review:\n"
                f"  OTOIRO:  {OTOIRO_TERMS_URL}\n"
                f"  Crusher: {CRUSHER_USAGE_URL}"
            )
        source_records.extend(preset_records)

    if args.sources_csv is not None:
        source_records.extend(read_sources_csv(args.sources_csv))

    # Source-record de-duplication by URL and creator/title.
    deduped: list[dict] = []
    seen_urls: set[str] = set()
    seen_titles: set[tuple[str, str]] = set()
    source_record_duplicates: list[dict] = []

    for rec in source_records:
        url = rec.get("url", "")
        title_key = (
            normalized_title_key(rec.get("creator", "")),
            normalized_title_key(rec.get("title", "")),
        )
        if url and url in seen_urls:
            source_record_duplicates.append({
                "reason": "duplicate_source_url",
                "title": rec.get("title", ""),
                "url": url,
            })
            continue
        if all(title_key) and title_key in seen_titles:
            source_record_duplicates.append({
                "reason": "duplicate_creator_title",
                "creator": rec.get("creator", ""),
                "title": rec.get("title", ""),
                "url": url,
            })
            continue

        if url:
            seen_urls.add(url)
        if all(title_key):
            seen_titles.add(title_key)
        deduped.append(rec)

    source_records = deduped
    if source_records:
        downloaded_provenance = download_manifest_sources(
            source_records, source_dir, args.timeout, args.force
        )

    project_paths, convertable, project_duplicates = discover_projects(
        source_dir, downloaded_provenance
    )
    duplicate_report = source_record_duplicates + project_duplicates
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
            n = 2
            while song_id in used:
                song_id = f"{preferred}_{n}"
                n += 1
            used.add(song_id)
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
    prepared: list[PreparedSong] = []

    for song_id, path in indexed:
        try:
            project = parse_project(path)
            track, scores = select_track(project, args.track_strategy, args.track_index)
            notes = normalize_notes(monophonize(track.notes))
            if len(notes) < args.min_notes:
                raise ValueError(
                    f"Only {len(notes)} usable notes after monophonic reduction "
                    f"(< --min-notes {args.min_notes})"
                )
            prepared.append(PreparedSong(
                song_id=song_id,
                path=path,
                project=project,
                track=track,
                scores=scores,
                notes=notes,
                provenance=provenance_for_path(path, downloaded_provenance),
            ))
        except Exception as exc:
            failures.append({
                "id": song_id,
                "source_file": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"    [FAILED] {path.name}: {type(exc).__name__}: {exc}")

    marker_report: list[dict] = []
    if args.marker_filter == "auto":
        # Detect source-specific sentinels per preset/provider group. A marker
        # used by OTOIRO must not cause the same pitch to be deleted from a
        # separate Crusher/local corpus where it may be a legitimate note.
        marker_groups: dict[str, list[PreparedSong]] = {}
        for song in prepared:
            group = (
                song.provenance.get("preset")
                or song.provenance.get("source_page")
                or "__local__"
            )
            marker_groups.setdefault(str(group), []).append(song)

        for group, group_songs in marker_groups.items():
            detected = detect_common_marker_pitches(
                group_songs,
                min_song_fraction=args.marker_song_fraction,
            )
            marker_pitches = {int(item["pitch_midi"]) for item in detected}
            for item in detected:
                item["group"] = group
                marker_report.append(item)
            if marker_pitches:
                print(f"==> Auto marker filter detected for {group}:")
                for item in detected:
                    print(
                        f"    MIDI {item['pitch_midi']}: "
                        f"songs={item['songs_containing']}/{len(group_songs)} "
                        f"median_share={item['median_event_share']:.4f} "
                        f"median_gap={item['median_extreme_gap_semitones']:.1f} st"
                    )
                for song in group_songs:
                    remove_marker_pitches(song, marker_pitches)

    if args.list:
        for song in prepared:
            print(
                f"{song.song_id:32} {song.project.format:6} "
                f"tracks={len(song.project.tracks):2d} "
                f"selected={song.track.index}:{song.track.name!r} notes={len(song.notes)} "
                f"marker_removed={song.marker_removed}"
            )
            for sc in song.scores[:5]:
                print(
                    f"    track {sc.track.index:2d} notes={len(sc.track.notes):4d} "
                    f"mono={sc.monophony:.3f} score={sc.score:.2f} name={sc.track.name!r}"
                )
    else:
        for ordinal, song in enumerate(prepared, start=1):
            if len(song.notes) < args.min_notes:
                failures.append({
                    "id": song.song_id,
                    "source_file": str(song.path),
                    "error": "Too few notes after marker filtering",
                })
                continue

            print(f"[{ordinal:03d}/{len(prepared):03d}] {song.song_id} <- {song.path.name}")
            wav_path = output / f"{song.song_id}.wav"
            symbolic_path = symbolic_dir / f"{song.song_id}.csv"

            if args.force or args.clean_output or song.marker_removed or not wav_path.exists():
                audio = render_notes(
                    song.notes,
                    resolution=song.project.resolution,
                    sr=args.sample_rate,
                    bpm=args.render_bpm,
                    gap_ms=args.articulation_gap_ms,
                )
                sf.write(wav_path, audio, args.sample_rate, subtype="PCM_16")

            write_symbolic_csv(
                symbolic_path,
                song.notes,
                song.project.resolution,
                args.render_bpm,
            )

            row = metadata_row_for(
                song.song_id,
                song.project,
                song.track,
                song.scores,
                song.notes,
                wav_path,
                args.render_bpm,
                provenance=song.provenance,
                marker_removed=song.marker_removed,
            )
            rows.append(row)

            print(
                f"    {song.project.format} track={song.track.index}:{song.track.name!r} "
                f"notes={len(song.notes)} mono={row['track_monophony']:.3f} "
                f"range={row['pitch_min_midi']}-{row['pitch_max_midi']} "
                f"marker_removed={song.marker_removed} -> {wav_path.name}"
            )

    if args.list:
        if convertable:
            print("\nFiles requiring UtaFormatix conversion:")
            for p in convertable:
                print(f"  {p}")
        return

    write_metadata_csv(output / "metadata.csv", rows)

    manifest = {
        "dataset": "Scaleify Vocaloid / singing-synth melody corpus",
        "generator_version": "4.1",
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
        "downloaded_sources": downloaded_provenance,
        "marker_filter": {
            "mode": args.marker_filter,
            "min_song_fraction": args.marker_song_fraction,
            "detected": marker_report,
        },
        "duplicate_sources_skipped": duplicate_report,
        "duplicate_sources_skipped_count": len(duplicate_report),
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
    print(f"Duplicates skipped: {len(duplicate_report)}")
    print(f"Output:    {output}")
    print(f"Train:     python train_style.py {output} --region Vocaloid")
    if convertable:
        print(f"Convert with UtaFormatix first: {len(convertable)} file(s)")

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()