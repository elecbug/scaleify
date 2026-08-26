#!/usr/bin/env python3
"""
Batch extraction of multiple high-confidence melody segments from MP3 files.

Design
------
1. Detect beats in the original mix and infer a likely bar phase.
2. Separate each song into 4 stems with Demucs:
      Vocals / Drums / Bass / Other
3. Discard Drums and Bass from melody candidates.
4. Apply HPSS to Other (and optionally Mix) to suppress residual percussion.
5. Analyze short bar-aligned windows, not the whole song as one melody.
6. Transcribe each candidate source with Spotify Basic Pitch.
7. Decode Basic Pitch RAW note/onset/contour activations directly at native
   model-frame resolution.
8. Fuse neural onset, inferred frame-rise onset, and waveform onset evidence.
9. Collapse polyphonic activations into one melody line with Viterbi.
10. Preserve same-pitch re-attacks from the fused onset posterior.
11. Keep only high-confidence segments.
10. Resynthesize each accepted segment as a clean monophonic WAV.
11. Reassemble accepted segments on the original song timeline into a
    per-song total WAV.
11. Preserve source-separation stems for inspection/reuse.
12. Write metadata.csv and rejected.csv.

Default segmentation:
    4 beats/bar
    2 bars/output segment
    1 bar left + 1 bar right context
    non-overlapping 2-bar stride

Example:
    python extract_melody_segments.py ./mp3 -o ./melody_dataset

Dependencies:
    pip install "numpy<2" scipy librosa soundfile basic-pitch         audio-separator onnxruntime "setuptools<82"

System dependency:
    ffmpeg

Notes:
- Default separation model: htdemucs_ft.yaml (4 stems).
- Drums and Bass are never sent to Basic Pitch.
- Other is additionally processed by librosa HPSS by default.
- Basic Pitch may use TensorFlow / TFLite / ONNX depending on installation.
- The primary decoder uses raw model outputs (note/onset/contour), not decoded
  Basic Pitch note-events. Event decoding remains only as a compatibility
  fallback if raw outputs are unavailable.
- If source separation is unavailable, use --no-separation.
- Separated stems are preserved under <output>/_stems/.
- Reassembled per-song outputs are written under <output>/total/.
"""

from __future__ import annotations

import os

# Compatibility for trusted legacy Demucs checkpoints on PyTorch >= 2.6.
os.environ.setdefault(
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
    "1",
)

import argparse
import csv
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf


ANALYSIS_SR = 22050
OUTPUT_SR = 44100
BEAT_HOP = 512
REST_STATE = 128


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    confidence: float


@dataclass
class CandidateResult:
    source_name: str
    notes: list[Note]
    score: float
    raw_score: float
    source_bias: float
    voiced_ratio: float
    mean_confidence: float
    large_jump_ratio: float
    octave_jump_ratio: float
    note_count: int


@dataclass
class BasicPitchPrediction:
    model_output: dict[str, np.ndarray]
    note_events: list[tuple]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract multiple bar-aligned monophonic melody WAVs from MP3 files."
    )

    p.add_argument("input_dir", type=Path)
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("--recursive", action="store_true")

    p.add_argument("--beats-per-bar", type=int, default=4)
    p.add_argument("--segment-bars", type=int, default=2)
    p.add_argument("--stride-bars", type=int, default=2)
    p.add_argument("--context-bars", type=int, default=1)

    p.add_argument(
        "--separator-model",
        default="htdemucs_ft.yaml",
        help=(
            "python-audio-separator model filename. "
            "Default: htdemucs_ft.yaml (Vocals/Drums/Bass/Other)."
        ),
    )
    p.add_argument(
        "--no-separation",
        action="store_true",
        help=(
            "Skip 4-stem separation. The extractor will use "
            "HPSS(mix) and mix candidates only."
        ),
    )
    p.add_argument(
        "--no-hpss-other",
        action="store_true",
        help="Do not create an HPSS harmonic candidate from the Other stem.",
    )
    p.add_argument(
        "--no-hpss-mix",
        action="store_true",
        help="Do not create an HPSS harmonic fallback from the original mix.",
    )
    p.add_argument(
        "--hpss-margin",
        type=float,
        default=2.0,
        help=(
            "HPSS harmonic/percussive separation margin. "
            "Larger values suppress percussion more aggressively. Default: 2.0"
        ),
    )
    p.add_argument(
        "--no-raw-other",
        action="store_true",
        help="Do not use the raw Other stem as a melody candidate.",
    )
    p.add_argument(
        "--raw-mix-fallback",
        action="store_true",
        help=(
            "Also analyze the untouched full mix. Disabled by default because "
            "drums can generate false note onsets."
        ),
    )

    p.add_argument("--min-midi", type=int, default=45, help="Lowest melody MIDI pitch.")
    p.add_argument("--max-midi", type=int, default=96, help="Highest melody MIDI pitch.")
    p.add_argument("--onset-threshold", type=float, default=0.50)
    p.add_argument("--frame-threshold", type=float, default=0.30)
    p.add_argument("--minimum-note-ms", type=float, default=80.0)

    p.add_argument("--viterbi-fps", type=float, default=50.0)
    p.add_argument("--min-output-note-ms", type=float, default=90.0)
    p.add_argument("--bridge-gap-ms", type=float, default=70.0)
    p.add_argument(
        "--retrigger-min-ms",
        type=float,
        default=70.0,
        help=(
            "Minimum spacing between same-pitch re-attack boundaries derived "
            "from Basic Pitch note onsets. Default: 70 ms."
        ),
    )
    p.add_argument(
        "--retrigger-pitch-tolerance",
        type=int,
        default=0,
        help=(
            "Allow an onset to retrigger the Viterbi note when its Basic Pitch "
            "pitch differs by at most this many semitones. Default: 0."
        ),
    )
    p.add_argument(
        "--raw-active-threshold",
        type=float,
        default=0.12,
        help=(
            "Minimum raw Basic Pitch note salience considered by Viterbi. "
            "Default: 0.12."
        ),
    )
    p.add_argument(
        "--raw-retrigger-threshold",
        type=float,
        default=0.30,
        help=(
            "Threshold for fused raw onset evidence when splitting repeated "
            "same-pitch notes. Default: 0.30."
        ),
    )
    p.add_argument(
        "--waveform-onset-weight",
        type=float,
        default=0.15,
        help=(
            "Weight of waveform spectral-onset evidence in the fused retrigger "
            "score. It is gated by neural evidence. Default: 0.15."
        ),
    )

    p.add_argument("--min-score", type=float, default=0.58)
    p.add_argument("--min-voiced-ratio", type=float, default=0.18)
    p.add_argument("--min-mean-confidence", type=float, default=0.35)
    p.add_argument("--min-notes", type=int, default=3)

    p.add_argument(
        "--keep-events",
        action="store_true",
        help="Write a JSON note-event file beside every accepted WAV.",
    )
    p.add_argument(
        "--no-total",
        action="store_true",
        help=(
            "Do not rebuild accepted segments into a per-song total WAV. "
            "By default, a total WAV is written under <output>/total/."
        ),
    )

    return p.parse_args()


def midi_to_hz(midi: int | float) -> float:
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def find_mp3s(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.mp3" if recursive else "*.mp3"
    return sorted(p for p in root.glob(pattern) if p.is_file())


def scalar_tempo(x) -> float:
    arr = np.asarray(x).reshape(-1)
    return float(arr[0]) if len(arr) else 0.0


def detect_beats_and_bar_phase(
    audio: np.ndarray,
    sr: int,
    beats_per_bar: int,
) -> tuple[float, np.ndarray, int]:
    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sr,
        hop_length=BEAT_HOP,
    )

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=BEAT_HOP,
        units="frames",
        trim=False,
    )

    beat_frames = np.asarray(beat_frames, dtype=int)

    if len(beat_frames) < beats_per_bar * 3:
        raise RuntimeError(
            f"Too few beats detected ({len(beat_frames)})."
        )

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=BEAT_HOP,
    )

    beat_strength = onset_env[
        np.clip(beat_frames, 0, len(onset_env) - 1)
    ]

    # Beat tracker does not directly provide downbeats.
    # Choose the modulo phase whose presumed first beat has the strongest
    # average onset accent. This is simple but generally better than blindly
    # assuming beat_frames[0] is a bar start.
    best_phase = 0
    best_score = -float("inf")

    for phase in range(beats_per_bar):
        idx = np.arange(phase, len(beat_strength), beats_per_bar)
        if len(idx) < 2:
            continue

        downbeat_strength = float(np.mean(beat_strength[idx]))

        # Give a smaller bonus when the candidate downbeat is stronger than
        # the following beat. This helps common 4/4 pop/folk material.
        contrast = 0.0
        next_idx = idx + 1
        valid = next_idx < len(beat_strength)
        if np.any(valid):
            contrast = float(
                np.mean(
                    beat_strength[idx[valid]]
                    - beat_strength[next_idx[valid]]
                )
            )

        score = downbeat_strength + 0.25 * contrast

        if score > best_score:
            best_score = score
            best_phase = phase

    return scalar_tempo(tempo), beat_times, best_phase


def build_segment_windows(
    beat_times: np.ndarray,
    phase: int,
    beats_per_bar: int,
    segment_bars: int,
    stride_bars: int,
    context_bars: int,
) -> list[dict]:
    bar_step = beats_per_bar
    start_beat_indices = list(
        range(phase, len(beat_times), bar_step)
    )

    windows: list[dict] = []

    # Need a following bar boundary to determine end time.
    max_bar = len(start_beat_indices) - 1

    bar_i = 0
    segment_index = 1

    while bar_i + segment_bars <= max_bar:
        center_start_beat = start_beat_indices[bar_i]
        center_end_beat = start_beat_indices[bar_i + segment_bars]

        left_bar = max(0, bar_i - context_bars)
        right_bar = min(max_bar, bar_i + segment_bars + context_bars)

        context_start_beat = start_beat_indices[left_bar]
        context_end_beat = start_beat_indices[right_bar]

        windows.append(
            {
                "segment_index": segment_index,
                "start_bar": bar_i + 1,
                "end_bar": bar_i + segment_bars,
                "center_start": float(beat_times[center_start_beat]),
                "center_end": float(beat_times[center_end_beat]),
                "context_start": float(beat_times[context_start_beat]),
                "context_end": float(beat_times[context_end_beat]),
            }
        )

        segment_index += 1
        bar_i += stride_bars

    return windows


class SeparatorWrapper:
    """
    audio-separator compatibility wrapper.

    This intentionally calls Separator.separate(path) with no custom output
    names so it works with older releases such as audio-separator 0.18.1.
    """

    def __init__(
        self,
        output_dir: Path,
        model_filename: str,
    ):
        from audio_separator.separator import Separator

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.separator = Separator(
            output_dir=str(self.output_dir),
            output_format="WAV",
        )

        self.separator.load_model(
            model_filename=model_filename
        )

    @staticmethod
    def _classify_stem(path: Path) -> str | None:
        name = path.name.lower()

        # Match explicit Demucs stem names first.
        if "vocals" in name or "vocal" in name:
            return "vocals"
        if "drums" in name or "drum" in name:
            return "drums"
        if "bass" in name:
            return "bass"
        if "other" in name:
            return "other"
        if "guitar" in name:
            return "guitar"
        if "piano" in name:
            return "piano"

        # Compatibility with two-stem models if a user explicitly selects one.
        if (
            "instrumental" in name
            or "no_vocal" in name
            or "karaoke" in name
        ):
            return "instrumental"

        return None

    def separate(self, src: Path) -> dict[str, Path]:
        # audio-separator 0.18.1 does not accept the newer
        # custom_output_names keyword. Keep this call minimal.
        returned = self.separator.separate(str(src))

        paths: list[Path] = []

        for item in returned or []:
            p = Path(item)

            if not p.is_absolute():
                p = self.output_dir / p

            if p.exists():
                paths.append(p)

        # Fallback for versions/models that return unexpected path forms.
        if not paths:
            safe_stem = src.stem.lower()
            paths = [
                p
                for p in self.output_dir.glob("*.wav")
                if safe_stem in p.stem.lower()
            ]

        result: dict[str, Path] = {}

        for p in paths:
            stem = self._classify_stem(p)

            if stem is not None:
                result[stem] = p

        return result


class BasicPitchTranscriber:
    def __init__(
        self,
        onset_threshold: float,
        frame_threshold: float,
        minimum_note_ms: float,
        min_midi: int,
        max_midi: int,
    ):
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model

        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_ms = minimum_note_ms
        self.min_midi = int(min_midi)
        self.max_midi = int(max_midi)
        self.min_hz = midi_to_hz(min_midi)
        self.max_hz = midi_to_hz(max_midi)

        # Load once; do not reload the neural network for every segment.
        self.model = Model(ICASSP_2022_MODEL_PATH)

    def predict(self, wav_path: Path) -> BasicPitchPrediction:
        from basic_pitch.inference import predict

        model_output, _, events = predict(
            str(wav_path),
            self.model,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=self.minimum_note_ms,
            minimum_frequency=self.min_hz,
            maximum_frequency=self.max_hz,
            multiple_pitch_bends=False,
            melodia_trick=True,
        )

        normalized: dict[str, np.ndarray] = {}
        if isinstance(model_output, dict):
            for key, value in model_output.items():
                try:
                    normalized[str(key)] = np.asarray(value, dtype=np.float32)
                except Exception:
                    pass

        return BasicPitchPrediction(
            model_output=normalized,
            note_events=list(events),
        )


def crop_audio(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    start = max(0, int(round(start_s * sr)))
    end = min(len(audio), int(round(end_s * sr)))

    if end <= start:
        return np.zeros(1, dtype=np.float32)

    return np.asarray(audio[start:end], dtype=np.float32)



BASIC_PITCH_MIDI_OFFSET = 21
BASIC_PITCH_NOTE_BINS = 88
BASIC_PITCH_CONTOUR_BINS_PER_SEMITONE = 3


def _time_frequency_matrix(
    value: np.ndarray,
    expected_bins: int,
) -> np.ndarray | None:
    """
    Normalize a Basic Pitch posteriorgram to shape (time, frequency).

    Different runtimes may retain singleton batch/channel axes, so squeeze
    them and transpose when the expected frequency-bin count is on axis 0.
    """
    arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        return None

    if arr.shape[1] == expected_bins:
        return arr

    if arr.shape[0] == expected_bins:
        return arr.T

    return None


def infer_frame_rise_onsets(
    note_roll: np.ndarray,
    onset_roll: np.ndarray,
    n_diff: int = 2,
) -> np.ndarray:
    """
    Infer onsets from upward changes in note activation.

    This mirrors Basic Pitch's own decoder idea: compare several preceding
    frame differences, keep positive rises, and rescale them to the predicted
    onset posterior's amplitude range.
    """
    if len(note_roll) == 0:
        return np.zeros_like(note_roll)

    diffs = []

    for n in range(1, n_diff + 1):
        shifted = np.zeros_like(note_roll)
        shifted[n:] = note_roll[:-n]
        diffs.append(note_roll - shifted)

    frame_diff = np.min(
        np.stack(diffs, axis=0),
        axis=0,
    )

    frame_diff = np.maximum(frame_diff, 0.0)
    frame_diff[:n_diff] = 0.0

    diff_max = float(np.max(frame_diff))
    onset_max = float(np.max(onset_roll))

    if diff_max > 1e-12 and onset_max > 1e-12:
        frame_diff *= onset_max / diff_max

    return np.asarray(frame_diff, dtype=np.float32)


def waveform_onset_curve(
    audio: np.ndarray,
    sr: int,
    n_frames: int,
) -> np.ndarray:
    """
    Compute a pitch-agnostic spectral onset curve and interpolate it onto the
    Basic Pitch raw model-frame timeline.
    """
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    onset_env = librosa.onset.onset_strength(
        y=np.asarray(audio, dtype=np.float32),
        sr=sr,
        hop_length=256,
    ).astype(np.float32)

    if len(onset_env) == 0:
        return np.zeros(n_frames, dtype=np.float32)

    positive = onset_env[onset_env > 0]
    scale = (
        float(np.percentile(positive, 95))
        if len(positive)
        else float(np.max(onset_env))
    )

    if scale > 1e-12:
        onset_env = np.clip(onset_env / scale, 0.0, 1.0)
    else:
        onset_env[:] = 0.0

    if len(onset_env) == 1:
        return np.full(
            n_frames,
            float(onset_env[0]),
            dtype=np.float32,
        )

    source_x = np.linspace(
        0.0,
        1.0,
        len(onset_env),
        dtype=np.float64,
    )
    target_x = np.linspace(
        0.0,
        1.0,
        n_frames,
        dtype=np.float64,
    )

    return np.interp(
        target_x,
        source_x,
        onset_env,
    ).astype(np.float32)


def raw_basic_pitch_decoder_inputs(
    prediction: BasicPitchPrediction,
    context_audio: np.ndarray,
    sr: int,
    min_midi: int,
    max_midi: int,
    waveform_weight: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Convert Basic Pitch raw model outputs into:
      - 128 x T melodic salience grid
      - 128 x T fused onset/retrigger grid
      - native effective frames/second

    Note output has 88 piano-key bins (MIDI 21..108).
    Contour output has three sub-bins per semitone and is reduced to a
    semitone-level confidence via max pooling.
    """
    model_output = prediction.model_output

    if "note" not in model_output or "onset" not in model_output:
        return None

    notes = _time_frequency_matrix(
        model_output["note"],
        BASIC_PITCH_NOTE_BINS,
    )
    onsets = _time_frequency_matrix(
        model_output["onset"],
        BASIC_PITCH_NOTE_BINS,
    )

    if notes is None or onsets is None:
        return None

    t_count = min(len(notes), len(onsets))
    if t_count < 2:
        return None

    notes = np.clip(notes[:t_count], 0.0, 1.0)
    onsets = np.clip(onsets[:t_count], 0.0, 1.0)

    contour_semis = np.zeros_like(notes)

    contour = model_output.get("contour")
    if contour is not None:
        contour_matrix = _time_frequency_matrix(
            contour,
            BASIC_PITCH_NOTE_BINS
            * BASIC_PITCH_CONTOUR_BINS_PER_SEMITONE,
        )

        if contour_matrix is not None and len(contour_matrix) >= t_count:
            contour_matrix = np.clip(
                contour_matrix[:t_count],
                0.0,
                1.0,
            )
            contour_semis = np.max(
                contour_matrix.reshape(
                    t_count,
                    BASIC_PITCH_NOTE_BINS,
                    BASIC_PITCH_CONTOUR_BINS_PER_SEMITONE,
                ),
                axis=2,
            )

    # Melodic pitch salience: the note head remains dominant, while contour
    # helps retain a stable melodic path through weaker note activations.
    salience_88 = np.clip(
        0.82 * notes
        + 0.18 * contour_semis
        + 0.04 * onsets,
        0.0,
        1.0,
    )

    frame_rise = infer_frame_rise_onsets(
        note_roll=notes,
        onset_roll=onsets,
        n_diff=2,
    )

    waveform = waveform_onset_curve(
        context_audio,
        sr,
        t_count,
    )

    # Global waveform attacks are useful for repeated synth notes, but drum
    # residuals must not split notes by themselves. Gate the waveform term by
    # neural onset/rise support for each pitch.
    neural_support = np.maximum(onsets, frame_rise)
    waveform_gate = np.clip(
        neural_support / 0.22,
        0.0,
        1.0,
    )

    w_wave = float(np.clip(waveform_weight, 0.0, 0.40))
    w_onset = 0.60
    w_rise = max(0.0, 1.0 - w_onset - w_wave)

    fused_onset_88 = np.clip(
        w_onset * onsets
        + w_rise * frame_rise
        + w_wave * waveform[:, None] * waveform_gate,
        0.0,
        1.0,
    )

    salience = np.zeros((128, t_count), dtype=np.float32)
    fused_onset = np.zeros((128, t_count), dtype=np.float32)

    for idx in range(BASIC_PITCH_NOTE_BINS):
        midi = BASIC_PITCH_MIDI_OFFSET + idx

        if midi < min_midi or midi > max_midi:
            continue

        salience[midi] = salience_88[:, idx]
        fused_onset[midi] = fused_onset_88[:, idx]

    duration_s = len(context_audio) / max(sr, 1)
    fps = (
        float(t_count) / duration_s
        if duration_s > 1e-9
        else 0.0
    )

    if fps <= 0:
        return None

    return salience, fused_onset, fps


def retrigger_boundaries_from_raw(
    path: np.ndarray,
    onset_score: np.ndarray,
    fps: float,
    threshold: float,
    min_spacing_ms: float,
    min_note_ms: float,
    pitch_tolerance: int,
) -> np.ndarray:
    """
    Split same-pitch Viterbi runs at strong local onset maxima.

    Re-trigger decisions are made after short-gap bridging, so a repeated note
    remains separable even when the pitch path itself is continuous.
    """
    n = len(path)
    boundaries = np.zeros(n, dtype=bool)

    if n < 3:
        return boundaries

    min_spacing_frames = max(
        1,
        int(round(min_spacing_ms / 1000.0 * fps)),
    )
    min_note_frames = max(
        1,
        int(round(min_note_ms / 1000.0 * fps)),
    )

    last_split_or_change = 0

    for t in range(1, n - 1):
        state = int(path[t])

        if state == REST_STATE:
            last_split_or_change = t
            continue

        if int(path[t - 1]) != state:
            last_split_or_change = t
            continue

        lo = max(0, state - pitch_tolerance)
        hi = min(127, state + pitch_tolerance)

        center = float(np.max(onset_score[lo:hi + 1, t]))
        if center < threshold:
            continue

        left = float(np.max(onset_score[lo:hi + 1, t - 1]))
        right = float(np.max(onset_score[lo:hi + 1, t + 1]))

        # Local-maximum requirement prevents a broad onset plateau from
        # producing several artificial repeated notes.
        if center < left or center < right:
            continue

        if t - last_split_or_change < min_note_frames:
            continue

        existing = np.flatnonzero(boundaries[:t])
        last_boundary = int(existing[-1]) if len(existing) else -10**9

        if t - last_boundary < min_spacing_frames:
            continue

        boundaries[t] = True
        last_split_or_change = t

    return boundaries


def note_events_to_confidence_grid(
    events: Iterable[tuple],
    duration_s: float,
    fps: float,
) -> np.ndarray:
    n_frames = max(1, int(math.ceil(duration_s * fps)))
    grid = np.zeros((128, n_frames), dtype=np.float32)

    for event in events:
        if len(event) < 4:
            continue

        start, end, pitch, amplitude = event[:4]

        pitch = int(round(float(pitch)))
        amplitude = float(amplitude)

        if not (0 <= pitch <= 127):
            continue
        if end <= start:
            continue

        i0 = max(0, int(math.floor(float(start) * fps)))
        i1 = min(n_frames, int(math.ceil(float(end) * fps)))

        if i1 <= i0:
            continue

        grid[pitch, i0:i1] = np.maximum(
            grid[pitch, i0:i1],
            amplitude,
        )

    return grid



def note_events_to_onset_grid(
    events: Iterable[tuple],
    duration_s: float,
    fps: float,
) -> np.ndarray:
    """
    Convert Basic Pitch note-event starts into a pitch x frame onset grid.

    Unlike the confidence grid, this preserves repeated attacks of the same
    MIDI pitch. A sequence such as G-G-G therefore retains three boundaries
    even when the Viterbi pitch path itself remains continuously on G.
    """
    n_frames = max(1, int(math.ceil(duration_s * fps)))
    onset_grid = np.zeros((128, n_frames), dtype=np.float32)

    for event in events:
        if len(event) < 4:
            continue

        start, end, pitch, amplitude = event[:4]
        pitch = int(round(float(pitch)))
        amplitude = float(amplitude)

        if not (0 <= pitch <= 127):
            continue
        if float(end) <= float(start):
            continue

        frame = int(round(float(start) * fps))
        frame = int(np.clip(frame, 0, n_frames - 1))

        onset_grid[pitch, frame] = max(
            float(onset_grid[pitch, frame]),
            amplitude,
        )

    return onset_grid


def retrigger_boundaries_from_onsets(
    path: np.ndarray,
    onset_grid: np.ndarray,
    fps: float,
    min_spacing_ms: float,
    pitch_tolerance: int,
    onset_threshold: float = 0.08,
) -> np.ndarray:
    """
    Return a boolean frame mask marking re-attack boundaries.

    A boundary is accepted when:
    - the Viterbi state is a sounding note,
    - Basic Pitch reports a note onset at the same pitch (or within the
      configured tolerance),
    - the boundary is not the first frame of a newly changed Viterbi pitch,
    - it is sufficiently far from the previous retrigger boundary.

    The second condition is what prevents repeated same-pitch notes from
    collapsing into one sustained event.
    """
    n = len(path)
    boundaries = np.zeros(n, dtype=bool)

    if n == 0:
        return boundaries

    min_frames = max(
        1,
        int(round(min_spacing_ms / 1000.0 * fps)),
    )

    last_boundary = -min_frames

    for t in range(1, n):
        state = int(path[t])

        if state == REST_STATE:
            continue

        # Pitch changes are already split naturally by path_to_notes().
        # We only need extra boundaries while the same pitch continues.
        if int(path[t - 1]) != state:
            continue

        lo = max(0, state - pitch_tolerance)
        hi = min(127, state + pitch_tolerance)

        if float(np.max(onset_grid[lo:hi + 1, t])) < onset_threshold:
            continue

        if t - last_boundary < min_frames:
            continue

        boundaries[t] = True
        last_boundary = t

    return boundaries


def transition_cost(prev_state: int, state: int) -> float:
    if prev_state == REST_STATE and state == REST_STATE:
        return 0.0

    if prev_state == REST_STATE or state == REST_STATE:
        return 0.16

    d = abs(state - prev_state)

    if d == 0:
        return 0.0
    if d <= 2:
        return 0.035 * d
    if d <= 5:
        return 0.09 * d

    cost = 0.45 + 0.08 * (d - 5)

    # Single-frame octave/harmonic mistakes are common in transcription.
    if d in (12, 24):
        cost += 0.55

    return cost


def viterbi_monophonic_path(
    grid: np.ndarray,
    active_threshold: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = grid.shape[1]

    candidate_states: list[list[int]] = []

    for t in range(n_frames):
        active = np.where(
            grid[:, t] >= active_threshold
        )[0].tolist()

        # Keep computation bounded in dense polyphonic material:
        # retain the strongest 12 pitch candidates at each frame.
        if len(active) > 12:
            active = sorted(
                active,
                key=lambda p: float(grid[p, t]),
                reverse=True,
            )[:12]

        active.append(REST_STATE)
        candidate_states.append(active)

    back: list[dict[int, int]] = []
    prev_scores: dict[int, float] = {}

    for state in candidate_states[0]:
        if state == REST_STATE:
            peak = float(np.max(grid[:, 0]))
            emission = 0.30 if peak < active_threshold else 0.02
        else:
            emission = 2.2 * float(grid[state, 0])

        prev_scores[state] = emission

    back.append({state: -1 for state in prev_scores})

    for t in range(1, n_frames):
        current_scores: dict[int, float] = {}
        current_back: dict[int, int] = {}

        peak = float(np.max(grid[:, t]))

        for state in candidate_states[t]:
            if state == REST_STATE:
                emission = (
                    0.32
                    if peak < active_threshold
                    else 0.015
                )
            else:
                emission = 2.2 * float(grid[state, t])

            best_prev = None
            best_score = -float("inf")

            for prev_state, prev_score in prev_scores.items():
                score = (
                    prev_score
                    + emission
                    - transition_cost(prev_state, state)
                )

                if score > best_score:
                    best_score = score
                    best_prev = prev_state

            current_scores[state] = best_score
            current_back[state] = int(best_prev)

        prev_scores = current_scores
        back.append(current_back)

    state = max(prev_scores, key=prev_scores.get)

    path = np.full(n_frames, REST_STATE, dtype=np.int16)

    for t in range(n_frames - 1, -1, -1):
        path[t] = state
        if t > 0:
            state = back[t][state]

    confidence = np.zeros(n_frames, dtype=np.float32)

    for t, state in enumerate(path):
        if state != REST_STATE:
            confidence[t] = grid[state, t]

    return path, confidence


def bridge_short_gaps(
    path: np.ndarray,
    max_gap_frames: int,
    protected_boundaries: np.ndarray | None = None,
) -> np.ndarray:
    out = path.copy()
    n = len(out)
    i = 0

    while i < n:
        if out[i] != REST_STATE:
            i += 1
            continue

        j = i
        while j < n and out[j] == REST_STATE:
            j += 1

        gap_len = j - i

        protected = False
        if protected_boundaries is not None:
            left = max(0, i)
            right = min(len(protected_boundaries), j + 1)
            protected = bool(np.any(protected_boundaries[left:right]))

        if (
            gap_len <= max_gap_frames
            and i > 0
            and j < n
            and out[i - 1] == out[j]
            and out[i - 1] != REST_STATE
            and not protected
        ):
            out[i:j] = out[i - 1]

        i = j

    return out


def path_to_notes(
    path: np.ndarray,
    confidence: np.ndarray,
    fps: float,
    min_note_ms: float,
    retrigger_boundaries: np.ndarray | None = None,
) -> list[Note]:
    notes: list[Note] = []
    n = len(path)
    i = 0

    while i < n:
        state = int(path[i])

        if state == REST_STATE:
            i += 1
            continue

        j = i + 1
        while j < n and int(path[j]) == state:
            if (
                retrigger_boundaries is not None
                and bool(retrigger_boundaries[j])
            ):
                break
            j += 1

        start = i / fps
        end = j / fps
        duration_ms = (end - start) * 1000.0

        if duration_ms >= min_note_ms:
            conf_values = confidence[i:j]
            conf_values = conf_values[conf_values > 0]

            conf = (
                float(np.mean(conf_values))
                if len(conf_values)
                else 0.0
            )

            notes.append(
                Note(
                    start=start,
                    end=end,
                    pitch=state,
                    confidence=conf,
                )
            )

        i = j

    return notes


def crop_notes_to_center(
    notes: list[Note],
    center_offset_s: float,
    center_duration_s: float,
) -> list[Note]:
    cropped: list[Note] = []

    center_end = center_offset_s + center_duration_s

    for note in notes:
        start = max(note.start, center_offset_s)
        end = min(note.end, center_end)

        if end <= start:
            continue

        cropped.append(
            Note(
                start=start - center_offset_s,
                end=end - center_offset_s,
                pitch=note.pitch,
                confidence=note.confidence,
            )
        )

    return cropped


def quality_metrics(
    notes: list[Note],
    duration_s: float,
) -> tuple[float, float, float, float, int, float]:
    if duration_s <= 0 or not notes:
        return 0.0, 0.0, 1.0, 1.0, 0, 0.0

    voiced = sum(max(0.0, n.end - n.start) for n in notes)
    voiced_ratio = min(1.0, voiced / duration_s)

    weighted_conf_num = sum(
        n.confidence * max(0.0, n.end - n.start)
        for n in notes
    )
    mean_confidence = (
        weighted_conf_num / voiced
        if voiced > 0
        else 0.0
    )

    jumps = [
        abs(notes[i].pitch - notes[i - 1].pitch)
        for i in range(1, len(notes))
    ]

    if jumps:
        large_jump_ratio = float(
            np.mean(np.asarray(jumps) >= 8)
        )
        octave_jump_ratio = float(
            np.mean(
                np.isin(
                    np.asarray(jumps),
                    [12, 24],
                )
            )
        )
    else:
        large_jump_ratio = 0.0
        octave_jump_ratio = 0.0

    note_count = len(notes)

    # Quality score intentionally rewards clean, sustained melodic evidence
    # and penalizes implausible jump-heavy paths.
    score = (
        0.42 * np.clip(mean_confidence, 0.0, 1.0)
        + 0.28 * np.clip(voiced_ratio / 0.55, 0.0, 1.0)
        + 0.15 * np.clip(note_count / 6.0, 0.0, 1.0)
        + 0.10 * (1.0 - large_jump_ratio)
        + 0.05 * (1.0 - octave_jump_ratio)
    )

    return (
        float(voiced_ratio),
        float(mean_confidence),
        float(large_jump_ratio),
        float(octave_jump_ratio),
        note_count,
        float(score),
    )


def analyze_candidate(
    source_name: str,
    source_audio: np.ndarray,
    sr: int,
    window: dict,
    transcriber: BasicPitchTranscriber,
    fps: float,
    bridge_gap_ms: float,
    min_output_note_ms: float,
    retrigger_min_ms: float,
    retrigger_pitch_tolerance: int,
    raw_active_threshold: float,
    raw_retrigger_threshold: float,
    waveform_onset_weight: float,
    tmp_dir: Path,
) -> CandidateResult:
    context_start = window["context_start"]
    context_end = window["context_end"]

    context = crop_audio(
        source_audio,
        sr,
        context_start,
        context_end,
    )

    if len(context) < int(0.25 * sr):
        bias = source_bias(source_name)
        return CandidateResult(
            source_name=source_name,
            notes=[],
            score=bias,
            raw_score=0.0,
            source_bias=bias,
            voiced_ratio=0.0,
            mean_confidence=0.0,
            large_jump_ratio=1.0,
            octave_jump_ratio=1.0,
            note_count=0,
        )

    context_wav = (
        tmp_dir
        / f"segment_{window['segment_index']:04d}_{source_name}.wav"
    )

    sf.write(
        context_wav,
        context,
        sr,
        subtype="PCM_16",
    )

    prediction = transcriber.predict(context_wav)

    duration_s = len(context) / sr

    raw_inputs = raw_basic_pitch_decoder_inputs(
        prediction=prediction,
        context_audio=context,
        sr=sr,
        min_midi=transcriber.min_midi,
        max_midi=transcriber.max_midi,
        waveform_weight=waveform_onset_weight,
    )

    if raw_inputs is not None:
        grid, fused_onset, decoder_fps = raw_inputs

        path, frame_conf = viterbi_monophonic_path(
            grid,
            active_threshold=raw_active_threshold,
        )

        bridge_frames = max(
            0,
            int(round(bridge_gap_ms / 1000.0 * decoder_fps)),
        )

        path = bridge_short_gaps(
            path,
            max_gap_frames=bridge_frames,
        )

        retrigger_boundaries = retrigger_boundaries_from_raw(
            path=path,
            onset_score=fused_onset,
            fps=decoder_fps,
            threshold=raw_retrigger_threshold,
            min_spacing_ms=retrigger_min_ms,
            min_note_ms=min_output_note_ms,
            pitch_tolerance=retrigger_pitch_tolerance,
        )

        fps = decoder_fps

    else:
        # Compatibility fallback for Basic Pitch runtimes that do not expose
        # raw note/onset posteriorgrams in a usable shape.
        events = prediction.note_events

        grid = note_events_to_confidence_grid(
            events,
            duration_s=duration_s,
            fps=fps,
        )

        onset_grid = note_events_to_onset_grid(
            events,
            duration_s=duration_s,
            fps=fps,
        )

        path, frame_conf = viterbi_monophonic_path(grid)

        bridge_frames = max(
            0,
            int(round(bridge_gap_ms / 1000.0 * fps)),
        )

        path = bridge_short_gaps(
            path,
            max_gap_frames=bridge_frames,
        )

        retrigger_boundaries = retrigger_boundaries_from_onsets(
            path=path,
            onset_grid=onset_grid,
            fps=fps,
            min_spacing_ms=retrigger_min_ms,
            pitch_tolerance=retrigger_pitch_tolerance,
        )

    # Rebuild confidence after bridging.
    for t, state in enumerate(path):
        if state == REST_STATE:
            frame_conf[t] = 0.0
        elif frame_conf[t] <= 0:
            frame_conf[t] = grid[state, t]

    notes = path_to_notes(
        path,
        frame_conf,
        fps=fps,
        min_note_ms=min_output_note_ms,
        retrigger_boundaries=retrigger_boundaries,
    )

    center_offset = (
        window["center_start"]
        - window["context_start"]
    )
    center_duration = (
        window["center_end"]
        - window["center_start"]
    )

    notes = crop_notes_to_center(
        notes,
        center_offset_s=center_offset,
        center_duration_s=center_duration,
    )

    (
        voiced_ratio,
        mean_confidence,
        large_jump_ratio,
        octave_jump_ratio,
        note_count,
        score,
    ) = quality_metrics(
        notes,
        duration_s=center_duration,
    )

    bias = source_bias(source_name)

    return CandidateResult(
        source_name=source_name,
        notes=notes,
        score=score + bias,
        raw_score=score,
        source_bias=bias,
        voiced_ratio=voiced_ratio,
        mean_confidence=mean_confidence,
        large_jump_ratio=large_jump_ratio,
        octave_jump_ratio=octave_jump_ratio,
        note_count=note_count,
    )


def synthesize_notes(
    notes: list[Note],
    duration_s: float,
    sr: int = OUTPUT_SR,
) -> np.ndarray:
    n_samples = max(1, int(math.ceil(duration_s * sr)))
    out = np.zeros(n_samples, dtype=np.float64)

    for note in notes:
        start_i = max(0, int(round(note.start * sr)))
        end_i = min(n_samples, int(round(note.end * sr)))

        if end_i <= start_i:
            continue

        n = end_i - start_i
        t = np.arange(n, dtype=np.float64) / sr
        freq = midi_to_hz(note.pitch)

        phase = 2.0 * np.pi * freq * t

        # Neutral, fundamental-dominant synthetic melody.
        tone = (
            0.82 * np.sin(phase)
            + 0.12 * np.sin(2.0 * phase)
            + 0.06 * np.sin(3.0 * phase)
        )

        env = np.ones(n, dtype=np.float64)
        fade = min(
            n // 2,
            max(1, int(0.008 * sr)),
        )

        if fade > 1:
            env[:fade] *= np.linspace(
                0.0, 1.0, fade, endpoint=False
            )
            env[-fade:] *= np.linspace(
                1.0, 0.0, fade
            )

        amp = 0.45 + 0.45 * np.clip(
            note.confidence,
            0.0,
            1.0,
        )

        out[start_i:end_i] += tone * env * amp

    peak = float(np.max(np.abs(out)))

    if peak > 0:
        out = out / peak * 0.86

    return out.astype(np.float32)


def accepted(
    result: CandidateResult,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    reasons = []

    if result.raw_score < args.min_score:
        reasons.append(f"score<{args.min_score:.2f}")

    if result.voiced_ratio < args.min_voiced_ratio:
        reasons.append(
            f"voiced<{args.min_voiced_ratio:.2f}"
        )

    if result.mean_confidence < args.min_mean_confidence:
        reasons.append(
            f"confidence<{args.min_mean_confidence:.2f}"
        )

    if result.note_count < args.min_notes:
        reasons.append(f"notes<{args.min_notes}")

    return (len(reasons) == 0, ";".join(reasons))


def load_mono(path: Path, sr: int = ANALYSIS_SR) -> np.ndarray:
    y, _ = librosa.load(
        path,
        sr=sr,
        mono=True,
    )
    return np.asarray(y, dtype=np.float32)


def hpss_harmonic(
    audio: np.ndarray,
    margin: float,
) -> np.ndarray:
    """
    Return the harmonic component of an audio signal.

    The input has already gone through source separation in the normal path,
    so HPSS is used only as a second-stage residual-percussion suppressor.
    """
    if len(audio) == 0:
        return audio.astype(np.float32, copy=True)

    harmonic, _ = librosa.effects.hpss(
        np.asarray(audio, dtype=np.float32),
        margin=(float(margin), float(margin)),
    )

    harmonic = np.asarray(harmonic, dtype=np.float32)

    peak = float(np.max(np.abs(harmonic))) if len(harmonic) else 0.0
    if peak > 1.0:
        harmonic = harmonic / peak

    return harmonic


def source_bias(source_name: str) -> float:
    """
    Small ranking prior.

    Cleaned stems are preferred when acoustic quality is otherwise similar.
    The bias is deliberately small; a clearly better transcription from
    another source can still win.
    """
    biases = {
        "other_harmonic": 0.045,
        "vocals": 0.035,
        "guitar": 0.025,
        "piano": 0.025,
        "mix_harmonic": 0.010,
        "other": 0.000,
        "instrumental_harmonic": 0.000,
        "instrumental": -0.020,
        "mix": -0.060,
    }
    return float(biases.get(source_name, 0.0))


def write_events_json(
    path: Path,
    source_file: str,
    source_kind: str,
    window: dict,
    result: CandidateResult,
) -> None:
    payload = {
        "source_file": source_file,
        "source_kind": source_kind,
        "start_bar": window["start_bar"],
        "end_bar": window["end_bar"],
        "start_s": window["center_start"],
        "end_s": window["center_end"],
        "quality": {
            "score": result.raw_score,
            "ranking_score": result.score,
            "source_bias": result.source_bias,
            "voiced_ratio": result.voiced_ratio,
            "mean_confidence": result.mean_confidence,
            "large_jump_ratio": result.large_jump_ratio,
            "octave_jump_ratio": result.octave_jump_ratio,
            "note_count": result.note_count,
        },
        "notes": [
            {
                "start": n.start,
                "end": n.end,
                "midi": n.pitch,
                "confidence": n.confidence,
            }
            for n in result.notes
        ],
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )



def add_segment_to_timeline(
    mix_sum: np.ndarray,
    weight_sum: np.ndarray,
    segment: np.ndarray,
    start_s: float,
    sr: int,
    fade_ms: float = 8.0,
) -> None:
    """
    Overlap-add one synthesized segment onto the original song timeline.

    If segments overlap, accumulated weights are used later to average them
    instead of increasing loudness at the overlap.
    """
    if len(segment) == 0:
        return

    start_i = max(0, int(round(start_s * sr)))
    if start_i >= len(mix_sum):
        return

    end_i = min(len(mix_sum), start_i + len(segment))
    n = end_i - start_i
    if n <= 0:
        return

    x = np.asarray(segment[:n], dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)

    fade = min(
        n // 2,
        max(0, int(round(fade_ms / 1000.0 * sr))),
    )

    if fade > 1:
        phase = np.linspace(0.0, np.pi / 2.0, fade)
        ramp = np.sin(phase) ** 2
        weights[:fade] = ramp
        weights[-fade:] = ramp[::-1]

    mix_sum[start_i:end_i] += x * weights
    weight_sum[start_i:end_i] += weights


def finalize_total_timeline(
    mix_sum: np.ndarray,
    weight_sum: np.ndarray,
    peak: float = 0.86,
) -> np.ndarray:
    """
    Convert overlap sums into one monophonic total WAV.

    Rejected/uncovered time ranges remain silent.
    """
    out = np.zeros_like(mix_sum, dtype=np.float64)

    covered = weight_sum > 1e-10
    out[covered] = mix_sum[covered] / weight_sum[covered]

    current_peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if current_peak > 0:
        out *= float(peak) / current_peak

    return out.astype(np.float32)


def process_song(
    src: Path,
    input_root: Path,
    output_root: Path,
    separator: SeparatorWrapper | None,
    transcriber: BasicPitchTranscriber,
    args: argparse.Namespace,
    metadata_writer: csv.DictWriter,
    rejected_writer: csv.DictWriter,
) -> tuple[int, int]:
    mix = load_mono(src)

    tempo, beat_times, phase = detect_beats_and_bar_phase(
        mix,
        ANALYSIS_SR,
        args.beats_per_bar,
    )

    windows = build_segment_windows(
        beat_times=beat_times,
        phase=phase,
        beats_per_bar=args.beats_per_bar,
        segment_bars=args.segment_bars,
        stride_bars=args.stride_bars,
        context_bars=args.context_bars,
    )

    if not windows:
        raise RuntimeError("No complete bar windows found.")

    rel_parent = src.parent.relative_to(input_root)
    song_out = output_root / "wav" / rel_parent
    song_out.mkdir(parents=True, exist_ok=True)

    total_out_dir = output_root / "total" / rel_parent
    total_out_dir.mkdir(parents=True, exist_ok=True)

    total_duration_s = len(mix) / ANALYSIS_SR
    total_samples = max(1, int(math.ceil(total_duration_s * OUTPUT_SR)))
    total_sum = np.zeros(total_samples, dtype=np.float64)
    total_weights = np.zeros(total_samples, dtype=np.float64)

    accepted_count = 0
    rejected_count = 0

    with tempfile.TemporaryDirectory(
        prefix=f"melody_{src.stem}_"
    ) as tmp:
        tmp_dir = Path(tmp)

        sources: dict[str, np.ndarray] = {}

        if separator is None:
            if not args.no_hpss_mix:
                try:
                    sources["mix_harmonic"] = hpss_harmonic(
                        mix,
                        margin=args.hpss_margin,
                    )
                except Exception as exc:
                    print(
                        f"  [WARN] HPSS(mix) failed: {exc}",
                        file=sys.stderr,
                    )

            # With --no-separation, retain the raw mix as a last resort.
            sources["mix"] = mix

        else:
            stems = separator.separate(src)

            # Drums and Bass are intentionally ignored.
            # They are never sent to Basic Pitch.
            if "drums" in stems:
                print("  [STEM] drums: discarded")
            if "bass" in stems:
                print("  [STEM] bass: discarded")

            # Vocals are usually already close to monophonic and should not
            # be HPSS-filtered by default, because consonants/attacks can be
            # damaged without improving the F0 line.
            if "vocals" in stems:
                try:
                    sources["vocals"] = load_mono(stems["vocals"])
                except Exception as exc:
                    print(
                        f"  [WARN] failed to load vocals: {exc}",
                        file=sys.stderr,
                    )

            # The Other stem is the main instrumental-melody candidate.
            if "other" in stems:
                try:
                    other = load_mono(stems["other"])

                    if not args.no_hpss_other:
                        sources["other_harmonic"] = hpss_harmonic(
                            other,
                            margin=args.hpss_margin,
                        )

                    if not args.no_raw_other:
                        sources["other"] = other

                except Exception as exc:
                    print(
                        f"  [WARN] failed to prepare other stem: {exc}",
                        file=sys.stderr,
                    )

            # htdemucs_6s users can also expose guitar/piano independently.
            for melodic_stem in ("guitar", "piano"):
                if melodic_stem not in stems:
                    continue

                try:
                    sources[melodic_stem] = load_mono(
                        stems[melodic_stem]
                    )
                except Exception as exc:
                    print(
                        f"  [WARN] failed to load {melodic_stem}: {exc}",
                        file=sys.stderr,
                    )

            # If the user selected a 2-stem model, keep compatibility:
            # HPSS the instrumental residual before transcription.
            if "instrumental" in stems:
                try:
                    instrumental = load_mono(
                        stems["instrumental"]
                    )
                    sources["instrumental_harmonic"] = hpss_harmonic(
                        instrumental,
                        margin=args.hpss_margin,
                    )
                    if not args.no_raw_other:
                        sources["instrumental"] = instrumental
                except Exception as exc:
                    print(
                        f"  [WARN] failed to prepare instrumental stem: {exc}",
                        file=sys.stderr,
                    )

            # Harmonic full-mix fallback catches lead material that leaks out
            # of all isolated stems, but suppresses most percussive transients.
            if not args.no_hpss_mix:
                try:
                    sources["mix_harmonic"] = hpss_harmonic(
                        mix,
                        margin=args.hpss_margin,
                    )
                except Exception as exc:
                    print(
                        f"  [WARN] HPSS(mix) failed: {exc}",
                        file=sys.stderr,
                    )

            # Raw mix is opt-in because percussion can create many false notes.
            if args.raw_mix_fallback:
                sources["mix"] = mix

        if not sources:
            raise RuntimeError(
                "No melody candidate sources could be prepared."
            )

        print(
            "  [CANDIDATES] "
            + ", ".join(sorted(sources.keys()))
        )

        for window in windows:
            results: list[CandidateResult] = []

            for source_name, source_audio in sources.items():
                try:
                    result = analyze_candidate(
                        source_name=source_name,
                        source_audio=source_audio,
                        sr=ANALYSIS_SR,
                        window=window,
                        transcriber=transcriber,
                        fps=args.viterbi_fps,
                        bridge_gap_ms=args.bridge_gap_ms,
                        min_output_note_ms=args.min_output_note_ms,
                        retrigger_min_ms=args.retrigger_min_ms,
                        retrigger_pitch_tolerance=args.retrigger_pitch_tolerance,
                        raw_active_threshold=args.raw_active_threshold,
                        raw_retrigger_threshold=args.raw_retrigger_threshold,
                        waveform_onset_weight=args.waveform_onset_weight,
                        tmp_dir=tmp_dir,
                    )
                    results.append(result)
                except Exception as exc:
                    print(
                        f"  [WARN] segment "
                        f"{window['segment_index']:04d} "
                        f"{source_name}: {exc}",
                        file=sys.stderr,
                    )

            if not results:
                rejected_count += 1
                rejected_writer.writerow(
                    {
                        "source_file": str(src.relative_to(input_root)),
                        "segment_index": window["segment_index"],
                        "start_bar": window["start_bar"],
                        "end_bar": window["end_bar"],
                        "start_s": f"{window['center_start']:.6f}",
                        "end_s": f"{window['center_end']:.6f}",
                        "reason": "no_candidate",
                        "best_source": "",
                        "score": "0",
                        "ranking_score": "0",
                        "source_bias": "0",
                        "voiced_ratio": "0",
                        "mean_confidence": "0",
                        "note_count": "0",
                    }
                )
                continue

            best = max(results, key=lambda r: r.score)
            ok, reason = accepted(best, args)

            segment_name = (
                f"{src.stem}"
                f"_b{window['start_bar']:04d}"
                f"-{window['end_bar']:04d}"
            )

            if not ok:
                rejected_count += 1
                rejected_writer.writerow(
                    {
                        "source_file": str(src.relative_to(input_root)),
                        "segment_index": window["segment_index"],
                        "start_bar": window["start_bar"],
                        "end_bar": window["end_bar"],
                        "start_s": f"{window['center_start']:.6f}",
                        "end_s": f"{window['center_end']:.6f}",
                        "reason": reason,
                        "best_source": best.source_name,
                        "score": f"{best.raw_score:.6f}",
                        "ranking_score": f"{best.score:.6f}",
                        "source_bias": f"{best.source_bias:.6f}",
                        "voiced_ratio": f"{best.voiced_ratio:.6f}",
                        "mean_confidence": f"{best.mean_confidence:.6f}",
                        "note_count": best.note_count,
                    }
                )
                continue

            duration_s = (
                window["center_end"]
                - window["center_start"]
            )

            melody = synthesize_notes(
                best.notes,
                duration_s=duration_s,
                sr=OUTPUT_SR,
            )

            wav_path = song_out / f"{segment_name}.wav"

            sf.write(
                wav_path,
                melody,
                OUTPUT_SR,
                subtype="PCM_16",
            )

            if not args.no_total:
                add_segment_to_timeline(
                    mix_sum=total_sum,
                    weight_sum=total_weights,
                    segment=melody,
                    start_s=window["center_start"],
                    sr=OUTPUT_SR,
                )

            if args.keep_events:
                write_events_json(
                    wav_path.with_suffix(".json"),
                    source_file=str(
                        src.relative_to(input_root)
                    ),
                    source_kind=best.source_name,
                    window=window,
                    result=best,
                )

            accepted_count += 1

            metadata_writer.writerow(
                {
                    "file": str(
                        wav_path.relative_to(output_root)
                    ),
                    "source_file": str(
                        src.relative_to(input_root)
                    ),
                    "segment_index": window["segment_index"],
                    "start_bar": window["start_bar"],
                    "end_bar": window["end_bar"],
                    "start_s": f"{window['center_start']:.6f}",
                    "end_s": f"{window['center_end']:.6f}",
                    "duration_s": f"{duration_s:.6f}",
                    "tempo_bpm": f"{tempo:.3f}",
                    "bar_phase": phase,
                    "source_kind": best.source_name,
                    "score": f"{best.raw_score:.6f}",
                    "ranking_score": f"{best.score:.6f}",
                    "source_bias": f"{best.source_bias:.6f}",
                    "voiced_ratio": f"{best.voiced_ratio:.6f}",
                    "mean_confidence": f"{best.mean_confidence:.6f}",
                    "large_jump_ratio": f"{best.large_jump_ratio:.6f}",
                    "octave_jump_ratio": f"{best.octave_jump_ratio:.6f}",
                    "note_count": best.note_count,
                }
            )

            print(
                f"  [OK] {segment_name} "
                f"| {best.source_name} "
                f"| score={best.raw_score:.3f} "
                f"| rank={best.score:.3f} "
                f"| notes={best.note_count}"
            )

    if not args.no_total:
        total_audio = finalize_total_timeline(
            total_sum,
            total_weights,
        )

        total_path = total_out_dir / f"{src.stem}_total.wav"

        sf.write(
            total_path,
            total_audio,
            OUTPUT_SR,
            subtype="PCM_16",
        )

        covered_samples = int(np.count_nonzero(total_weights > 1e-10))
        coverage = covered_samples / max(1, len(total_weights))

        print(
            f"  [TOTAL] {total_path.relative_to(output_root)} "
            f"| coverage={coverage:.1%}"
        )

    return accepted_count, rejected_count


def main() -> int:
    args = parse_args()

    input_root = args.input_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    if not input_root.is_dir():
        print(
            f"Input directory not found: {input_root}",
            file=sys.stderr,
        )
        return 2

    if args.beats_per_bar < 1:
        raise SystemExit("--beats-per-bar must be >= 1")
    if args.segment_bars < 1:
        raise SystemExit("--segment-bars must be >= 1")
    if args.stride_bars < 1:
        raise SystemExit("--stride-bars must be >= 1")
    if args.context_bars < 0:
        raise SystemExit("--context-bars must be >= 0")

    if args.hpss_margin < 1.0:
        raise SystemExit("--hpss-margin must be >= 1.0")
    if args.retrigger_min_ms < 0:
        raise SystemExit("--retrigger-min-ms must be >= 0")
    if args.retrigger_pitch_tolerance < 0:
        raise SystemExit("--retrigger-pitch-tolerance must be >= 0")
    if not 0.0 <= args.raw_active_threshold <= 1.0:
        raise SystemExit("--raw-active-threshold must be 0..1")
    if not 0.0 <= args.raw_retrigger_threshold <= 1.0:
        raise SystemExit("--raw-retrigger-threshold must be 0..1")
    if not 0.0 <= args.waveform_onset_weight <= 0.40:
        raise SystemExit("--waveform-onset-weight must be 0..0.40")

    files = find_mp3s(
        input_root,
        recursive=args.recursive,
    )

    if not files:
        print(
            f"No MP3 files found: {input_root}",
            file=sys.stderr,
        )
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "wav").mkdir(
        parents=True,
        exist_ok=True,
    )
    if not args.no_total:
        (output_root / "total").mkdir(
            parents=True,
            exist_ok=True,
        )

    separator = None

    if not args.no_separation:
        print("Loading source-separation model...")
        separator = SeparatorWrapper(
            output_dir=output_root / "_stems",
            model_filename=args.separator_model,
        )

    print("Loading Basic Pitch model...")
    transcriber = BasicPitchTranscriber(
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        minimum_note_ms=args.minimum_note_ms,
        min_midi=args.min_midi,
        max_midi=args.max_midi,
    )

    metadata_path = output_root / "metadata.csv"
    rejected_path = output_root / "rejected.csv"

    metadata_fields = [
        "file",
        "source_file",
        "segment_index",
        "start_bar",
        "end_bar",
        "start_s",
        "end_s",
        "duration_s",
        "tempo_bpm",
        "bar_phase",
        "source_kind",
        "score",
        "ranking_score",
        "source_bias",
        "voiced_ratio",
        "mean_confidence",
        "large_jump_ratio",
        "octave_jump_ratio",
        "note_count",
    ]

    rejected_fields = [
        "source_file",
        "segment_index",
        "start_bar",
        "end_bar",
        "start_s",
        "end_s",
        "reason",
        "best_source",
        "score",
        "ranking_score",
        "source_bias",
        "voiced_ratio",
        "mean_confidence",
        "note_count",
    ]

    total_ok = 0
    total_rejected = 0
    failed_songs = 0

    with (
        metadata_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as mf,
        rejected_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as rf,
    ):
        metadata_writer = csv.DictWriter(
            mf,
            fieldnames=metadata_fields,
        )
        metadata_writer.writeheader()

        rejected_writer = csv.DictWriter(
            rf,
            fieldnames=rejected_fields,
        )
        rejected_writer.writeheader()

        for index, src in enumerate(files, start=1):
            print(
                f"\n[{index}/{len(files)}] "
                f"{src.relative_to(input_root)}"
            )

            try:
                ok, rejected = process_song(
                    src=src,
                    input_root=input_root,
                    output_root=output_root,
                    separator=separator,
                    transcriber=transcriber,
                    args=args,
                    metadata_writer=metadata_writer,
                    rejected_writer=rejected_writer,
                )

                total_ok += ok
                total_rejected += rejected

            except Exception as exc:
                failed_songs += 1
                print(
                    f"[FAIL] {src}: {exc}",
                    file=sys.stderr,
                )

    print()
    print(
        "Done. "
        f"songs={len(files)}, "
        f"failed_songs={failed_songs}, "
        f"accepted_segments={total_ok}, "
        f"rejected_segments={total_rejected}"
    )
    print(f"Metadata : {metadata_path}")
    print(f"Rejected : {rejected_path}")
    if not args.no_total:
        print(f"Total WAV: {output_root / 'total'}")
    if not args.no_separation:
        print(f"Stems    : {output_root / '_stems'}")

    return 1 if failed_songs == len(files) else 0


if __name__ == "__main__":
    raise SystemExit(main())