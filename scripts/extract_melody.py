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
8. Use HPSS/harmonic audio for pitch tracking and the corresponding RAW stem
   for pitch-local attack detection.
9. Fuse neural onset, frame-rise onset, and pitch-local CQT attack evidence.
10. Decode same-pitch re-attacks with a dip-aware duration dynamic program.
11. Merge short low-confidence pitch chatter after decoding.
12. Extract MAIN and BASS roles independently.
12. Keep only high-confidence segments for each role.
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
- Reassembled per-song outputs are written separately under
  <output>/total/main/ and <output>/total/bass/.
- Accepted training segments are likewise separated into
  <output>/wav/main/ and <output>/wav/bass/.
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


@dataclass
class AnalysisSource:
    name: str
    pitch_audio: np.ndarray
    attack_audio: np.ndarray
    role: str
    min_midi: int
    max_midi: int


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
    p.add_argument("--min-output-note-ms", type=float, default=80.0)
    p.add_argument("--bridge-gap-ms", type=float, default=70.0)
    p.add_argument(
        "--retrigger-min-ms",
        type=float,
        default=90.0,
        help=(
            "Minimum spacing between same-pitch re-attack boundaries. "
            "Default: 90 ms."
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
        default=0.38,
        help=(
            "Threshold for fused raw onset evidence when splitting repeated "
            "same-pitch notes. Default: 0.38."
        ),
    )
    p.add_argument(
        "--attack-onset-weight",
        "--waveform-onset-weight",
        dest="attack_onset_weight",
        type=float,
        default=0.14,
        help=(
            "Weight of pitch-local CQT attack evidence from the RAW stem in "
            "the fused retrigger score. Default: 0.14."
        ),
    )
    p.add_argument(
        "--tempo-retrigger-fraction",
        type=float,
        default=0.125,
        help=(
            "Tempo-aware minimum same-pitch retrigger spacing as a fraction "
            "of one beat. Combined with --retrigger-min-ms. Default: 0.125."
        ),
    )
    p.add_argument(
        "--retrigger-split-penalty",
        type=float,
        default=0.28,
        help=(
            "Dynamic-program penalty for inserting a same-pitch boundary. "
            "Higher values suppress over-segmentation. Default: 0.28."
        ),
    )
    p.add_argument(
        "--retrigger-dip-window-ms",
        type=float,
        default=70.0,
        help=(
            "Look-back window for sustain-dip validation before a repeated "
            "note attack. Default: 70 ms."
        ),
    )
    p.add_argument(
        "--retrigger-min-dip",
        type=float,
        default=0.10,
        help=(
            "Minimum relative pre-attack salience dip expected for ambiguous "
            "same-pitch retriggers. Strong neural attacks may bypass this. "
            "Default: 0.10."
        ),
    )
    p.add_argument(
        "--retrigger-strong-margin",
        type=float,
        default=0.22,
        help=(
            "Onset evidence above threshold+margin may split without a clear "
            "pre-attack dip. Default: 0.22."
        ),
    )
    p.add_argument(
        "--cleanup-fragment-ms",
        type=float,
        default=95.0,
        help=(
            "Maximum duration of a low-confidence pitch fragment eligible for "
            "post-decode cleanup. Default: 95 ms."
        ),
    )
    p.add_argument(
        "--cleanup-confidence",
        type=float,
        default=0.42,
        help=(
            "Confidence ceiling for micro-fragment cleanup. Default: 0.42."
        ),
    )
    p.add_argument(
        "--cleanup-max-semitones",
        type=int,
        default=2,
        help=(
            "Maximum pitch distance of a short middle fragment from matching "
            "neighbors for cleanup. Default: 2 semitones."
        ),
    )

    p.add_argument(
        "--decoder-passes",
        type=int,
        choices=[1, 2],
        default=2,
        help=(
            "Analyze each candidate with one context or with a two-context "
            "ensemble and keep the higher-quality decode. Default: 2."
        ),
    )
    p.add_argument(
        "--decoder-extra-context-ms",
        type=float,
        default=350.0,
        help=(
            "Extra context on each side for decoder pass 2. Default: 350 ms."
        ),
    )
    p.add_argument(
        "--bass-min-midi",
        type=int,
        default=28,
        help="Lowest MIDI pitch for the bass role. Default: 28 (E1).",
    )
    p.add_argument(
        "--bass-max-midi",
        type=int,
        default=67,
        help="Highest MIDI pitch for the bass role. Default: 67 (G4).",
    )
    p.add_argument(
        "--bass-min-output-note-ms",
        type=float,
        default=110.0,
        help="Minimum decoded BASS note duration. Default: 110 ms.",
    )
    p.add_argument(
        "--bass-raw-retrigger-threshold",
        type=float,
        default=0.42,
        help="BASS repeated-note onset threshold. Default: 0.42.",
    )
    p.add_argument(
        "--bass-attack-onset-weight",
        type=float,
        default=0.11,
        help="BASS RAW-stem pitch-local attack weight. Default: 0.11.",
    )
    p.add_argument(
        "--bass-retrigger-split-penalty",
        type=float,
        default=0.32,
        help="BASS same-pitch split penalty. Default: 0.32.",
    )
    p.add_argument(
        "--bass-retrigger-min-ms",
        type=float,
        default=110.0,
        help="Minimum BASS same-pitch retrigger spacing. Default: 110 ms.",
    )
    p.add_argument("--bass-min-score", type=float, default=0.50)
    p.add_argument("--bass-min-voiced-ratio", type=float, default=0.15)
    p.add_argument("--bass-min-mean-confidence", type=float, default=0.30)
    p.add_argument("--bass-min-notes", type=int, default=2)

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


def pitch_local_attack_grid(
    audio: np.ndarray,
    sr: int,
    n_frames: int,
    min_midi: int,
    max_midi: int,
) -> np.ndarray:
    """
    Pitch-local spectral-flux attack evidence from the RAW stem.

    Unlike a global onset envelope, this follows the fundamental and low
    harmonics of each candidate MIDI pitch. A drum transient therefore does
    not automatically become a same-pitch retrigger unless energy also rises
    around the currently tracked harmonic series.
    """
    out = np.zeros((128, n_frames), dtype=np.float32)

    if n_frames <= 0 or len(audio) < 512:
        return out

    cqt_min_midi = max(21, min_midi)
    cqt_max_midi = min(108, max_midi + 24)
    if cqt_max_midi <= cqt_min_midi:
        return out

    n_bins = cqt_max_midi - cqt_min_midi + 1
    try:
        cqt = librosa.cqt(
            y=np.asarray(audio, dtype=np.float32),
            sr=sr,
            hop_length=256,
            fmin=midi_to_hz(cqt_min_midi),
            n_bins=n_bins,
            bins_per_octave=12,
        )
    except Exception:
        return out

    mag = np.abs(cqt).astype(np.float32)
    log_mag = np.log1p(12.0 * mag)

    flux = np.zeros_like(log_mag)
    if log_mag.shape[1] > 1:
        flux[:, 1:] = np.maximum(
            0.0,
            log_mag[:, 1:] - log_mag[:, :-1],
        )

    source_x = np.linspace(0.0, 1.0, flux.shape[1], dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)

    # Fundamental, 2nd, 3rd and 4th harmonics in semitone offsets.
    harmonic_offsets = (0, 12, 19, 24)
    harmonic_weights = (0.55, 0.24, 0.13, 0.08)

    for midi in range(max(0, min_midi), min(127, max_midi) + 1):
        curve = np.zeros(flux.shape[1], dtype=np.float32)
        weight_sum = 0.0

        for offset, weight in zip(harmonic_offsets, harmonic_weights):
            hm = midi + offset
            idx = hm - cqt_min_midi
            if 0 <= idx < flux.shape[0]:
                curve += float(weight) * flux[idx]
                weight_sum += float(weight)

        if weight_sum <= 0:
            continue

        curve /= weight_sum
        positive = curve[curve > 0]
        scale = (
            float(np.percentile(positive, 95))
            if len(positive)
            else float(np.max(curve))
        )

        if scale > 1e-12:
            curve = np.clip(curve / scale, 0.0, 1.0)
        else:
            continue

        if len(curve) == 1:
            out[midi, :] = float(curve[0])
        else:
            out[midi, :] = np.interp(
                target_x,
                source_x,
                curve,
            ).astype(np.float32)

    return out


def raw_basic_pitch_decoder_inputs(
    prediction: BasicPitchPrediction,
    pitch_context_audio: np.ndarray,
    attack_context_audio: np.ndarray,
    sr: int,
    min_midi: int,
    max_midi: int,
    attack_weight: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Build pitch salience and fused re-attack evidence.

    Basic Pitch runs on the pitch-oriented source (normally HPSS harmonic).
    Pitch-local CQT flux is measured independently on the corresponding RAW
    stem, preserving attack transients that HPSS may weaken.
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

    local_attack = pitch_local_attack_grid(
        audio=attack_context_audio,
        sr=sr,
        n_frames=t_count,
        min_midi=min_midi,
        max_midi=max_midi,
    )

    salience = np.zeros((128, t_count), dtype=np.float32)
    onset128 = np.zeros((128, t_count), dtype=np.float32)
    rise128 = np.zeros((128, t_count), dtype=np.float32)

    for idx in range(BASIC_PITCH_NOTE_BINS):
        midi = BASIC_PITCH_MIDI_OFFSET + idx
        if midi < min_midi or midi > max_midi:
            continue
        salience[midi] = salience_88[:, idx]
        onset128[midi] = onsets[:, idx]
        rise128[midi] = frame_rise[:, idx]

    w_attack = float(np.clip(attack_weight, 0.0, 0.40))
    w_onset = 0.55
    w_rise = max(0.0, 1.0 - w_onset - w_attack)

    # Pitch-local attack can rescue a missed neural onset, but only while the
    # corresponding note is actually salient.
    attack_gate = np.clip(salience / 0.18, 0.0, 1.0)

    fused_onset = np.clip(
        w_onset * onset128
        + w_rise * rise128
        + w_attack * local_attack * attack_gate,
        0.0,
        1.0,
    )

    duration_s = len(pitch_context_audio) / max(sr, 1)
    fps = float(t_count) / duration_s if duration_s > 1e-9 else 0.0

    if fps <= 0:
        return None

    return salience, fused_onset, fps


def duration_aware_retrigger_boundaries(
    path: np.ndarray,
    onset_score: np.ndarray,
    pitch_salience: np.ndarray,
    fps: float,
    threshold: float,
    absolute_min_ms: float,
    min_note_ms: float,
    pitch_tolerance: int,
    tempo_bpm: float,
    tempo_fraction: float,
    split_penalty: float,
    dip_window_ms: float,
    min_dip: float,
    strong_margin: float,
) -> np.ndarray:
    """
    Select same-pitch re-attacks with a dip-aware semi-Markov decoder.

    Weak/ambiguous attack peaks are accepted only when the sustained pitch
    salience shows a preceding dip, which is typical of a real envelope
    re-attack. Very strong neural onset evidence can bypass the dip condition,
    preserving genuinely hard repeated notes that remain continuously voiced.
    """
    n = len(path)
    boundaries = np.zeros(n, dtype=bool)

    if n < 3:
        return boundaries

    beat_ms = (
        60000.0 / tempo_bpm
        if tempo_bpm > 1e-6
        else 500.0
    )
    tempo_min_ms = max(0.0, beat_ms * max(0.0, tempo_fraction))
    dynamic_min_ms = max(
        float(absolute_min_ms),
        float(min_note_ms),
        tempo_min_ms,
    )
    min_frames = max(
        1,
        int(round(dynamic_min_ms / 1000.0 * fps)),
    )
    dip_frames = max(
        2,
        int(round(max(0.0, dip_window_ms) / 1000.0 * fps)),
    )

    i = 0
    while i < n:
        state = int(path[i])

        if state == REST_STATE:
            i += 1
            continue

        j = i + 1
        while j < n and int(path[j]) == state:
            j += 1

        run_len = j - i
        if run_len < 2 * min_frames + 1:
            i = j
            continue

        lo = max(0, state - pitch_tolerance)
        hi = min(127, state + pitch_tolerance)

        onset_curve = np.max(
            onset_score[lo:hi + 1, i:j],
            axis=0,
        )
        salience_curve = np.max(
            pitch_salience[lo:hi + 1, i:j],
            axis=0,
        )

        candidates: list[tuple[int, float]] = []

        for rel in range(1, run_len - 1):
            score = float(onset_curve[rel])

            if score < threshold:
                continue
            if (
                score < float(onset_curve[rel - 1])
                or score < float(onset_curve[rel + 1])
            ):
                continue

            pos = i + rel
            if pos - i < min_frames or j - pos < min_frames:
                continue

            # Compare the local pre-attack minimum against the surrounding
            # sustain level. A meaningful drop supports a true re-articulation.
            pre0 = max(0, rel - dip_frames)
            pre1 = rel
            post1 = min(run_len, rel + max(2, dip_frames // 2))

            pre = salience_curve[pre0:pre1]
            post = salience_curve[rel:post1]

            if len(pre):
                pre_floor = float(np.percentile(pre, 20))
                pre_level = float(np.percentile(pre, 75))
            else:
                pre_floor = float(salience_curve[rel])
                pre_level = pre_floor

            post_level = (
                float(np.percentile(post, 75))
                if len(post)
                else float(salience_curve[rel])
            )

            reference = max(
                1e-6,
                pre_level,
                post_level,
                float(salience_curve[rel]),
            )
            dip = float(
                np.clip(
                    (reference - pre_floor) / reference,
                    0.0,
                    1.0,
                )
            )

            strong_attack = score >= (
                float(threshold) + float(strong_margin)
            )

            # Ambiguous peaks without an envelope/salience dip are the main
            # source of chattering. Strong neural attacks remain admissible.
            if not strong_attack and dip < min_dip:
                continue

            normalized = (
                (score - threshold)
                / max(1e-6, 1.0 - threshold)
            )

            # Dip provides a bounded bonus rather than acting as an absolute
            # requirement for strong attacks.
            dip_bonus = 0.24 * np.clip(
                (dip - min_dip) / max(1e-6, 1.0 - min_dip),
                0.0,
                1.0,
            )
            strong_bonus = 0.10 if strong_attack else 0.0

            utility = (
                normalized
                + float(dip_bonus)
                + strong_bonus
                - float(split_penalty)
            )

            if utility > -0.15:
                candidates.append((pos, utility))

        if not candidates:
            i = j
            continue

        # Weighted interval scheduling across valid split points.
        m = len(candidates)
        best = np.zeros(m + 1, dtype=np.float64)
        prev_choice = np.full(m + 1, -1, dtype=np.int32)
        took = np.zeros(m + 1, dtype=bool)
        positions = [p for p, _ in candidates]

        for k in range(1, m + 1):
            pos, utility = candidates[k - 1]

            prev_k = 0
            for q in range(k - 1, 0, -1):
                if pos - positions[q - 1] >= min_frames:
                    prev_k = q
                    break

            take_score = best[prev_k] + utility
            skip_score = best[k - 1]

            if take_score > skip_score and utility > 0:
                best[k] = take_score
                prev_choice[k] = prev_k
                took[k] = True
            else:
                best[k] = skip_score
                prev_choice[k] = k - 1

        k = m
        chosen: list[int] = []

        while k > 0:
            if took[k]:
                chosen.append(candidates[k - 1][0])
                k = int(prev_choice[k])
            else:
                k -= 1

        for pos in reversed(chosen):
            boundaries[pos] = True

        i = j

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



def cleanup_micro_fragments(
    notes: list[Note],
    max_fragment_ms: float,
    confidence_ceiling: float,
    max_semitones: int,
) -> list[Note]:
    """
    Remove short low-confidence pitch chatter after decoding.

    Conservative rules:
    - A-B-A: if B is short, low-confidence and close in pitch, absorb it into
      the matching A neighbors.
    - Adjacent same-pitch fragments are merged only when one is both short and
      low-confidence, avoiding destruction of deliberate repeated notes.
    """
    if len(notes) < 2:
        return notes

    work = list(notes)
    changed = True

    while changed and len(work) >= 2:
        changed = False

        # A-B-A transient artifact.
        i = 1
        while i + 1 < len(work):
            left = work[i - 1]
            mid = work[i]
            right = work[i + 1]

            mid_ms = (mid.end - mid.start) * 1000.0
            close_pitch = (
                abs(mid.pitch - left.pitch) <= max_semitones
                and abs(mid.pitch - right.pitch) <= max_semitones
            )
            neighbors_match = left.pitch == right.pitch
            low_conf = mid.confidence <= confidence_ceiling

            if (
                mid_ms <= max_fragment_ms
                and low_conf
                and close_pitch
                and neighbors_match
            ):
                merged_conf = float(
                    np.average(
                        [left.confidence, right.confidence],
                        weights=[
                            max(1e-6, left.end - left.start),
                            max(1e-6, right.end - right.start),
                        ],
                    )
                )
                work[i - 1:i + 2] = [
                    Note(
                        start=left.start,
                        end=right.end,
                        pitch=left.pitch,
                        confidence=merged_conf,
                    )
                ]
                changed = True
                break

            i += 1

        if changed:
            continue

        # Same-pitch accidental micro-split.
        i = 0
        while i + 1 < len(work):
            a = work[i]
            b = work[i + 1]

            if a.pitch != b.pitch:
                i += 1
                continue

            a_ms = (a.end - a.start) * 1000.0
            b_ms = (b.end - b.start) * 1000.0

            a_noisy = (
                a_ms <= max_fragment_ms
                and a.confidence <= confidence_ceiling
            )
            b_noisy = (
                b_ms <= max_fragment_ms
                and b.confidence <= confidence_ceiling
            )

            if not (a_noisy or b_noisy):
                i += 1
                continue

            dur_a = max(1e-6, a.end - a.start)
            dur_b = max(1e-6, b.end - b.start)
            merged_conf = float(
                (a.confidence * dur_a + b.confidence * dur_b)
                / (dur_a + dur_b)
            )

            work[i:i + 2] = [
                Note(
                    start=a.start,
                    end=b.end,
                    pitch=a.pitch,
                    confidence=merged_conf,
                )
            ]
            changed = True
            break

    return work


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


def _decode_candidate_pass(
    source_name: str,
    pitch_audio: np.ndarray,
    attack_audio: np.ndarray,
    sr: int,
    analysis_start: float,
    analysis_end: float,
    center_start: float,
    center_end: float,
    transcriber: BasicPitchTranscriber,
    fallback_fps: float,
    bridge_gap_ms: float,
    min_output_note_ms: float,
    retrigger_min_ms: float,
    retrigger_pitch_tolerance: int,
    raw_active_threshold: float,
    raw_retrigger_threshold: float,
    attack_onset_weight: float,
    tempo_bpm: float,
    tempo_retrigger_fraction: float,
    retrigger_split_penalty: float,
    retrigger_dip_window_ms: float,
    retrigger_min_dip: float,
    retrigger_strong_margin: float,
    cleanup_fragment_ms: float,
    cleanup_confidence: float,
    cleanup_max_semitones: int,
    decode_min_midi: int,
    decode_max_midi: int,
    tmp_dir: Path,
    pass_index: int,
    segment_index: int,
) -> list[Note]:
    pitch_context = crop_audio(
        pitch_audio,
        sr,
        analysis_start,
        analysis_end,
    )
    attack_context = crop_audio(
        attack_audio,
        sr,
        analysis_start,
        analysis_end,
    )

    if len(pitch_context) < int(0.25 * sr):
        return []

    context_wav = (
        tmp_dir
        / f"segment_{segment_index:04d}_{source_name}_p{pass_index}.wav"
    )

    sf.write(
        context_wav,
        pitch_context,
        sr,
        subtype="PCM_16",
    )

    prediction = transcriber.predict(context_wav)
    duration_s = len(pitch_context) / sr

    raw_inputs = raw_basic_pitch_decoder_inputs(
        prediction=prediction,
        pitch_context_audio=pitch_context,
        attack_context_audio=attack_context,
        sr=sr,
        min_midi=decode_min_midi,
        max_midi=decode_max_midi,
        attack_weight=attack_onset_weight,
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

        retrigger_boundaries = duration_aware_retrigger_boundaries(
            path=path,
            onset_score=fused_onset,
            pitch_salience=grid,
            fps=decoder_fps,
            threshold=raw_retrigger_threshold,
            absolute_min_ms=retrigger_min_ms,
            min_note_ms=min_output_note_ms,
            pitch_tolerance=retrigger_pitch_tolerance,
            tempo_bpm=tempo_bpm,
            tempo_fraction=tempo_retrigger_fraction,
            split_penalty=retrigger_split_penalty,
            dip_window_ms=retrigger_dip_window_ms,
            min_dip=retrigger_min_dip,
            strong_margin=retrigger_strong_margin,
        )
        fps = decoder_fps

    else:
        # Compatibility fallback.
        events = [
            e
            for e in prediction.note_events
            if len(e) >= 3
            and decode_min_midi <= int(round(float(e[2]))) <= decode_max_midi
        ]

        grid = note_events_to_confidence_grid(
            events,
            duration_s=duration_s,
            fps=fallback_fps,
        )
        onset_grid = note_events_to_onset_grid(
            events,
            duration_s=duration_s,
            fps=fallback_fps,
        )
        path, frame_conf = viterbi_monophonic_path(grid)

        bridge_frames = max(
            0,
            int(round(bridge_gap_ms / 1000.0 * fallback_fps)),
        )
        path = bridge_short_gaps(
            path,
            max_gap_frames=bridge_frames,
        )
        retrigger_boundaries = retrigger_boundaries_from_onsets(
            path=path,
            onset_grid=onset_grid,
            fps=fallback_fps,
            min_spacing_ms=retrigger_min_ms,
            pitch_tolerance=retrigger_pitch_tolerance,
        )
        fps = fallback_fps

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

    notes = cleanup_micro_fragments(
        notes,
        max_fragment_ms=cleanup_fragment_ms,
        confidence_ceiling=cleanup_confidence,
        max_semitones=cleanup_max_semitones,
    )

    center_offset = center_start - analysis_start
    center_duration = center_end - center_start

    return crop_notes_to_center(
        notes,
        center_offset_s=center_offset,
        center_duration_s=center_duration,
    )


def analyze_candidate(
    source_name: str,
    source_audio: np.ndarray,
    attack_audio: np.ndarray,
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
    attack_onset_weight: float,
    tempo_bpm: float,
    tempo_retrigger_fraction: float,
    retrigger_split_penalty: float,
    retrigger_dip_window_ms: float,
    retrigger_min_dip: float,
    retrigger_strong_margin: float,
    cleanup_fragment_ms: float,
    cleanup_confidence: float,
    cleanup_max_semitones: int,
    decode_min_midi: int,
    decode_max_midi: int,
    decoder_passes: int,
    decoder_extra_context_ms: float,
    tmp_dir: Path,
) -> CandidateResult:
    total_duration = len(source_audio) / sr

    pass_ranges = [
        (
            window["context_start"],
            window["context_end"],
        )
    ]

    if decoder_passes >= 2:
        extra = max(0.0, decoder_extra_context_ms / 1000.0)
        pass_ranges.append(
            (
                max(0.0, window["context_start"] - extra),
                min(total_duration, window["context_end"] + extra),
            )
        )

    decoded: list[tuple[list[Note], float]] = []

    for pass_index, (analysis_start, analysis_end) in enumerate(
        pass_ranges,
        start=1,
    ):
        notes = _decode_candidate_pass(
            source_name=source_name,
            pitch_audio=source_audio,
            attack_audio=attack_audio,
            sr=sr,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            center_start=window["center_start"],
            center_end=window["center_end"],
            transcriber=transcriber,
            fallback_fps=fps,
            bridge_gap_ms=bridge_gap_ms,
            min_output_note_ms=min_output_note_ms,
            retrigger_min_ms=retrigger_min_ms,
            retrigger_pitch_tolerance=retrigger_pitch_tolerance,
            raw_active_threshold=raw_active_threshold,
            raw_retrigger_threshold=raw_retrigger_threshold,
            attack_onset_weight=attack_onset_weight,
            tempo_bpm=tempo_bpm,
            tempo_retrigger_fraction=tempo_retrigger_fraction,
            retrigger_split_penalty=retrigger_split_penalty,
            retrigger_dip_window_ms=retrigger_dip_window_ms,
            retrigger_min_dip=retrigger_min_dip,
            retrigger_strong_margin=retrigger_strong_margin,
            cleanup_fragment_ms=cleanup_fragment_ms,
            cleanup_confidence=cleanup_confidence,
            cleanup_max_semitones=cleanup_max_semitones,
            decode_min_midi=decode_min_midi,
            decode_max_midi=decode_max_midi,
            tmp_dir=tmp_dir,
            pass_index=pass_index,
            segment_index=window["segment_index"],
        )

        center_duration = (
            window["center_end"] - window["center_start"]
        )
        metrics = quality_metrics(
            notes,
            duration_s=center_duration,
        )
        decoded.append((notes, float(metrics[-1])))

    # Multi-context ensemble: keep the decode with the better acoustic/path
    # quality. This costs another Basic Pitch pass but is more robust when a
    # note attack lies close to a context boundary.
    notes = max(decoded, key=lambda item: item[1])[0] if decoded else []

    center_duration = (
        window["center_end"] - window["center_start"]
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



def accepted_for_role(
    result: CandidateResult,
    args: argparse.Namespace,
    role: str,
) -> tuple[bool, str]:
    if role == "main":
        return accepted(result, args)

    reasons = []
    if result.raw_score < args.bass_min_score:
        reasons.append(f"score<{args.bass_min_score:.2f}")
    if result.voiced_ratio < args.bass_min_voiced_ratio:
        reasons.append(f"voiced<{args.bass_min_voiced_ratio:.2f}")
    if result.mean_confidence < args.bass_min_mean_confidence:
        reasons.append(
            f"confidence<{args.bass_min_mean_confidence:.2f}"
        )
    if result.note_count < args.bass_min_notes:
        reasons.append(f"notes<{args.bass_min_notes}")

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
        "bass_harmonic": 0.045,
        "bass": 0.020,
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

    role_wav_dirs = {
        "main": output_root / "wav" / "main" / rel_parent,
        "bass": output_root / "wav" / "bass" / rel_parent,
    }
    role_total_dirs = {
        "main": output_root / "total" / "main" / rel_parent,
        "bass": output_root / "total" / "bass" / rel_parent,
    }

    for path in role_wav_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    if not args.no_total:
        for path in role_total_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    total_duration_s = len(mix) / ANALYSIS_SR
    total_samples = max(
        1,
        int(math.ceil(total_duration_s * OUTPUT_SR)),
    )

    role_sums = {
        "main": np.zeros(total_samples, dtype=np.float64),
        "bass": np.zeros(total_samples, dtype=np.float64),
    }
    role_weights = {
        "main": np.zeros(total_samples, dtype=np.float64),
        "bass": np.zeros(total_samples, dtype=np.float64),
    }

    accepted_count = 0
    rejected_count = 0

    with tempfile.TemporaryDirectory(
        prefix=f"melody_{src.stem}_"
    ) as tmp:
        tmp_dir = Path(tmp)

        main_sources: list[AnalysisSource] = []
        bass_sources: list[AnalysisSource] = []

        if separator is None:
            if not args.no_hpss_mix:
                try:
                    harmonic = hpss_harmonic(
                        mix,
                        margin=args.hpss_margin,
                    )
                    main_sources.append(
                        AnalysisSource(
                            name="mix_harmonic",
                            pitch_audio=harmonic,
                            attack_audio=mix,
                            role="main",
                            min_midi=args.min_midi,
                            max_midi=args.max_midi,
                        )
                    )
                except Exception as exc:
                    print(
                        f"  [WARN] HPSS(mix) failed: {exc}",
                        file=sys.stderr,
                    )

            main_sources.append(
                AnalysisSource(
                    name="mix",
                    pitch_audio=mix,
                    attack_audio=mix,
                    role="main",
                    min_midi=args.min_midi,
                    max_midi=args.max_midi,
                )
            )

        else:
            stems = separator.separate(src)

            if "drums" in stems:
                print("  [STEM] drums: preserved, not transcribed")

            if "vocals" in stems:
                try:
                    vocals = load_mono(stems["vocals"])
                    main_sources.append(
                        AnalysisSource(
                            name="vocals",
                            pitch_audio=vocals,
                            attack_audio=vocals,
                            role="main",
                            min_midi=args.min_midi,
                            max_midi=args.max_midi,
                        )
                    )
                except Exception as exc:
                    print(
                        f"  [WARN] failed to load vocals: {exc}",
                        file=sys.stderr,
                    )

            if "other" in stems:
                try:
                    other = load_mono(stems["other"])

                    if not args.no_hpss_other:
                        harmonic = hpss_harmonic(
                            other,
                            margin=args.hpss_margin,
                        )
                        # Critical pairing: harmonic for pitch, RAW Other for
                        # attack/retrigger detection.
                        main_sources.append(
                            AnalysisSource(
                                name="other_harmonic",
                                pitch_audio=harmonic,
                                attack_audio=other,
                                role="main",
                                min_midi=args.min_midi,
                                max_midi=args.max_midi,
                            )
                        )

                    if not args.no_raw_other:
                        main_sources.append(
                            AnalysisSource(
                                name="other",
                                pitch_audio=other,
                                attack_audio=other,
                                role="main",
                                min_midi=args.min_midi,
                                max_midi=args.max_midi,
                            )
                        )

                except Exception as exc:
                    print(
                        f"  [WARN] failed to prepare other stem: {exc}",
                        file=sys.stderr,
                    )

            # Dedicated BASS role. Use HPSS bass for pitch stability and raw
            # bass for attack timing, plus raw bass as a fallback candidate.
            if "bass" in stems:
                try:
                    bass = load_mono(stems["bass"])
                    bass_harmonic = hpss_harmonic(
                        bass,
                        margin=args.hpss_margin,
                    )

                    bass_sources.append(
                        AnalysisSource(
                            name="bass_harmonic",
                            pitch_audio=bass_harmonic,
                            attack_audio=bass,
                            role="bass",
                            min_midi=args.bass_min_midi,
                            max_midi=args.bass_max_midi,
                        )
                    )
                    bass_sources.append(
                        AnalysisSource(
                            name="bass",
                            pitch_audio=bass,
                            attack_audio=bass,
                            role="bass",
                            min_midi=args.bass_min_midi,
                            max_midi=args.bass_max_midi,
                        )
                    )
                except Exception as exc:
                    print(
                        f"  [WARN] failed to prepare bass stem: {exc}",
                        file=sys.stderr,
                    )
            else:
                print("  [BASS] no bass stem available")

            for melodic_stem in ("guitar", "piano"):
                if melodic_stem not in stems:
                    continue
                try:
                    audio = load_mono(stems[melodic_stem])
                    main_sources.append(
                        AnalysisSource(
                            name=melodic_stem,
                            pitch_audio=audio,
                            attack_audio=audio,
                            role="main",
                            min_midi=args.min_midi,
                            max_midi=args.max_midi,
                        )
                    )
                except Exception as exc:
                    print(
                        f"  [WARN] failed to load {melodic_stem}: {exc}",
                        file=sys.stderr,
                    )

            if "instrumental" in stems:
                try:
                    instrumental = load_mono(stems["instrumental"])
                    harmonic = hpss_harmonic(
                        instrumental,
                        margin=args.hpss_margin,
                    )
                    main_sources.append(
                        AnalysisSource(
                            name="instrumental_harmonic",
                            pitch_audio=harmonic,
                            attack_audio=instrumental,
                            role="main",
                            min_midi=args.min_midi,
                            max_midi=args.max_midi,
                        )
                    )
                    if not args.no_raw_other:
                        main_sources.append(
                            AnalysisSource(
                                name="instrumental",
                                pitch_audio=instrumental,
                                attack_audio=instrumental,
                                role="main",
                                min_midi=args.min_midi,
                                max_midi=args.max_midi,
                            )
                        )
                except Exception as exc:
                    print(
                        f"  [WARN] failed to prepare instrumental stem: {exc}",
                        file=sys.stderr,
                    )

            if not args.no_hpss_mix:
                try:
                    mix_harmonic = hpss_harmonic(
                        mix,
                        margin=args.hpss_margin,
                    )
                    main_sources.append(
                        AnalysisSource(
                            name="mix_harmonic",
                            pitch_audio=mix_harmonic,
                            attack_audio=mix,
                            role="main",
                            min_midi=args.min_midi,
                            max_midi=args.max_midi,
                        )
                    )
                except Exception as exc:
                    print(
                        f"  [WARN] HPSS(mix) failed: {exc}",
                        file=sys.stderr,
                    )

            if args.raw_mix_fallback:
                main_sources.append(
                    AnalysisSource(
                        name="mix",
                        pitch_audio=mix,
                        attack_audio=mix,
                        role="main",
                        min_midi=args.min_midi,
                        max_midi=args.max_midi,
                    )
                )

        if not main_sources:
            raise RuntimeError(
                "No MAIN melody candidate sources could be prepared."
            )

        print(
            "  [MAIN CANDIDATES] "
            + ", ".join(sorted(s.name for s in main_sources))
        )
        if bass_sources:
            print(
                "  [BASS CANDIDATES] "
                + ", ".join(sorted(s.name for s in bass_sources))
            )

        role_sources = {
            "main": main_sources,
            "bass": bass_sources,
        }

        for window in windows:
            for role in ("main", "bass"):
                candidates = role_sources[role]

                if not candidates:
                    continue

                results: list[CandidateResult] = []

                for source in candidates:
                    try:
                        result = analyze_candidate(
                            source_name=source.name,
                            source_audio=source.pitch_audio,
                            attack_audio=source.attack_audio,
                            sr=ANALYSIS_SR,
                            window=window,
                            transcriber=transcriber,
                            fps=args.viterbi_fps,
                            bridge_gap_ms=args.bridge_gap_ms,
                            min_output_note_ms=(
                                args.bass_min_output_note_ms
                                if role == "bass"
                                else args.min_output_note_ms
                            ),
                            retrigger_min_ms=(
                                args.bass_retrigger_min_ms
                                if role == "bass"
                                else args.retrigger_min_ms
                            ),
                            retrigger_pitch_tolerance=args.retrigger_pitch_tolerance,
                            raw_active_threshold=args.raw_active_threshold,
                            raw_retrigger_threshold=(
                                args.bass_raw_retrigger_threshold
                                if role == "bass"
                                else args.raw_retrigger_threshold
                            ),
                            attack_onset_weight=(
                                args.bass_attack_onset_weight
                                if role == "bass"
                                else args.attack_onset_weight
                            ),
                            tempo_bpm=tempo,
                            tempo_retrigger_fraction=args.tempo_retrigger_fraction,
                            retrigger_split_penalty=(
                                args.bass_retrigger_split_penalty
                                if role == "bass"
                                else args.retrigger_split_penalty
                            ),
                            retrigger_dip_window_ms=args.retrigger_dip_window_ms,
                            retrigger_min_dip=args.retrigger_min_dip,
                            retrigger_strong_margin=args.retrigger_strong_margin,
                            cleanup_fragment_ms=(
                                max(args.cleanup_fragment_ms, 120.0)
                                if role == "bass"
                                else args.cleanup_fragment_ms
                            ),
                            cleanup_confidence=args.cleanup_confidence,
                            cleanup_max_semitones=args.cleanup_max_semitones,
                            decode_min_midi=source.min_midi,
                            decode_max_midi=source.max_midi,
                            decoder_passes=args.decoder_passes,
                            decoder_extra_context_ms=args.decoder_extra_context_ms,
                            tmp_dir=tmp_dir,
                        )
                        results.append(result)
                    except Exception as exc:
                        print(
                            f"  [WARN] {role} segment "
                            f"{window['segment_index']:04d} "
                            f"{source.name}: {exc}",
                            file=sys.stderr,
                        )

                if not results:
                    rejected_count += 1
                    rejected_writer.writerow(
                        {
                            "role": role,
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
                ok, reason = accepted_for_role(
                    best,
                    args,
                    role,
                )

                segment_name = (
                    f"{src.stem}"
                    f"_b{window['start_bar']:04d}"
                    f"-{window['end_bar']:04d}"
                    f"_{role}"
                )

                if not ok:
                    rejected_count += 1
                    rejected_writer.writerow(
                        {
                            "role": role,
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
                    window["center_end"] - window["center_start"]
                )
                melody = synthesize_notes(
                    best.notes,
                    duration_s=duration_s,
                    sr=OUTPUT_SR,
                )

                wav_path = (
                    role_wav_dirs[role]
                    / f"{segment_name}.wav"
                )

                sf.write(
                    wav_path,
                    melody,
                    OUTPUT_SR,
                    subtype="PCM_16",
                )

                if not args.no_total:
                    add_segment_to_timeline(
                        mix_sum=role_sums[role],
                        weight_sum=role_weights[role],
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
                        "role": role,
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
                    f"  [OK:{role.upper()}] {segment_name} "
                    f"| {best.source_name} "
                    f"| score={best.raw_score:.3f} "
                    f"| rank={best.score:.3f} "
                    f"| notes={best.note_count}"
                )

    if not args.no_total:
        for role in ("main", "bass"):
            if role == "bass" and not bass_sources:
                continue

            total_audio = finalize_total_timeline(
                role_sums[role],
                role_weights[role],
            )

            total_path = (
                role_total_dirs[role]
                / f"{src.stem}_{role}_total.wav"
            )

            sf.write(
                total_path,
                total_audio,
                OUTPUT_SR,
                subtype="PCM_16",
            )

            covered_samples = int(
                np.count_nonzero(role_weights[role] > 1e-10)
            )
            coverage = (
                covered_samples
                / max(1, len(role_weights[role]))
            )

            print(
                f"  [TOTAL:{role.upper()}] "
                f"{total_path.relative_to(output_root)} "
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
    if not 0.0 <= args.attack_onset_weight <= 0.40:
        raise SystemExit("--attack-onset-weight must be 0..0.40")
    if args.tempo_retrigger_fraction < 0.0:
        raise SystemExit("--tempo-retrigger-fraction must be >= 0")
    if args.retrigger_split_penalty < 0.0:
        raise SystemExit("--retrigger-split-penalty must be >= 0")
    if args.retrigger_dip_window_ms < 0.0:
        raise SystemExit("--retrigger-dip-window-ms must be >= 0")
    if not 0.0 <= args.retrigger_min_dip <= 1.0:
        raise SystemExit("--retrigger-min-dip must be 0..1")
    if args.retrigger_strong_margin < 0.0:
        raise SystemExit("--retrigger-strong-margin must be >= 0")
    if args.cleanup_fragment_ms < 0.0:
        raise SystemExit("--cleanup-fragment-ms must be >= 0")
    if not 0.0 <= args.cleanup_confidence <= 1.0:
        raise SystemExit("--cleanup-confidence must be 0..1")
    if args.cleanup_max_semitones < 0:
        raise SystemExit("--cleanup-max-semitones must be >= 0")
    if args.bass_min_output_note_ms < 0.0:
        raise SystemExit("--bass-min-output-note-ms must be >= 0")
    if not 0.0 <= args.bass_raw_retrigger_threshold <= 1.0:
        raise SystemExit("--bass-raw-retrigger-threshold must be 0..1")
    if not 0.0 <= args.bass_attack_onset_weight <= 0.40:
        raise SystemExit("--bass-attack-onset-weight must be 0..0.40")
    if args.bass_retrigger_split_penalty < 0.0:
        raise SystemExit("--bass-retrigger-split-penalty must be >= 0")
    if args.bass_retrigger_min_ms < 0.0:
        raise SystemExit("--bass-retrigger-min-ms must be >= 0")
    if args.decoder_extra_context_ms < 0.0:
        raise SystemExit("--decoder-extra-context-ms must be >= 0")
    if not 0 <= args.bass_min_midi <= args.bass_max_midi <= 127:
        raise SystemExit("bass MIDI range must satisfy 0 <= min <= max <= 127")

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
    for role in ("main", "bass"):
        (output_root / "wav" / role).mkdir(
            parents=True,
            exist_ok=True,
        )
        if not args.no_total:
            (output_root / "total" / role).mkdir(
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
        min_midi=min(args.min_midi, args.bass_min_midi),
        max_midi=max(args.max_midi, args.bass_max_midi),
    )

    metadata_path = output_root / "metadata.csv"
    rejected_path = output_root / "rejected.csv"

    metadata_fields = [
        "role",
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
        "role",
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
        print(f"Total MAIN: {output_root / 'total' / 'main'}")
        print(f"Total BASS: {output_root / 'total' / 'bass'}")
    if not args.no_separation:
        print(f"Stems    : {output_root / '_stems'}")

    return 1 if failed_songs == len(files) else 0


if __name__ == "__main__":
    raise SystemExit(main())