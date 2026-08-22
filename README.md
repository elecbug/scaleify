# Scaleify v9

`v9` is a **monophonic melodic style-transfer prototype**. Feed it an isolated melody, solo instrument, vocal melody, or another predominantly monophonic file.

## Pipeline

```text
MP3/WAV/FLAC
  -> decode
  -> YIN / pYIN F0
  -> note events
  -> phrase segmentation
  -> higher-order Viterbi grammar
       - scale constraint
       - degree preference
       - interval preference
       - direction-aware transitions
       - trigrams
       - preferred 4/5-note phrases
       - phrase cadence
       - optional phrase-level modulation
  -> rhythm rewrite
  -> degree-conditioned grace/slide/vibrato
  -> optional degree microtuning
  -> clean synthesis
  -> WAV + event CSV + metrics JSON
```

## Included styles

The package includes a deliberately small set of cultural/experimental profiles:

- `chinese_gong`
- `japanese_in`
- `korean_pyeongjo`
- `arabic_hijaz`
- `indian_bhairav`
- `irish_dorian`
- `spanish_flamenco`
- `hungarian_minor`
- `swedish_dorian_polska`
- `neutral_major` (reference)

The European profiles are intentionally narrow experiments rather than national summaries.
For example, `irish_dorian` targets Dorian-colored Irish tune behavior, while
`swedish_dorian_polska` is a Dorian/Polska-inspired engineering approximation.

List them with:

```bash
python scaleify.py --list-styles
```

## Install

```bash
python -m pip install -r requirements.txt
```

`imageio-ffmpeg` is only the robust decoder fallback for formats such as MP3.

## Basic use

```bash
python scaleify.py test.wav \
  --style japanese_in \
  --root C \
  --style-amount 0.9 \
  --rhythm-amount 0.8 \
  --timbre pluck
```

For `Erika` generated in G major, for example:

```bash
python scaleify.py test/erika_test.wav \
  --style arabic_hijaz \
  --root G \
  --style-amount 0.9 \
  --timbre reed
```

Automatic root estimation is also supported:

```bash
python scaleify.py melody.mp3 --style chinese_gong --root auto
```

## Feature switches

```bash
--no-rhythm
--no-ornaments
--no-microtuning
--no-modulation
```

This makes ablation tests straightforward.

## Reports

Each run produces:

```text
reports/<input>_<style>_events.csv
reports/<input>_<style>_metrics.json
```

The metrics JSON contains:

- mean / median / maximum pitch displacement
- melody preservation score
- contour preservation score
- scale compliance
- preferred interval score
- direction-aware transition score
- trigram / phrase score
- cadence score
- aggregate grammar score
- preferred phrase hit count
- rhythm-change magnitude
- modulation phrase count

These are **engineering diagnostics**, not perceptual or musicological ground truth.

## External style profiles

All culture-specific behavior lives under `styles/*.json`. Python does not need to be edited to add a style. Copy `styles/template.json`, change `id`, and tune the profile.

### `scale`

Semitone offsets from the selected root in 12-TET.

```json
"scale": [0, 1, 5, 7, 8]
```

### `grammar`

Important fields:

```json
"interval_weights": {"1": 1.8, "4": 1.6},
"degree_weights": {"0": 1.3, "5": 0.8},
"ascending_transition_weights": {"0>1": 1.4},
"descending_transition_weights": {"1>0": 1.5},
"trigram_weights": {"0>1>5": 2.0},
"preferred_phrases": [
  {"degrees": [0, 1, 5, 7], "weight": 2.2}
],
"cadence_patterns": [
  {"degrees": [5, 1, 0], "weight": 2.0}
]
```

The DP state retains recent target-note history, so trigrams and short multi-note phrases affect the selected melody rather than being applied only after rendering.

### `rhythm`

Rhythm is transformed at note-event level. It can quantize relative durations, hold structural degrees longer, alter gaps, and extend phrase endings. With `preserve_phrase_duration=true`, local rhythm changes do not change the total phrase length.

### `ornaments`

Supported types:

- `grace`
- `slide_in`
- `slide_out`
- `vibrato`

Rules can be restricted to particular scale degrees.

### `tuning`

Optional degree-specific cent offsets:

```json
"tuning": {
  "degree_cents": {"1": -10, "4": -6}
}
```

The shipped non-zero values are deliberately conservative **experimental stylization parameters**, not authoritative intonation tables.

### `modulation`

Phrase-level alternate roots/scales can be declared externally:

```json
"modulation": {
  "enabled": true,
  "options": [
    {
      "name": "alternate_mode",
      "root_offset": 5,
      "scale": [0, 2, 3, 5, 7, 8, 10],
      "min_phrase_events": 6,
      "switch_penalty": 3.0,
      "activation_bonus": 0.3
    }
  ]
}
```

For each detected phrase, the mapper compares the base style and allowed modulation candidates and chooses the lower-cost path. This is a generic engineering abstraction; it is not a complete maqam/raga modulation model.

## Important limitation

The included profiles are heuristic style-transfer models. A culture or tradition cannot be reduced to one scale or one JSON file. The profiles intentionally focus on a few audible cues for experimentation: scale-degree use, melodic transition statistics, phrase endings, rhythm, ornaments, and tuning.

## Onset-aware repeated-note segmentation (v9.1)

Pitch changes alone cannot distinguish repeated equal-pitch notes such as
`C C | G G | A A`. v9.1 therefore combines F0 segmentation with an onset
/re-attack detector. A new note event begins when either the tracked pitch
changes enough or a sufficiently separated onset is detected.

Relevant CLI options:

```bash
--onset-delta 0.15
--onset-min-separation-ms 70
--onset-retrigger-min-ms 80
--no-onset-segmentation
```

- `--onset-delta`: higher values are more conservative. For clean synthetic
  melody tests, about `0.10` to `0.20` is a useful range.
- `--onset-min-separation-ms`: minimum spacing between attack candidates.
- `--onset-retrigger-min-ms`: a detected onset cannot split a note until the
  current note is at least this old. This prevents the initial attack from
  splitting a single note into two events.
- `--no-onset-segmentation`: restores the old F0-change-only behavior for
  ablation/debugging.

Regression check with the bundled/generated Twinkle melody:

- old F0-only segmentation: 24 detected events
- onset-aware segmentation: 42 detected events

The expected melody contains 42 pitched notes.


## Corpus-driven style tuning

`train_style.py` updates an existing style profile from every WAV file in a
folder. The source JSON is never overwritten.

```bash
python train_style.py styles/japanese_in.json corpus/japanese
```

Default output:

```text
styles/japanese_in_tuned.json
styles/japanese_in_tuned_training_report.json
```

If `japanese_in_tuned.json` already exists, the trainer creates
`japanese_in_tuned_2.json`, then `_3`, etc. The output profile also receives a
new style `id`, so the original and tuned JSON can stay in the same `styles/`
directory without duplicate-ID errors.

Explicit output:

```bash
python train_style.py \
    styles/japanese_in.json \
    corpus/japanese \
    --output styles/japanese_in_corpus.json
```

The trainer learns target-corpus statistics for degree/interval weights,
direction-aware transitions, trigrams, recurring 4/5-note phrases, cadence
behavior, relative note-duration patterns, and optional degree-specific
microtuning. It preserves transform-safety parameters, ornaments and modulation
policy because those cannot be inferred reliably from target WAVs alone.

For best results, use monophonic or melody-dominant WAV files.


# v10: unsupervised corpus trainer

The trainer no longer requires an input style JSON.

```bash
python train_style.py dataset/japan
```

Default output:

```text
styles/generated/
├── japan_cluster_1.json
├── japan_cluster_2.json
├── ...
├── japan_cluster_assignments.csv
└── japan_cluster_report.json
```

The number of clusters is selected automatically by NumPy k-means +
silhouette score:

```bash
python train_style.py dataset/japan --clusters auto
```

Or force a count:

```bash
python train_style.py dataset/japan --clusters 3
```

The generated styles infer their own scale. Notes outside the inferred scale
are excluded from melodic grammar and break n-gram runs. Equal-note re-attacks
remain in rhythm statistics but are excluded from melodic transition weights.
Cadence patterns require both minimum occurrence count and cross-file support.

No ornament or modulation rules are invented automatically.
