# Scaleify

Scaleify is an experimental framework for **melodic style extraction and transfer**.

It analyzes predominantly monophonic melodies, learns recurring melodic characteristics from a corpus, and applies those characteristics to another melody.

The project originally started as a scale-based cultural style-transfer prototype, but has gradually evolved toward **unsupervised corpus-driven melodic modeling**.

## What Scaleify models

Scaleify currently focuses on melodic characteristics such as:

- pitch-class usage
- preferred intervals
- ascending / descending transitions
- short recurring melodic patterns
- phrase endings and cadence tendencies
- relative note durations
- optional ornaments and microtuning

It does not model full musical style.

Harmony, accompaniment, lyrics, singer identity, instrumentation, and production style are outside the current scope.

## Repository layout

The main working directories are roughly:

```text
datasets/
    training corpora

results/
    generated test audio
    styles/
        learned style profiles

scripts/
    scaleify.py
    train_style.py
    tune_test.sh
    gen/
        dataset generators
```

`tune_test.sh` resolves repository-relative paths automatically.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Style transfer

```bash
python3 scripts/scaleify.py results/twinkle_twinkle_test.wav \
    --style japan_cluster_1 \
    --style-dir results/styles \
    --root C \
    --style-amount 0.9 \
    --rhythm-amount 0.55 \
    --timbre reed
```

Automatic root estimation:

```bash
python3 scripts/scaleify.py melody.wav \
    --style japan_cluster_1 \
    --style-dir results/styles \
    --root auto
```

## Corpus-driven training

```bash
python3 scripts/train_style.py datasets/japan \
    --output results/styles
```

Automatic clustering:

```bash
python3 scripts/train_style.py datasets/japan \
    --output results/styles \
    --clusters auto
```

Fixed cluster count:

```bash
python3 scripts/train_style.py datasets/japan \
    --output results/styles \
    --clusters 3
```

For the Vocaloid corpus:

```bash
python3 scripts/train_style.py datasets/vocaloid \
    --output results/styles \
    --scale-max-notes 12
```

## Pitch vocabulary

Early Scaleify versions assumed that a style could be represented by a relatively small musical scale.

This worked reasonably well for several traditional-music corpora, where compact 5–7 note structures often explained most of the melodic material.

Modern corpora exposed a limitation of this assumption.

For example, recent Vocaloid experiments required approximately 9–10 pitch classes to explain more than 90% of the melody events.

For this reason, the current `scale` representation is better interpreted more generally as a **core pitch-class vocabulary** when working with chromatic modern music.

## Experimental corpora

Current datasets include experiments related to:

* Japanese traditional music
* Korean traditional music
* Chinese traditional music
* JSMel
* modern Vocaloid / DECO*27 melodies

Generated corpora are stored below:

```text
datasets/
```

These datasets are used to study whether the same unsupervised representation can recover both compact traditional pitch structures and broader modern melodic vocabularies.

The resulting profiles describe the training corpus only. They should not be interpreted as complete representations of a country, culture, genre, or historical period.

## Vocaloid experiment

Scaleify includes an experimental workflow for symbolic singing-synth melodies.

The current corpus primarily uses officially distributed DECO*27 / OTOIRO melody data.

Only the monophonic vocal melody is modeled. Lyrics, voicebank characteristics, tuning curves, accompaniment, and production are intentionally excluded.

For Vocaloid experiments, a broader pitch vocabulary can be allowed during training:

```bash
python3 train_style.py ../datasets/vocaloid \
    --output ../results/styles \
    --scale-max-notes 12
```

## Listening tests

Current listening-test melodies are stored in `results/`:

```text
results/
├── erika_test.wav
├── korobeiniki_test.wav
├── twinkle_twinkle_test.wav
└── styles/
```

Transformed files are also written alongside the corresponding test material.

Typical output names are:

```text
erika_test_japan_cluster_1_v9_1.wav
korobeiniki_test_vocaloid_cluster_1_v9_1.wav
twinkle_twinkle_test_china_cluster_2_v9_1.wav
```

For formal perceptual experiments, these filenames should be replaced with randomized blind identifiers and the mapping retained separately.


## Batch experiment helper

The easiest way to reproduce the current dataset generation, training, and listening-test workflow is:

```bash
./scripts/tune_test.sh japan
./scripts/tune_test.sh korea
./scripts/tune_test.sh china
./scripts/tune_test.sh jsmel
./scripts/tune_test.sh vocaloid
```

Force corpus regeneration:

```bash
./scripts/tune_test.sh vocaloid --download
```

Force retraining:

```bash
./scripts/tune_test.sh vocaloid --training
```

Force both:

```bash
./scripts/tune_test.sh vocaloid --download --training
```

Run all configured corpora:

```bash
./scripts/tune_test.sh all
```

The helper uses:

```text
datasets/               corpus data
results/                listening-test audio
results/styles/         trained style profiles
scripts/gen/            dataset generators
```

## Diagnostics

Scaleify produces numerical diagnostics for properties such as:

* pitch displacement
* contour preservation
* interval preference
* transition consistency
* phrase-pattern matching
* cadence behavior
* rhythm modification

These values are engineering diagnostics, not measures of cultural authenticity or perceptual quality.

## Project evolution

Scaleify has progressed roughly through:

```text
handcrafted style profiles
        ↓
onset-aware melody extraction
        ↓
corpus tuning
        ↓
unsupervised style learning
        ↓
core + auxiliary pitch vocabulary
        ↓
traditional and modern corpus comparison
```

The current research question is closer to:

> Can recurring melodic characteristics be learned from a corpus and transferred to another melody in a perceptually meaningful way?

## Limitations

Scaleify is still a research prototype.

Important limitations include:

* melody-only modeling
* dependence on corpus quality
* imperfect tonic estimation
* audio transcription errors
* heuristic cluster selection
* incomplete modeling of chromatic and modulating music
* perceptual validation still in progress

The system should therefore be treated as an experimental tool for studying melodic representation and transfer, not as an automatic classifier of musical cultures.

## Future work

Current directions include:

* direct symbolic-data training
* improved cluster selection and stability analysis
* better chromatic pitch modeling
* larger modern-music corpora
* producer-level melodic comparison
* formal blind listening studies
