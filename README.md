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

In particular, harmony, accompaniment, lyrics, singer identity, instrumentation, and production style are outside the current scope.

## Basic workflow

A typical experiment consists of:

```text
melody corpus
    ↓
unsupervised style learning
    ↓
melodic style profile
    ↓
target melody
    ↓
style transfer
    ↓
transformed melody
```

Scaleify can work with manually written style profiles, but the current research direction primarily uses profiles learned directly from corpora.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Style transfer

Example:

```bash
python scaleify.py melody.wav \
    --style japan_cluster_1 \
    --root C \
    --style-amount 0.9 \
    --rhythm-amount 0.55 \
    --timbre reed
```

Automatic root estimation is also supported:

```bash
python scaleify.py melody.wav \
    --style japan_cluster_1 \
    --root auto
```

## Corpus-driven training

The current trainer can infer style profiles directly from a folder of melodies.

```bash
python train_style.py dataset/japan
```

The trainer performs unsupervised clustering and learns a separate melodic profile for each discovered cluster.

Learned properties include:

* core pitch-class vocabulary
* auxiliary pitch classes
* interval statistics
* transition statistics
* recurring short phrases
* cadence behavior
* rhythm tendencies

The number of clusters can be selected automatically or specified manually.

```bash
python train_style.py dataset/japan --clusters auto
```

```bash
python train_style.py dataset/japan --clusters 3
```

## Pitch vocabulary

Early Scaleify versions assumed that a style could be represented by a relatively small musical scale.

This worked reasonably well for several traditional-music corpora, where compact 5–7 note structures often explained most of the melodic material.

Modern corpora exposed a limitation of this assumption.

For example, recent Vocaloid experiments required approximately 9–10 pitch classes to explain more than 90% of the melody events.

For this reason, the current `scale` representation is better interpreted more generally as a **core pitch-class vocabulary** when working with chromatic modern music.

## Experimental corpora

Current experiments include corpora related to:

* Japanese traditional music
* Korean traditional music
* Chinese traditional music
* JSMel
* modern Vocaloid / DECO*27 melodies

These datasets are used to study whether the same unsupervised representation can recover both compact traditional pitch structures and broader modern melodic vocabularies.

The resulting profiles describe the training corpus only. They should not be interpreted as complete representations of a country, culture, genre, or historical period.

## Vocaloid experiment

Scaleify includes an experimental workflow for symbolic singing-synth melodies.

The current corpus primarily uses officially distributed DECO*27 / OTOIRO melody data.

Only the monophonic vocal melody is modeled. Lyrics, voicebank characteristics, tuning curves, accompaniment, and production are intentionally excluded.

Despite this restriction, preliminary listening tests suggest that corpus-derived profiles can introduce melodic behavior that listeners may associate with modern Vocaloid composition.

This remains a perceptual research question rather than a validated conclusion.

## Listening tests

Several familiar melodies are currently used as test material, including:

* Erika
* Korobeiniki
* Twinkle Twinkle Little Star

Using familiar source melodies makes it easier to compare how strongly different learned profiles alter the perceived melodic character.

For formal experiments, transformed files should be presented blindly and randomized.

One planned evaluation asks listeners to select the transformation that feels most familiar, then compares those choices with their prior exposure to musical traditions or Vocaloid music.

## Diagnostics

Scaleify produces numerical metrics for properties such as:

* pitch displacement
* contour preservation
* interval preference
* transition consistency
* phrase-pattern matching
* cadence behavior
* rhythm modification

These metrics are intended as engineering diagnostics.

They are not measures of cultural authenticity or perceptual quality.

## Project evolution

Scaleify has progressed roughly through the following stages:

```text
handcrafted scale/style profiles
        ↓
onset-aware melody extraction
        ↓
corpus tuning of existing profiles
        ↓
fully unsupervised style learning
        ↓
core + auxiliary pitch vocabulary
        ↓
traditional and modern corpus comparison
```

The current research question is no longer simply:

> Can a melody be converted to a predefined cultural scale?

Instead, it is closer to:

> Can recurring melodic characteristics be learned from a corpus and transferred to another melody in a perceptually meaningful way?

## Limitations

Scaleify is still a research prototype.

Important limitations include:

* melody-only modeling
* dependence on corpus quality
* imperfect tonic estimation
* audio transcription errors
* heuristic cluster-count selection
* limited modeling of chromatic and modulating music
* no formal perceptual validation yet

The system should therefore be treated as an experimental tool for studying melodic representation and transfer, not as an automatic classifier of musical cultures.

## Future work

Current directions include:

* direct symbolic-data training
* improved cluster selection and stability analysis
* better chromatic pitch modeling
* larger modern-music corpora
* producer-level melodic comparison
* formal blind listening studies