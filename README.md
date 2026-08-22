# Scaleify v8

`scaleify_fullmix_v8.py`는 v7의 오디오 파이프라인을 유지하면서, 음계 선택을 단순 nearest-note quantization에서 **event-level Viterbi melodic-grammar mapping**으로 바꾼 버전입니다.

## 핵심 변경

- 문화권/스타일 특성을 Python 코드 밖의 `styles/*.json`으로 분리
- 초기 프로필은 5개 문화권 + 비교용 Major만 제공
- 각 프로필은 다음을 독립적으로 정의
  - `scale`: 12-TET pitch-class 집합
  - `interval_weights`: 선호하는 음정 크기
  - `degree_weights`: 중심음/구조음 선호도
  - `transition_weights`: 특정 scale-degree 진행 선호도
  - `cadence_degrees`: 종지 선호도
  - `ornament`: grace note / vibrato 설정
- Viterbi는 원 멜로디와 너무 멀어지지 않으면서 위 스타일 문법 점수가 높은 target-note path를 선택
- JSON 하나를 추가하면 코드를 수정하지 않아도 `--style`로 사용 가능

> 주의: 이 프로필들은 스타일 변환 실험을 위한 heuristic입니다. 특히 maqam/raga/국악은 실제로 미분음, 시김새, 상·하행, 음의 기능, 관용구, 장단 등 더 많은 요소로 정의됩니다.

## 현재 프로필

```text
neutral_major
chinese_gong
japanese_in
korean_pyeongjo
arabic_hijaz
indian_bhairav
```

확인:

```bash
python scaleify_fullmix_v8.py --list-styles
```

## 단선율 테스트

```bash
python scaleify_fullmix_v8.py test.wav \
  --no-demucs \
  --style japanese_in \
  --root C \
  --mix-mode replace \
  --pitch-method yin \
  --timbre pluck \
  --style-amount 1.0
```

Hijaz 비교:

```bash
python scaleify_fullmix_v8.py test.wav \
  --no-demucs \
  --style arabic_hijaz \
  --root C \
  --mix-mode replace \
  --pitch-method yin \
  --timbre reed \
  --style-amount 1.0
```

## MP3 / full-mix 테스트

```bash
python scaleify_fullmix_v8.py song.mp3 \
  --style japanese_in \
  --root auto \
  --target-stems vocals \
  --mix-mode hybrid \
  --pitch-method yin \
  --style-amount 0.8
```

악기 위주의 곡은 `--target-stems other`를 시험할 수 있지만, Demucs `other`가 다성음이면 YIN/pYIN이 한 개의 지배적 F0만 추적한다는 한계가 있습니다.

## 새 스타일 추가

예를 들어 `styles/my_style.json`을 추가합니다.

```json
{
  "id": "my_style",
  "label": "My Style",
  "region": "Example",
  "description": "Experimental profile",
  "scale": [0, 2, 5, 7, 9],
  "grammar": {
    "pitch_deviation_weight": 0.8,
    "motion_preservation_weight": 0.25,
    "contour_penalty": 0.8,
    "leap_penalty": 0.1,
    "max_preferred_leap": 7,
    "candidate_shift_semitones": 5,
    "candidate_count": 6,
    "event_pitch_change": 0.8,
    "min_event_frames": 2,
    "interval_weights": {
      "2": 1.0,
      "3": 1.3
    },
    "degree_weights": {
      "0": 1.2,
      "7": 0.8
    },
    "transition_weights": {
      "7>0": 1.0,
      "2>0": 0.5
    },
    "cadence_degrees": {
      "0": 1.5
    }
  },
  "ornament": {
    "grace_probability": 0.3,
    "grace_scale_steps": -1,
    "grace_fraction": 0.15,
    "vibrato_cents": 10,
    "vibrato_hz": 5.0,
    "vibrato_degrees": [0, 7]
  }
}
```

그 뒤 바로:

```bash
python scaleify_fullmix_v8.py --list-styles
python scaleify_fullmix_v8.py test.wav --no-demucs --style my_style --root C --mix-mode replace
```

## Weight 의미

### `interval_weights`

출력 멜로디의 두 연속 음 사이 절대 semitone 간격에 대한 **보너스**입니다.

```json
"interval_weights": {
  "1": 1.5,
  "3": 2.0
}
```

이면 반음과 단3도/증2도 크기의 움직임을 더 선호합니다.

### `degree_weights`

으뜸음을 0으로 본 pitch-class offset에 대한 보너스입니다.

```json
"degree_weights": {
  "0": 1.5,
  "7": 1.0
}
```

이면 tonic과 perfect fifth에 더 오래 머무르는 경로가 유리해집니다.

### `transition_weights`

scale degree offset 사이의 방향성 있는 전이에 대한 보너스입니다.

```json
"transition_weights": {
  "0>1": 1.5,
  "1>4": 2.0,
  "4>5": 1.5
}
```

Hijaz의 특징적인 하부 jins 진행 같은 것을 강하게 유도할 때 사용합니다.

### `style-amount`

- `0`: 스타일 문법 bonus를 끄고 원 멜로디 보존을 우선
- `1`: JSON에 정의된 grammar/ornament를 최대 강도로 적용

음계 자체는 `style-amount=0`이어도 유지됩니다.

## 파일 구조

```text
scaleify_v8/
├── scaleify_fullmix_v8.py
├── style_profiles.py
├── requirements.txt
├── README.md
└── styles/
    ├── neutral_major.json
    ├── chinese_gong.json
    ├── japanese_in.json
    ├── korean_pyeongjo.json
    ├── arabic_hijaz.json
    └── indian_bhairav.json
```
