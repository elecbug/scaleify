```sh
# Install dependencies
pip3 install -r requirements.txt

# Generate test audio file
python3 generator/twinkle_twinkle_generator.py
python3 generator/korobeiniki_generator.py
python3 generator/erika_generator.py

# Run scaleify with the test audio file
python3 scaleify.py test/twinkle_twinkle_test.wav --style arabic_hijaz --root C --style-amount 0.9 --rhythm-amount 0.55 --timbre sine
python3 scaleify.py test/twinkle_twinkle_test.wav --style chinese_gong --root C --style-amount 0.9 --rhythm-amount 0.55 --timbre sine
python3 scaleify.py test/twinkle_twinkle_test.wav --style hungarian_minor --root C --style-amount 0.9 --rhythm-amount 0.55 --timbre sine
python3 scaleify.py test/twinkle_twinkle_test.wav --style indian_bhairav --root C --style-amount 0.9 --rhythm-amount 0.55 --timbre sine
python3 scaleify.py test/twinkle_twinkle_test.wav --style irish_dorian --root C --style-amount 0.9 --rhythm-amount 0.55 --timbre sine
python3 scaleify.py test/twinkle_twinkle_test.wav --style japanese_in --root C --style-amount 0.9 --rhythm-amount 0.55 --timbre sine
python3 scaleify.py test/twinkle_twinkle_test.wav --style korean_pyeongjo --root C --style-amount 0.9 --rhythm-amount 0.55 --timbre sine
python3 scaleify.py test/twinkle_twinkle_test.wav --style spanish_flamenco --root C --style-amount 0.9 --rhythm-amount 0.55 --timbre sine
python3 scaleify.py test/twinkle_twinkle_test.wav --style swedish_dorian_polska --root C --style-amount 0.9 --rhythm-amount 0.55 --timbre sine

python3 scaleify.py test/korobeiniki_test.wav --style arabic_hijaz --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre pluck
python3 scaleify.py test/korobeiniki_test.wav --style chinese_gong --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre pluck
python3 scaleify.py test/korobeiniki_test.wav --style hungarian_minor --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre pluck
python3 scaleify.py test/korobeiniki_test.wav --style indian_bhairav --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre pluck
python3 scaleify.py test/korobeiniki_test.wav --style irish_dorian --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre pluck
python3 scaleify.py test/korobeiniki_test.wav --style japanese_in --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre pluck
python3 scaleify.py test/korobeiniki_test.wav --style korean_pyeongjo --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre pluck
python3 scaleify.py test/korobeiniki_test.wav --style spanish_flamenco --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre pluck
python3 scaleify.py test/korobeiniki_test.wav --style swedish_dorian_polska --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre pluck

python3 scaleify.py test/erika_test.wav --style arabic_hijaz --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/erika_test.wav --style chinese_gong --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/erika_test.wav --style hungarian_minor --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/erika_test.wav --style indian_bhairav --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/erika_test.wav --style irish_dorian --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/erika_test.wav --style japanese_in --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/erika_test.wav --style korean_pyeongjo --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/erika_test.wav --style spanish_flamenco --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/erika_test.wav --style swedish_dorian_polska --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
```

```sh
# Feature switches
--no-rhythm
--no-ornaments
--no-microtuning
--no-modulation
```

```sh
python3 generator/dataset/japan_1892_dataset_generator.py
python3 train_style.py dataset/japan/ --output data/styles_tuned

python3 scaleify.py test/erika_test.wav --style japan_cluster_1 --style-dir data/styles_tuned/ --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/erika_test.wav --style japan_cluster_2 --style-dir data/styles_tuned/ --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/erika_test.wav --style japan_cluster_3 --style-dir data/styles_tuned/ --root G --style-amount 0.9 --rhythm-amount 0.55 --timbre reed

python3 scaleify.py test/korobeiniki_test.wav --style japan_cluster_1 --style-dir data/styles_tuned/ --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/korobeiniki_test.wav --style japan_cluster_2 --style-dir data/styles_tuned/ --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
python3 scaleify.py test/korobeiniki_test.wav --style japan_cluster_3 --style-dir data/styles_tuned/ --root A --style-amount 0.9 --rhythm-amount 0.55 --timbre reed
```
