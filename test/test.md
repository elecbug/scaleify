```sh
# Install dependencies
pip3 install -r requirements-scaleify.txt

# Generate test audio file
python3 generator/twinkle_twinkle_generator.py
python3 generator/korobeiniki_generator.py

# Run scaleify with the test audio file
python3 scaleify.py test/twinkle_twinkle_test.wav --style japanese_in --root C --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/twinkle_twinkle_test.wav --style arabic_hijaz --root C --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/twinkle_twinkle_test.wav --style iwato_12tet --root C --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/twinkle_twinkle_test.wav --style indian_todi --root C --no-demucs --mix-mode replace --style-amount 1.0

python3 scaleify.py test/korobeiniki_test.wav --style japanese_in --root A --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/korobeiniki_test.wav --style arabic_hijaz --root A --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/korobeiniki_test.wav --style iwato_12tet --root A --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/korobeiniki_test.wav --style indian_todi --root A --no-demucs --mix-mode replace --style-amount 1.0

# Run scaleify with the test audio file and Demucs for full-mix testing
# python3 scaleify.py test/korobeiniki_test.wav --style japanese_in --root A --target-stems other --mix-mode hybrid --pitch-method yin --timbre pluck --device cuda --style-amount 1.0
# python3 scaleify.py test/korobeiniki_test.wav --style arabic_hijaz --root A --target-stems other --mix-mode hybrid --pitch-method yin --timbre pluck --device cuda --style-amount 1.0
# python3 scaleify.py test/korobeiniki_test.wav --style iwato_12tet --root A --target-stems other --mix-mode hybrid --pitch-method yin --timbre pluck --device cuda --style-amount 1.0
# python3 scaleify.py test/korobeiniki_test.wav --style indian_todi --root A --target-stems other --mix-mode hybrid --pitch-method yin --timbre pluck --device cuda --style-amount 1.0
```