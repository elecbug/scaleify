```sh
# Install dependencies
pip3 install -r requirements.txt

# Generate test audio file
python3 generator/twinkle_twinkle_generator.py
python3 generator/korobeiniki_generator.py
python3 generator/erika_generator.py

# Run scaleify with the test audio file
python3 scaleify.py test/twinkle_twinkle_test.wav --style japanese_in --root C --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/twinkle_twinkle_test.wav --style arabic_hijaz --root C --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/twinkle_twinkle_test.wav --style indian_bhairav --root C --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/twinkle_twinkle_test.wav --style chinese_gong --root C --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/twinkle_twinkle_test.wav --style korean_pyeongjo --root C --no-demucs --mix-mode replace --style-amount 1.0

python3 scaleify.py test/korobeiniki_test.wav --style japanese_in --root A --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/korobeiniki_test.wav --style arabic_hijaz --root A --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/korobeiniki_test.wav --style indian_bhairav --root A --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/korobeiniki_test.wav --style chinese_gong --root A --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/korobeiniki_test.wav --style korean_pyeongjo --root A --no-demucs --mix-mode replace --style-amount 1.0

python3 scaleify.py test/erika_test.wav --style japanese_in --root G --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/erika_test.wav --style arabic_hijaz --root G --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/erika_test.wav --style indian_bhairav --root G --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/erika_test.wav --style chinese_gong --root G --no-demucs --mix-mode replace --style-amount 1.0
python3 scaleify.py test/erika_test.wav --style korean_pyeongjo --root G --no-demucs --mix-mode replace --style-amount 1.0
```