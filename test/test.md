```sh
# Install dependencies
pip3 install -r requirements-scaleify.txt

# Generate test audio file
python3 gen_test_audio.py
python3 scaleify.py test/test_audio.wav --no-demucs --root C
```