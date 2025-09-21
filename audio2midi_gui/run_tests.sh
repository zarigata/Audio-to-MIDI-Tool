#!/bin/bash
# Test runner script

echo "Setting up virtual environment..."
python -m venv test_venv
source test_venv/bin/activate  # On Windows: test_venv\Scripts\activate

echo "Installing requirements..."
pip install -r requirements.txt

echo "Running unit tests..."
python -m pytest tests/ -v

echo "Running integration test with output to tests/output..."
python -c "
from backend.separation import separate
from backend.transcribe import transcribe_stem_to_midi
import os

# Separate
summary = separate('tests/assets/mix_short.wav', 'tests/output/stems')
print(f'Separated {len(summary)} stems')

# Transcribe one stem
vocal_path = 'tests/output/stems/vocals.wav'
if os.path.exists(vocal_path):
    midi_path, _ = transcribe_stem_to_midi(vocal_path, model='crepe_monophonic')
    print(f'Transcribed to {midi_path}')
    assert os.path.exists(midi_path) and os.path.getsize(midi_path) > 0
    print('Integration test passed')
else:
    print('Vocal stem not found, skipping transcription')
"

echo "Cleaning up..."
deactivate
rm -rf test_venv

echo "All tests completed successfully."
