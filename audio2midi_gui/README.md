# Audio2MIDI GUI

A Python-based GUI application for converting audio files to MIDI using source separation, transcription, and instrument detection.

## Requirements

- Python 3.10+
- For GPU support: CUDA-compatible GPU (optional, CPU fallback available)

## Installation

1. Clone or download the repository.
2. Install dependencies:
   ```
   python -m pip install -r requirements.txt
   ```

### Dependency Notes

- This project uses both PyTorch (for Demucs) and TensorFlow (for Crepe and Spleeter). Versions are pinned to ensure compatibility.
- If you encounter conflicts, consider using virtual environments or Docker.
- For GPU acceleration, install PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (adjust for your CUDA version).
- CPU-only: The pinned versions default to CPU.

## Model Downloads

Before running, download required models:

- Demucs models: Automatically downloaded on first use.
- Crepe model: Automatically downloaded on first use.
- Spleeter models: Run `spleeter separate -p spleeter:2stems -o output/ audio_file.mp3` once to download models (replace with your audio file).

## Running the Application

Run the GUI:
```
python app/main.py
```

Or use the script:
```
run.bat  # On Windows
```

## Packaging

To create an executable:
```
pyinstaller --onefile app/main.py
```

This will generate a standalone executable in the `dist/` folder.

## License

Apache-2.0
