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

To create a standalone executable:

### Linux/macOS
Run the build script:
```
chmod +x build.sh
./build.sh
```
This uses PyInstaller to create a single-folder app in `dist/audio2midi_gui/`.

### Windows
Install PyInstaller and run:
```
pip install pyinstaller
pyinstaller --clean --noconfirm --onedir --name audio2midi_gui app/main.py --hidden-import backend.separation --hidden-import backend.instrument_detect --hidden-import backend.transcribe --hidden-import backend.midi_writer --add-data models;models --add-data tests/assets;tests/assets
```
Find the app in `dist/audio2midi_gui/`.

### GPU Environments
For GPU acceleration:
- Install CUDA toolkit 11.8 (compatible with PyTorch 2.0.1).
- Use: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- For TensorFlow (Crepe/Spleeter): Install compatible CUDA 11.8.
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

## Troubleshooting

### Missing CUDA
Error: "CUDA not available"
- Install CUDA 11.8 from NVIDIA website.
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Out of Memory (OOM)
- Reduce batch size or use CPU mode.
- For Demucs: Set device='cpu' in options.

### Model Download Failures
- Check internet; models download on first use.
- Manual download: For Demucs, run `python -c "from demucs import pretrained; pretrained.get_model('mdx_extra')"`
- For Spleeter: Run a separation command once.

### GUI Not Starting
- Ensure PySide6 is installed.
- On Linux, install system deps: `apt-get install libxcb-xinerama0`

## License

Apache-2.0
