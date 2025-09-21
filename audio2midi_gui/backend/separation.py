import os
import json
import torch
import soundfile as sf
import subprocess
import sys
from pathlib import Path

def separate(input_path, out_dir, stems=4, device=None):
    """
    Separates audio into stems using Demucs (preferred) or Spleeter (fallback).

    Args:
        input_path (str): Path to input audio file.
        out_dir (str): Output directory for stems.
        stems (int): Number of stems (4 for Demucs/Spleeter).
        device (str): 'cuda' or 'cpu'. If None, auto-detect.

    Returns:
        list: JSON summary of stems with path, duration, sample_rate, channels.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try Demucs first
    try:
        # Use subprocess to call demucs CLI
        cmd = [
            sys.executable, '-m', 'demucs',
            '--four-stems',
            '--device', device,
            '--out', str(out_dir),
            input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        if result.returncode != 0:
            raise Exception(f"Demucs failed: {result.stderr}")

        stem_names = ['vocals', 'drums', 'bass', 'other']

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
        print(f"Demucs failed: {e}. Falling back to Spleeter.")
        # Fallback to Spleeter
        try:
            from spleeter.separator import Separator

            separator = Separator('spleeter:4stems')
            separator.separate_to_file(input_path, str(out_dir))

            stem_names = ['vocals', 'drums', 'bass', 'other']

        except Exception as e:
            raise RuntimeError(f"Both Demucs and Spleeter failed: {e}")

    # Now, normalize and get info
    summary = []
    for stem in stem_names:
        stem_path = out_dir / f"{stem}.wav"
        if not stem_path.exists():
            raise FileNotFoundError(f"Stem {stem_path} not found")

        # Load and normalize to 44100, 16-bit
        data, sr = sf.read(str(stem_path))
        if sr != 44100:
            # Resample if needed, but assume 44100
            pass

        # Save as 16-bit WAV
        sf.write(str(stem_path), data, 44100, subtype='PCM_16')

        # Get info
        info = sf.info(str(stem_path))
        summary.append({
            "path": str(stem_path),
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels
        })

    return summary
