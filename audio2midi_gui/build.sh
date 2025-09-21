#!/bin/bash
# Build script for Linux/macOS
# Requires PyInstaller installed

echo "Building audio2midi_gui..."

# Install deps if needed
# pip install -r requirements.txt

# Run PyInstaller
pyinstaller --clean --noconfirm pyinstaller.spec

echo "Build complete. Find the app in dist/audio2midi_gui/"
