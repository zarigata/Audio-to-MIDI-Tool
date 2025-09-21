import os
import json
import numpy as np
import librosa
import pretty_midi
import crepe
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# GM program mapping
INSTRUMENT_TO_PROGRAM = {
    'piano': 0,
    'guitar': 24,
    'bass': 32,
    'drums': 0,  # Percussion
    'vocals': 52,  # Choir
    'synth': 80,
    'unknown': 0
}

def transcribe_stem_to_midi(stem_path, instrument_hint=None, model='auto', out_midi_path=None, device='cpu', time_precision=10):
    """
    Transcribe stem to MIDI.

    Returns: midi_path, summary_dict
    """
    if out_midi_path is None:
        stem_name = Path(stem_path).stem
        out_midi_path = f"{stem_name}.mid"

    audio, sr = librosa.load(stem_path, sr=None)

    # Detect tempo
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

    # Choose model
    if model == 'auto':
        if instrument_hint == 'piano':
            model = 'onsets_frames'
        elif instrument_hint in ['vocals', 'guitar']:
            model = 'crepe_monophonic'
        elif instrument_hint == 'drums':
            model = 'percussion_template'
        else:
            if device == 'cuda':
                try:
                    import mt3
                    model = 'mt3'
                except ImportError:
                    model = 'heuristic_polyphonic'
            else:
                model = 'heuristic_polyphonic'

    # Run transcription
    if model == 'onsets_frames':
        notes = _transcribe_onsets_frames(audio, sr)
    elif model == 'crepe_monophonic':
        notes = _transcribe_crepe_mono(audio, sr, time_precision)
    elif model == 'mt3':
        notes = _transcribe_mt3(audio, sr)
    else:
        notes = _transcribe_heuristic(audio, sr)

    # Create MIDI
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=INSTRUMENT_TO_PROGRAM.get(instrument_hint, 0))
    for note in notes:
        midi_note = pretty_midi.Note(
            velocity=int(note['velocity'] * 127),
            pitch=note['pitch'],
            start=note['onset'],
            end=note['offset']
        )
        instrument.notes.append(midi_note)
    midi.instruments.append(instrument)
    midi.write(out_midi_path)

    summary = {
        'midi_path': out_midi_path,
        'notes': notes,
        'tempo': tempo,
        'model_used': model
    }

    return out_midi_path, summary

def _transcribe_onsets_frames(audio, sr):
    # Placeholder: basic onset detection
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
    # Assume pitches from chroma or something
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    pitches = []
    for i in range(len(onsets)):
        pitch = np.argmax(chroma[:, min(i, chroma.shape[1]-1)]) + 60  # Rough
        pitches.append(pitch)
    notes = []
    for i, onset in enumerate(onsets):
        notes.append({
            'onset': onset,
            'offset': onset + 0.5,
            'pitch': pitches[i] if i < len(pitches) else 60,
            'velocity': 0.8
        })
    return notes

def _transcribe_crepe_mono(audio, sr, time_precision):
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
    # Onset detection
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
    notes = []
    for onset in onsets:
        idx = np.argmin(np.abs(time - onset))
        pitch = frequency[idx]
        vel = confidence[idx]
        offset = onset + 0.5  # Rough
        # Quantize onset/offset to time_precision ms
        precision_sec = time_precision / 1000
        onset = round(onset / precision_sec) * precision_sec
        offset = round(offset / precision_sec) * precision_sec
        notes.append({
            'onset': onset,
            'offset': offset,
            'pitch': int(round(pitch)),
            'velocity': vel
        })
    return notes

def _transcribe_mt3(audio, sr):
    # Placeholder
    logging.info("MT3 not implemented, using heuristic")
    return _transcribe_heuristic(audio, sr)

def _transcribe_heuristic(audio, sr):
    # Spectral peaks
    stft = librosa.stft(audio)
    mag = np.abs(stft)
    peaks = librosa.util.peak_pick(mag.mean(axis=0), 5, 5, 5, 5, 0.5, 10)
    times = librosa.times_like(stft, sr=sr)
    pitches = []
    for peak in peaks:
        freqs = librosa.fft_frequencies(sr=sr)
        pitch = librosa.hz_to_midi(freqs[np.argmax(mag[:, peak])])
        pitches.append(pitch)
    notes = []
    for i, peak in enumerate(peaks):
        onset = times[peak]
        notes.append({
            'onset': onset,
            'offset': onset + 0.5,
            'pitch': int(round(pitches[i])) if i < len(pitches) else 60,
            'velocity': 0.7
        })
    return notes
