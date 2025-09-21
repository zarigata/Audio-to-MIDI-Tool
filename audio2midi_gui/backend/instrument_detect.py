import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import soundfile as sf
from pathlib import Path

# Pre-trained k-NN classifier with synthetic data
# Features: [mfcc_mean, spectral_centroid, zero_crossing_rate, rms]
# Labels: vocals, drums, bass, piano, guitar, synth, unknown

# Synthetic training data
training_features = [
    [0.1, 3000, 0.1, 0.3],  # vocals
    [0.05, 2000, 0.05, 0.5],  # drums
    [0.0, 1500, 0.02, 0.4],  # bass
    [0.2, 2500, 0.08, 0.2],  # piano
    [0.15, 2200, 0.06, 0.25],  # guitar
    [0.12, 1800, 0.04, 0.35],  # synth
    [0.08, 2000, 0.03, 0.3],  # unknown
]

training_labels = ['vocals', 'drums', 'bass', 'piano', 'guitar', 'synth', 'unknown']

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(training_features, training_labels)

def extract_features(audio, sr):
    # Compute features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean()

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0].mean()

    rms = librosa.feature.rms(y=audio)[0].mean()

    # For simplicity, use mean mfcc, centroid, zcr, rms
    features = [mfcc_mean[0], spectral_centroid, zero_crossing_rate, rms]  # Use first MFCC

    return features

def analyze_stem(stem_path, use_advanced=False):
    """
    Analyze stem to determine instrument type.

    Args:
        stem_path (str): Path to stem WAV.
        use_advanced (bool): If True, use external model (placeholder).

    Returns:
        str: Instrument type.
    """
    if use_advanced:
        # Placeholder for MT3 or musicnn
        # For now, return 'unknown' or call external
        return 'unknown'  # TODO: integrate MT3/musicnn

    audio, sr = librosa.load(stem_path, sr=None)
    features = extract_features(audio, sr)
    prediction = knn.predict([features])[0]
    return prediction

def choose_transcription_model(stem_info):
    """
    Choose transcription model based on stem info.

    Args:
        stem_info (dict): Info from analyze_stem or separation summary.

    Returns:
        str: Recommended model.
    """
    instrument = stem_info.get('instrument', 'unknown')

    if instrument == 'piano':
        return 'onsets_frames'
    elif instrument in ['vocals', 'guitar']:
        return 'crepe_monophonic'
    elif instrument == 'drums':
        return 'percussion_template'
    else:  # polyphonic, synth, unknown
        # Check if MT3 available, else heuristic
        try:
            import mt3  # Placeholder
            return 'mt3'
        except ImportError:
            return 'heuristic_polyphonic'
