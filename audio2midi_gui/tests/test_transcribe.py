import unittest
import os
from backend.transcribe import transcribe_stem_to_midi

class TestTranscribe(unittest.TestCase):

    def test_transcribe_sine_notes(self):
        stem_path = 'tests/assets/sine_notes.wav'
        midi_path, summary = transcribe_stem_to_midi(stem_path, instrument_hint='piano', model='crepe_monophonic')

        # Check MIDI exists
        self.assertTrue(os.path.exists(midi_path))

        # Check notes
        notes = summary['notes']
        self.assertGreater(len(notes), 0)

        # Expected onsets: 0, 0.5, 1.0
        expected_onsets = [0, 0.5, 1.0]
        for i, note in enumerate(notes[:3]):
            self.assertAlmostEqual(note['onset'], expected_onsets[i], delta=0.05)  # 50ms tolerance

        # Pitches: C4=60, D4=62, E4=64
        expected_pitches = [60, 62, 64]
        for i, note in enumerate(notes[:3]):
            self.assertAlmostEqual(note['pitch'], expected_pitches[i], delta=2)

if __name__ == '__main__':
    unittest.main()
