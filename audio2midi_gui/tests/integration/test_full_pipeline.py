import unittest
import os
import tempfile
from pathlib import Path
from backend.separation import separate
from backend.instrument_detect import analyze_stem, choose_transcription_model
from backend.transcribe import transcribe_stem_to_midi

class TestFullPipeline(unittest.TestCase):

    def test_full_pipeline(self):
        # Use the mix_short.wav
        input_path = 'tests/assets/mix_short.wav'
        self.assertTrue(os.path.exists(input_path), "Test asset missing")

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / 'stems'
            # Separate
            summary = separate(input_path, str(out_dir))
            self.assertIsInstance(summary, list)
            self.assertEqual(len(summary), 4)  # 4 stems

            # Check stems exist
            for stem_info in summary:
                self.assertTrue(os.path.exists(stem_info['path']))

            # Analyze one stem
            vocal_path = out_dir / 'vocals.wav'
            if vocal_path.exists():
                instrument = analyze_stem(str(vocal_path))
                self.assertIn(instrument, ['vocals', 'drums', 'bass', 'piano', 'guitar', 'synth', 'unknown'])

                # Choose model
                model = choose_transcription_model({'instrument': instrument})
                self.assertIn(model, ['onsets_frames', 'crepe_monophonic', 'percussion_template', 'mt3', 'heuristic_polyphonic'])

                # Transcribe
                midi_path, trans_summary = transcribe_stem_to_midi(str(vocal_path), instrument_hint=instrument, model=model)
                self.assertTrue(os.path.exists(midi_path))
                self.assertIn('notes', trans_summary)

if __name__ == '__main__':
    unittest.main()
