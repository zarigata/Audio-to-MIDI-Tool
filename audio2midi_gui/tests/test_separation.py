import unittest
import os
import tempfile
from pathlib import Path
from backend.separation import separate

class TestSeparation(unittest.TestCase):

    def test_separate_short_mix(self):
        # Test with the short mix
        input_path = 'tests/assets/mix_short.wav'
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / 'stems'
            summary = separate(input_path, str(out_dir))

            # Check if files exist
            expected_stems = ['vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']
            for stem in expected_stems:
                self.assertTrue((out_dir / stem).exists(), f"Stem {stem} not found")

            # Check summary
            self.assertEqual(len(summary), 4)
            for item in summary:
                self.assertIn('path', item)
                self.assertIn('duration', item)
                self.assertIn('sample_rate', item)
                self.assertIn('channels', item)
                self.assertGreater(item['duration'], 0)
                self.assertEqual(item['sample_rate'], 44100)

if __name__ == '__main__':
    unittest.main()
