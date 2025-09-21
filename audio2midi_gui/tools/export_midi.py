#!/usr/bin/env python
import argparse
import json
from backend.midi_writer import write_midi_from_notes

def main():
    parser = argparse.ArgumentParser(description="Export MIDI from JSON notes")
    parser.add_argument("notes_json", help="Path to JSON file with notes")
    parser.add_argument("out_path", help="Output MIDI path or directory")
    parser.add_argument("--tempo", type=float, default=120, help="Tempo BPM")
    parser.add_argument("--program", type=int, default=0, help="GM program")
    parser.add_argument("--separate", action="store_true", help="Write separate MIDI per track")

    args = parser.parse_args()

    with open(args.notes_json, 'r') as f:
        notes = json.load(f)

    write_midi_from_notes(notes, args.out_path, args.tempo, args.program, args.separate)

if __name__ == "__main__":
    main()
