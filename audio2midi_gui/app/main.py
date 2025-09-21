"""
Audio to MIDI GUI Application

This is the main entry point for the GUI application.
"""

import sys
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout


def main():
    print("audio2midi ready")
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Audio2MIDI GUI")
    layout = QVBoxLayout()
    label = QLabel("Audio2MIDI GUI Ready")
    layout.addWidget(label)
    window.setLayout(layout)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
