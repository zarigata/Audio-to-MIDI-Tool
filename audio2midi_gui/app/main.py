"""
Audio to MIDI GUI Application

This is the main entry point for the GUI application.
"""

import sys
from PySide6.QtWidgets import QApplication
from gui import Audio2MIDIGUI

def main():
    app = QApplication(sys.argv)
    window = Audio2MIDIGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
