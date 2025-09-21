# gui.py
# Main GUI implementation

import sys
import os
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QProgressBar, QTextEdit, QComboBox, QSlider, QGroupBox,
    QCheckBox, QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QListWidget,
    QListWidgetItem, QSplitter, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QKeySequence, QShortcut, QAction
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import librosa
import pygame.midi
from backend.separation import separate
from backend.instrument_detect import analyze_stem, choose_transcription_model
from backend.transcribe import transcribe_stem_to_midi
from backend.midi_writer import write_midi_from_notes

SETTINGS_FILE = Path.home() / "audio2midi_settings.json"

class WorkerThread(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(object)

    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        try:
            result = self.func(*self.args)
            self.finished.emit(result)
        except Exception as e:
            self.log.emit(f"Error: {e}")

class Audio2MIDIGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = self.load_settings()
        self.audio_path = None
        self.stems = []
        self.midis = {}
        self.setAcceptDrops(True)
        self.init_ui()
        self.init_midi()

    # ... existing code ...

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if file_path.lower().endswith(('.mp3', '.wav', '.flac')):
            self.audio_path = file_path
            self.file_label.setText(Path(file_path).name)
            self.plot_waveform()
        event.accept()

    def init_ui(self):
        self.setWindowTitle("Audio2MIDI GUI")
        self.setGeometry(100, 100, 1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Menu
        self.create_menu()

        # File picker
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.file_label)
        open_btn = QPushButton("Open File")
        open_btn.clicked.connect(self.open_file)
        file_layout.addWidget(open_btn)
        layout.addLayout(file_layout)

        # Waveform
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout(options_group)
        self.sep_combo = QComboBox()
        self.sep_combo.addItems(["Demucs", "Spleeter"])
        options_layout.addWidget(QLabel("Separation:"))
        options_layout.addWidget(self.sep_combo)

        self.trans_combo = QComboBox()
        self.trans_combo.addItems(["Auto", "OnsetsFrames", "CREPE", "MT3"])
        options_layout.addWidget(QLabel("Transcription:"))
        options_layout.addWidget(self.trans_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        options_layout.addWidget(QLabel("Device:"))
        options_layout.addWidget(self.device_combo)

        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["none", "8th", "16th"])
        options_layout.addWidget(QLabel("Quantize:"))
        options_layout.addWidget(self.quant_combo)

        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(0, 100)
        options_layout.addWidget(QLabel("Quality:"))
        options_layout.addWidget(self.quality_slider)

        layout.addWidget(options_group)

        # Buttons
        btn_layout = QHBoxLayout()
        sep_btn = QPushButton("Separate Stems")
        sep_btn.clicked.connect(self.separate_stems)
        btn_layout.addWidget(sep_btn)

        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self.analyze_stems)
        btn_layout.addWidget(analyze_btn)

        layout.addLayout(btn_layout)

        # Stems list
        self.stems_list = QListWidget()
        layout.addWidget(self.stems_list)

        # Progress and log
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        layout.addWidget(self.log_text)

        # Shortcuts
        QShortcut(QKeySequence.Open, self, self.open_file)
        QShortcut(QKeySequence.Save, self, self.export_midi)
        QShortcut(QKeySequence.Cancel, self, self.cancel_job)

    def create_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio Files (*.mp3 *.wav *.flac)")
        if path:
            self.audio_path = path
            self.file_label.setText(Path(path).name)
            self.plot_waveform()

    def plot_waveform(self):
        if self.audio_path:
            audio, sr = librosa.load(self.audio_path, sr=None)
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(audio)
            ax.set_title("Waveform")
            self.canvas.draw()

    def separate_stems(self):
        if not self.audio_path:
            return
        out_dir = Path(self.audio_path).parent / "stems"
        device = self.device_combo.currentText()
        self.run_worker(separate, self.audio_path, str(out_dir), device=device)

    def analyze_stems(self):
        # Placeholder
        self.log("Analyzing stems...")

    def run_worker(self, func, *args):
        self.worker = WorkerThread(func, *args)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self, result):
        if isinstance(result, list):  # Summary from separation
            self.stems = result
            self.update_stems_list()

    def update_stems_list(self):
        self.stems_list.clear()
        for stem in self.stems:
            item = QListWidgetItem(f"{Path(stem['path']).name}")
            self.stems_list.addItem(item)

    def export_midi(self):
        # Placeholder
        pass

    def cancel_job(self):
        if hasattr(self, 'worker'):
            self.worker.terminate()

    def log(self, msg):
        self.log_text.append(msg)

    def init_midi(self):
        try:
            pygame.midi.init()
            self.midi_out = pygame.midi.Output(pygame.midi.get_default_output_id())
        except:
            self.midi_out = None

    def show_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.settings = dialog.get_settings()
            self.save_settings()

    def load_settings(self):
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        return {"model_cache": "models", "gpu": False}

    def save_settings(self):
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(self.settings, f)

class SettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QFormLayout(self)
        self.cache_edit = QLineEdit(settings.get("model_cache", "models"))
        layout.addRow("Model Cache:", self.cache_edit)
        self.gpu_check = QCheckBox()
        self.gpu_check.setChecked(settings.get("gpu", False))
        layout.addRow("Prefer GPU:", self.gpu_check)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self):
        return {"model_cache": self.cache_edit.text(), "gpu": self.gpu_check.isChecked()}
