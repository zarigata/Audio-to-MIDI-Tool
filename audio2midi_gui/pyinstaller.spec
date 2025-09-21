# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models', 'models'),
        ('tests/assets', 'tests/assets'),
    ],
    hiddenimports=[
        'backend.separation',
        'backend.instrument_detect',
        'backend.transcribe',
        'backend.midi_writer',
        'matplotlib.backends.backend_qt5agg',
        'pygame.midi',
        'demucs',
        'spleeter',
        'crepe',
        'pretty_midi',
        'sklearn.neighbors',
        'librosa',
        'torch',
        'tensorflow',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='audio2midi_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='audio2midi_gui',
)
