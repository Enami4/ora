# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

# Get the base directory
base_dir = Path(SPECPATH)

# Add the source directory to Python path
sys.path.insert(0, str(base_dir))

# Import to get hidden imports
try:
    import regulatory_processor
    import PyQt5
    import pandas
    import openpyxl
    import nltk
    import anthropic
    import PyPDF2
    import pdfplumber
except ImportError as e:
    print(f"Warning: Could not import {e.name}")

block_cipher = None

# Define data files to include
datas = [
    # Include the entire regulatory_processor module
    ('regulatory_processor', 'regulatory_processor'),
    # Include static resources (logo, etc.)
    ('static', 'static'),
    # Include NLTK data if present
]

# Try to include NLTK data
try:
    import nltk
    nltk_data_path = nltk.data.find('tokenizers/punkt')
    if os.path.exists(nltk_data_path):
        datas.append((nltk_data_path, 'nltk_data/tokenizers/punkt'))
except:
    pass

# Hidden imports - modules that PyInstaller might miss
hiddenimports = [
    # Core modules
    'regulatory_processor',
    'regulatory_processor.config',
    'regulatory_processor.extractors',
    'regulatory_processor.chunkers',
    'regulatory_processor.validators',
    'regulatory_processor.exporters',
    'regulatory_processor.utils',
    'regulatory_processor.processor',
    
    # PyQt5 modules
    'PyQt5.QtCore',
    'PyQt5.QtGui', 
    'PyQt5.QtWidgets',
    'PyQt5.QtPrintSupport',
    
    # Data processing
    'pandas',
    'openpyxl',
    'openpyxl.workbook',
    'openpyxl.styles',
    'openpyxl.utils',
    'xlsxwriter',
    
    # PDF processing
    'PyPDF2',
    'pdfplumber',
    'pdfplumber.pdf',
    
    # AI and NLP
    'anthropic',
    'nltk',
    'nltk.tokenize',
    'nltk.data',
    
    # Other dependencies
    'datetime',
    'json',
    'logging',
    'hashlib',
    'pathlib',
    'traceback',
    're',
    'os',
    'sys',
    'warnings',
    
    # Additional PyQt5 modules that might be needed
    'PyQt5.sip',
    'PyQt5.QtNetwork',
    
    # Pandas dependencies
    'pandas._libs',
    'pandas._libs.tslibs',
    'pandas.io',
    'pandas.io.formats',
    'pandas.io.formats.style',
    
    # OpenPyXL dependencies
    'openpyxl.xml',
    'openpyxl.chart',
    'openpyxl.drawing',
    'openpyxl.formula',
    
    # Additional dependencies
    'pkg_resources',
    'packaging',
    'packaging.version',
    'packaging.specifiers',
    'packaging.requirements',
]

# Analysis configuration
a = Analysis(
    ['regulatory_gui_branded.py'],
    pathex=[str(base_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'matplotlib',
        'scipy',
        'numpy.distutils',
        'tkinter',
        'unittest',
        'test',
        'pytest',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate entries
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='JABE_Regulatory_Processor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False for GUI app (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='static/img/JABE_LOGO_02.png',  # App icon
    version_file=None,
)

# Create application bundle (for macOS, optional for Windows)
app = BUNDLE(
    exe,
    name='JABE_Regulatory_Processor.app',
    icon='static/img/JABE_LOGO_02.png',
    bundle_identifier='com.jabe.regulatory-processor',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleDisplayName': 'JABE Regulatory Processor',
        'CFBundleVersion': '2.0.0',
        'CFBundleShortVersionString': '2.0.0',
        'NSHumanReadableCopyright': 'Copyright © 2024 JABE. All rights reserved.',
    },
)