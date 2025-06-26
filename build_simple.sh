#!/bin/bash

echo "JABE Regulatory Processor - Simple Build Script"
echo "==============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed or not in PATH"
    exit 1
fi

# Install/upgrade required packages
echo "Installing required packages..."
pip3 install pyinstaller pillow

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist build

# Build the executable
echo "Building JABE Regulatory Processor executable..."
pyinstaller \
    --onefile \
    --windowed \
    --name "JABE_Regulatory_Processor" \
    --icon "static/img/JABE_LOGO_02.png" \
    --add-data "regulatory_processor:regulatory_processor" \
    --add-data "static:static" \
    --hidden-import "regulatory_processor" \
    --hidden-import "PyQt5.QtCore" \
    --hidden-import "PyQt5.QtGui" \
    --hidden-import "PyQt5.QtWidgets" \
    --hidden-import "pandas" \
    --hidden-import "openpyxl" \
    --hidden-import "anthropic" \
    --hidden-import "PyPDF2" \
    --hidden-import "pdfplumber" \
    --hidden-import "nltk" \
    regulatory_gui_branded.py

if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS: Executable built successfully!"
    echo "Location: dist/JABE_Regulatory_Processor"
    echo ""
    echo "You can now run the executable directly."
else
    echo "ERROR: Build failed"
    exit 1
fi