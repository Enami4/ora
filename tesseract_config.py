"""Configuration for Tesseract OCR."""

import os
import pytesseract

# Windows default installation path
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Check if Tesseract is installed
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"Tesseract found at: {TESSERACT_PATH}")
else:
    print(f"Tesseract not found at: {TESSERACT_PATH}")
    print("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("Or update TESSERACT_PATH in this file to point to your installation.")