@echo off
echo JABE Regulatory Processor - Simple Build Script
echo ===============================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install/upgrade required packages
echo Installing required packages...
pip install pyinstaller pillow

REM Clean previous builds
echo Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM Build the executable
echo Building JABE Regulatory Processor executable...
pyinstaller --onefile --windowed --name "JABE_Regulatory_Processor" --icon "static/img/JABE_LOGO_02.png" --add-data "regulatory_processor;regulatory_processor" --add-data "static;static" --hidden-import "regulatory_processor" --hidden-import "PyQt5.QtCore" --hidden-import "PyQt5.QtGui" --hidden-import "PyQt5.QtWidgets" --hidden-import "pandas" --hidden-import "openpyxl" --hidden-import "anthropic" --hidden-import "PyPDF2" --hidden-import "pdfplumber" --hidden-import "nltk" regulatory_gui_branded.py

if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo SUCCESS: Executable built successfully!
echo Location: dist/JABE_Regulatory_Processor.exe
echo.
echo You can now run the executable by double-clicking it.
pause