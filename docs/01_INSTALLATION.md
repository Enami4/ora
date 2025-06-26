# 🔧 Installation & Setup

## 📋 Requirements

- **Python 3.8+** (3.9+ recommended)
- **Windows 10+**, **macOS 10.14+**, or **Linux Ubuntu 18.04+**
- **4GB RAM minimum** (8GB+ recommended for AI features)
- **Internet connection** (for AI features)

## ⚡ Quick Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python regulatory_gui.py
```

If the GUI opens successfully, you're ready to go! 🎉

## 🔑 API Key Setup (For AI Features)

### Get Anthropic API Key
1. Visit [https://console.anthropic.com](https://console.anthropic.com)
2. Create account and generate API key
3. Set environment variable:

**Windows:**
```cmd
set ANTHROPIC_API_KEY=your-api-key-here
```

**macOS/Linux:**
```bash
export ANTHROPIC_API_KEY=your-api-key-here
```

### Alternative: Enter in GUI
You can also enter your API key directly in the application's Configuration tab.

## 📦 Package Dependencies

### Core Requirements
- `PyPDF2>=3.0.0` - PDF text extraction
- `pdfplumber>=0.9.0` - Advanced PDF parsing
- `pandas>=1.5.0` - Data manipulation
- `openpyxl>=3.1.0` - Excel file generation
- `anthropic>=0.8.0` - AI validation (optional)

### GUI Requirements
- `PyQt5>=5.15.0` - Graphical interface

### Optional Dependencies
- `nltk>=3.8` - Advanced text processing
- `python-dotenv>=1.0.0` - Environment file support

## 🔧 Advanced Setup

### Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv regulatory_env

# Activate (Windows)
regulatory_env\Scripts\activate

# Activate (macOS/Linux)
source regulatory_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Development Setup
```bash
# Clone with tests
pip install -r requirements.txt
pip install pytest pytest-qt  # For running tests

# Run tests
pytest tests/
```

## 🐛 Troubleshooting Installation

### Common Issues

**"No module named 'PyQt5'"**
```bash
pip install PyQt5>=5.15.0
```

**"Permission denied" on Windows**
```bash
# Run as administrator or use:
pip install --user -r requirements.txt
```

**"SSL Certificate errors"**
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org -r requirements.txt
```

**macOS PyQt5 Issues**
```bash
# If PyQt5 fails on macOS:
brew install pyqt5
pip install PyQt5
```

### Verify Installation
```bash
python -c "from regulatory_processor import RegulatoryDocumentProcessor; print('✅ Core modules OK')"
python -c "from PyQt5.QtWidgets import QApplication; print('✅ GUI modules OK')"
```

## 🚀 Next Steps

✅ Installation complete? → [Quick Start Guide](./02_QUICK_START.md)

🖥️ Want to use the GUI? → [GUI User Guide](./04_GUI_GUIDE.md)

⌨️ Prefer command line? → [Command Line Guide](./05_CLI_GUIDE.md)