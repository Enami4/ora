# 🖥️ JABE Executable Build Guide

## 📦 Creating Desktop Executable (.exe)

This guide shows how to create a standalone executable file for the JABE Regulatory Document Processor that can be run on any Windows computer without Python installation.

## 🚀 Quick Build (Recommended)

### Option 1: Simple Batch Script (Windows)
```bash
# Double-click this file or run in command prompt
build_simple.bat
```

### Option 2: Shell Script (Linux/WSL)
```bash
# Run in terminal
./build_simple.sh
```

### Option 3: Python Build Script (Advanced)
```bash
python build_executable.py
```

## 📋 Prerequisites

### Required Software
- **Python 3.8+** installed and in PATH
- **pip** package manager
- **Git** (if cloning repository)

### Install Dependencies
```bash
# Install all requirements including PyInstaller
pip install -r requirements.txt

# Or install build tools only
pip install pyinstaller pillow pywin32
```

## 🔨 Build Process Steps

### Step 1: Prepare Environment
```bash
cd /mnt/c/Users/doupa/Desktop/Ventures/Orabank

# Verify Python installation
python --version

# Install build dependencies
pip install pyinstaller pillow
```

### Step 2: Build Executable
```bash
# Method 1: Simple build (fastest)
build_simple.bat

# Method 2: Advanced build with packaging
python build_executable.py

# Method 3: Manual PyInstaller command
pyinstaller JABE_Regulatory_Processor.spec
```

### Step 3: Test Executable
```bash
# Run the generated executable
dist/JABE_Regulatory_Processor.exe

# Or test with build script
python build_executable.py --test
```

### Step 4: Create Desktop Shortcut
```bash
# Create desktop shortcut automatically
python create_desktop_shortcut.py
```

## 📁 Output Structure

After successful build:
```
/Orabank/
├── dist/
│   └── JABE_Regulatory_Processor.exe    # ← Main executable file
├── build/                               # Build artifacts (can be deleted)
├── releases/                           # Distribution packages
│   ├── JABE_Regulatory_Processor_v2.0_20241218_143022/
│   │   ├── JABE_Regulatory_Processor.exe
│   │   └── README.txt
│   └── JABE_Regulatory_Processor_v2.0_20241218_143022.zip
└── JABE_Regulatory_Processor.spec       # PyInstaller configuration
```

## 📊 File Sizes (Approximate)

| Component | Size |
|-----------|------|
| Executable (.exe) | 150-250 MB |
| Distribution ZIP | 80-120 MB |
| Source code | 5-10 MB |

**Note**: Large size due to bundled Python interpreter, PyQt5, and AI libraries.

## 🎯 Distribution Options

### Option 1: Single Executable File
- **File**: `JABE_Regulatory_Processor.exe`
- **Pros**: Single file, easy to share
- **Cons**: Large file size (150-250 MB)
- **Use case**: Internal distribution

### Option 2: Distribution Package
- **File**: `JABE_Regulatory_Processor_v2.0_YYYYMMDD_HHMMSS.zip`
- **Contents**: Executable + README
- **Pros**: Professional packaging
- **Cons**: Multiple files
- **Use case**: Client delivery

### Option 3: Installer (Advanced)
- **Tool**: NSIS, Inno Setup, or WiX
- **Pros**: Professional installation
- **Cons**: Complex setup
- **Use case**: Enterprise deployment

## 🖥️ Running the Executable

### On Your Computer
```bash
# Navigate to dist folder
cd dist

# Run executable
./JABE_Regulatory_Processor.exe

# Or double-click in Windows Explorer
```

### On Other Computers
1. **Copy** `JABE_Regulatory_Processor.exe` to target computer
2. **Double-click** to run (no Python installation needed)
3. **First run** may take 10-30 seconds to extract bundled files
4. **Antivirus** may scan the file (normal behavior)

## 🛠️ Troubleshooting

### Build Issues

#### "PyInstaller not found"
```bash
pip install pyinstaller
```

#### "Missing module" errors
```bash
# Install missing packages
pip install package_name

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

#### "Icon file not found"
- Ensure `static/img/JABE_LOGO_02.png` exists
- Or remove `--icon` parameter from build command

#### Large executable size
- Normal for PyQt5 + AI libraries
- Consider using `--exclude-module` for unused packages
- Use UPX compression (included in build script)

### Runtime Issues

#### "Application failed to start"
- **Antivirus blocking**: Add exception for the executable
- **Missing DLLs**: Run on computer with Visual C++ Redistributable
- **Path issues**: Run from folder containing the executable

#### "API key not found" 
- Set environment variable: `ANTHROPIC_API_KEY`
- Or enter API key in the application settings

#### "Slow startup"
- First run extracts bundled files (normal)
- Subsequent runs should be faster
- Consider using `--onedir` instead of `--onefile` for faster startup

### Performance Optimization

#### Faster Startup
```bash
# Use --onedir mode (creates folder instead of single file)
pyinstaller --onedir --windowed regulatory_gui_branded.py
```

#### Smaller Size
```bash
# Exclude unnecessary modules
pyinstaller --exclude-module matplotlib --exclude-module scipy regulatory_gui_branded.py
```

## 🚀 Advanced Build Options

### Custom Icon
```bash
# Convert PNG to ICO first
python -c "from PIL import Image; Image.open('logo.png').save('logo.ico')"

# Use in build
pyinstaller --icon logo.ico regulatory_gui_branded.py
```

### Debug Mode
```bash
# Build with console for debugging
pyinstaller --onefile --console regulatory_gui_branded.py
```

### Optimized Build
```bash
# Use spec file for fine control
pyinstaller JABE_Regulatory_Processor.spec
```

## 📦 Deployment Checklist

### Before Distribution
- [ ] Test executable on clean Windows system
- [ ] Verify all features work without Python
- [ ] Check file size is acceptable
- [ ] Test with sample documents
- [ ] Verify logo and branding display correctly

### For Client Delivery
- [ ] Create distribution ZIP package
- [ ] Include README with instructions
- [ ] Test on client's typical system
- [ ] Provide support documentation
- [ ] Include version information

### For Internal Use
- [ ] Create desktop shortcuts for users
- [ ] Add to company software catalog
- [ ] Document installation process
- [ ] Train users on new interface

## 🔄 Build Automation

### Continuous Integration
```yaml
# Example GitHub Actions workflow
name: Build Executable
on: [push]
jobs:
  build:
    runs-on: windows-latest
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - run: pip install -r requirements.txt
    - run: python build_executable.py
    - uses: actions/upload-artifact@v2
      with:
        name: JABE-Executable
        path: dist/
```

### Batch Building
```bash
# Build multiple versions
for config in basic ai-enhanced; do
    python build_executable.py --config $config
done
```

## 📈 Version Management

### File Naming Convention
```
JABE_Regulatory_Processor_v{version}_{timestamp}.{ext}

Examples:
- JABE_Regulatory_Processor_v2.0_20241218_143022.exe
- JABE_Regulatory_Processor_v2.0_20241218_143022.zip
```

### Version Information
```python
# Add to build script
version_info = {
    'version': '2.0.0',
    'description': 'JABE Regulatory Document Processor',
    'copyright': '© 2024 JABE. All rights reserved.',
    'company': 'JABE'
}
```

## 🎯 Best Practices

### Build Environment
- Use **dedicated build machine** for consistency
- **Version control** all build scripts
- **Test builds** on multiple Windows versions
- **Document build process** for team members

### Distribution
- **Virus scan** executables before distribution
- **Digital signing** for enterprise deployment
- **Version tracking** for support purposes
- **User documentation** for installation

### Security
- **Code signing** certificate for trusted execution
- **Antivirus whitelist** submissions if needed
- **Network security** considerations for AI features
- **Data privacy** compliance for document processing

The executable provides a complete, self-contained solution that can be easily distributed and run on any Windows computer without requiring Python or technical setup.