# 🚀 Which Script Should I Use?

## ✅ RECOMMENDED SCRIPTS

### 1. **GUI Application (Easiest)**
```bash
python regulatory_gui.py
```
**Use when:** You prefer a visual interface with point-and-click controls

**Features:**
- ✅ JABE logo and branding
- ✅ Visual feedback and progress bars
- ✅ Client-friendly and technical export options
- ✅ Settings validation with field highlighting
- ✅ Works on all platforms

---

### 2. **Unified Command Line (Most Flexible)**
```bash
python process_regulations_unified.py
```
**Use when:** You want command-line control with all options

**Examples:**
```bash
# Client-friendly report (default)
python process_regulations_unified.py

# Technical report  
python process_regulations_unified.py --technical

# Both formats
python process_regulations_unified.py --both

# Single file
python process_regulations_unified.py --single "document.pdf"

# With user info
python process_regulations_unified.py --user "John Doe"

# Without AI (faster)
python process_regulations_unified.py --no-ai
```

---

## ❌ LEGACY SCRIPTS (Don't Use)

| Script | Issue | Use Instead |
|--------|-------|-------------|
| `process_regulations_ai.py` | ❌ Hardcoded WSL paths | `process_regulations_unified.py` |
| `process_regulations_ai_windows.py` | ❌ Windows-only, limited | `process_regulations_unified.py` |
| `process_regulations_client.py` | ❌ Client-only, no options | `process_regulations_unified.py --client` |
| `process_regulations.py` | ❌ Basic version, no AI | `process_regulations_unified.py --no-ai` |

---

## 🎯 Quick Decision Guide

**I want the easiest experience:**
```bash
python regulatory_gui.py
```

**I want client-friendly Excel reports:**
```bash
python process_regulations_unified.py
```

**I want technical/detailed reports:**
```bash
python process_regulations_unified.py --technical
```

**I want both report types:**
```bash
python process_regulations_unified.py --both
```

**I want to process just one file:**
```bash
python process_regulations_unified.py --single "path/to/file.pdf"
```

---

## 📋 Output Formats Explained

### Client-Friendly Excel (Default)
- **Executive Summary**: Key priorities and statistics
- **Document Overview**: Simple regulation summaries  
- **Compliance Articles**: Regulation + article + priority + action needed
- **Priority Actions**: Sorted by CRITICAL → HIGH → MEDIUM → LOW
- **Glossary**: Regulatory terms explained

### Technical Excel
- **Document_Metadata**: Technical file information
- **Text_Chunks**: Raw extracted text segments with AI scores
- **Validation_Results**: Detailed AI validation metrics
- **Articles**: Full article extraction with context
- **Statistics**: Processing metrics and analysis

---

## 🔧 Setup Requirements

**For AI Features:**
```bash
set ANTHROPIC_API_KEY=your-api-key-here
```

**For GUI:**
```bash
pip install PyQt5
```

**All dependencies:**
```bash
pip install -r requirements.txt
```