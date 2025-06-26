# 🚀 Quick Start Guide

Get up and running with JABE Regulatory Document Processor in under 5 minutes!

## 🎯 Choose Your Path

### 👨‍💼 Business User (GUI)
**Perfect for:** First-time users, business analysts, compliance officers

1. **Launch the application:**
   ```bash
   python regulatory_gui.py
   ```

2. **Fill in your information:**
   - Enter your first and last name
   - Select documents/folders to process
   - Choose output directory

3. **Configure (optional):**
   - Enable AI for detailed analysis
   - Set API key for enhanced features

4. **Process:**
   - Click "Validate Inputs" → "Start Processing"
   - Monitor progress in real-time
   - Download results when complete

### 👨‍💻 Power User (Command Line)
**Perfect for:** IT staff, developers, batch processing

**Basic client report:**
```bash
python process_regulations_unified.py
```

**With AI analysis:**
```bash
set ANTHROPIC_API_KEY=your-key-here
python process_regulations_unified.py --user "John Doe"
```

**Technical report:**
```bash
python process_regulations_unified.py --technical
```

## 📁 Document Preparation

### Supported Formats
- ✅ **PDF files** - All regulatory documents
- ✅ **Folders** - Batch process multiple files
- ✅ **Nested directories** - Recursive processing

### File Organization Tips
```
📁 Your Project Folder/
├── 📄 Code Pénal Gabonais.pdf
├── 📁 Instruction_COBAC/
│   ├── 📄 I-2004_01.pdf
│   └── 📄 More instructions...
└── 📁 Reglement_COBAC/
    ├── 📄 R-2018-01.pdf
    └── 📄 More regulations...
```

## ⚡ First Run Checklist

### ✅ Before You Start
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Documents ready in PDF format
- [ ] (Optional) Anthropic API key for AI features

### 🎯 Your First Processing Session

1. **Start small** - Try with 1-2 documents first
2. **Check output** - Verify Excel file generation
3. **Review results** - Open Excel to see compliance analysis
4. **Scale up** - Process full document set

## 📊 Understanding Your Results

### Client-Friendly Excel (Default)
Perfect for business stakeholders:

- **📋 Executive Summary** - Key statistics and priorities
- **📚 Document Overview** - List of all processed regulations
- **⚖️ Compliance Articles** - Detailed requirements with priority levels
- **🎯 Priority Actions** - Action items sorted by urgency
- **📖 Glossary** - Regulatory terms explained

### Priority Levels Explained
- 🔴 **CRITICAL** - Immediate action required (core compliance)
- 🟠 **HIGH** - Address within 30 days (significant impact)
- 🟡 **MEDIUM** - Quarterly review (standard requirements)
- 🟢 **LOW** - Best practices (continuous improvement)

## 🆘 Quick Troubleshooting

### Common First-Run Issues

**GUI won't start:**
```bash
pip install PyQt5
python regulatory_gui.py
```

**"No documents found":**
- Check file paths are correct
- Ensure PDFs are not corrupted
- Verify folder permissions

**AI features not working:**
- Set ANTHROPIC_API_KEY environment variable
- Check internet connection
- Verify API key format (starts with `sk-ant-`)

**Processing takes too long:**
- Start with smaller document sets
- Disable AI for faster processing (`--no-ai`)
- Close other applications

## 🎓 Learning Path

### Week 1: Get Familiar
1. ✅ Complete this quick start
2. 📖 Read [GUI User Guide](./04_GUI_GUIDE.md)
3. 🔍 Explore your first Excel output

### Week 2: Optimize
1. ⚙️ Configure [AI Features](./08_AI_FEATURES.md)
2. 📈 Set up [Batch Processing](./09_BATCH_PROCESSING.md)
3. 🎯 Customize [Configuration](./07_CONFIGURATION.md)

### Ongoing: Master
1. 🏗️ Understand [Architecture](./10_ARCHITECTURE.md)
2. 🚀 Build [Executables](./12_BUILD_GUIDE.md)
3. 🔧 Optimize [Performance](./14_PERFORMANCE.md)

## 🎉 Success Indicators

You're on the right track when you see:

✅ **Green progress bars** in the GUI
✅ **"Processing complete"** messages
✅ **Excel files** generated with content
✅ **Color-coded priorities** in the output
✅ **Compliance articles** properly extracted

## 📞 Need More Help?

- 🐛 **Issues?** → [Troubleshooting Guide](./13_TROUBLESHOOTING.md)
- 🖥️ **GUI Questions?** → [Complete GUI Guide](./04_GUI_GUIDE.md)
- ⌨️ **Command Line?** → [CLI Reference](./05_CLI_GUIDE.md)
- 🏗️ **Architecture?** → [System Overview](./10_ARCHITECTURE.md)

---

*Ready to process your regulatory documents? Let's get started! 🚀*