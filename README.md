# 🏛️ JABE Regulatory Document Processor

**AI-Powered Compliance Analysis for COBAC Regulations**

Transform complex regulatory documents into actionable compliance insights with our intelligent document processor. Built specifically for Central African banking regulations (COBAC/CEMAC), this tool delivers both technical analysis and business-friendly reports.

## 🚀 Quick Start

### Option 1: GUI Application (Recommended)
```bash
python regulatory_gui.py
```
*Perfect for business users - point, click, and get results!*

### Option 2: Command Line (Power Users)
```bash
python process_regulations_unified.py
```
*Full control with flexible options for IT teams*

## ✨ Key Features

- 🤖 **AI-Powered Analysis** - Claude 3.5 Sonnet for intelligent document understanding
- 📊 **Dual Output Formats** - Client-friendly reports + technical analysis
- 🎯 **Priority Classification** - CRITICAL/HIGH/MEDIUM/LOW compliance priorities
- 🏛️ **COBAC Specialized** - Built for Central African banking regulations
- 🖥️ **Modern GUI** - JABE-branded interface with real-time progress
- ⚡ **Batch Processing** - Handle multiple documents efficiently
- 📈 **Excel Reports** - Professional, color-coded compliance dashboards

## 📚 Documentation

**🎯 New User?** Start here: **[📖 Complete Documentation](./docs/README.md)**

Quick Links:
- 🔧 **[Installation Guide](./docs/01_INSTALLATION.md)** - Get set up in 5 minutes
- 🚀 **[Quick Start](./docs/02_QUICK_START.md)** - Your first processing session
- 🖥️ **[GUI Guide](./docs/04_GUI_GUIDE.md)** - Visual interface walkthrough
- ⌨️ **[CLI Guide](./docs/05_CLI_GUIDE.md)** - Command-line power user options

## Module Structure

```
regulatory_processor/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── extractors.py        # PDF text extraction
├── chunkers.py          # Text chunking algorithms
├── validators.py        # AI validation and materiality assessment
├── exporters.py         # Excel/CSV export with validation results
├── utils.py             # Utility functions
└── processor.py         # Main processor class
```

## Configuration

```python
from regulatory_processor import ProcessorConfig

config = ProcessorConfig(
    chunk_size=1000,           # Characters per chunk
    chunk_overlap=200,         # Overlap between chunks
    max_file_size_mb=100,      # Maximum PDF size
    log_level="INFO",          # Logging level
    clean_text=True,           # Clean extracted text
    
    # AI Features
    enable_ai_validation=True, # Enable AI validation
    extract_articles=True,     # Extract regulatory articles
    assess_materiality=True,   # Assess article materiality
    anthropic_api_key="sk-...", # Your Anthropic API key
    ai_model="claude-3-haiku-20240307"
)

processor = RegulatoryDocumentProcessor(config)
```

## Usage Examples

### Basic Processing
```bash
python process_regulations.py
```

### AI-Enhanced Processing
```bash
# Set your API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Run with AI validation
python process_regulations_ai.py
```

### Process Single File with AI
```bash
python process_regulations_ai.py --single "path/to/file.pdf"
```

## Output Format

The Excel output contains:
- **Document_Metadata**: File information, extraction dates, page counts
- **Articles**: Extracted articles with context, regulation name, article numbers, content, and materiality assessment
- **Text_Chunks**: Segmented text with validation scores (completeness, reliability, legal structure)
- **Validation_Results**: AI validation scores and quality metrics per document
- **Full_Texts**: Complete document text (optional)
- **Statistics**: Processing metrics and document type counts
- **Summary**: Overall processing summary

## AI Validation Scores

### Multi-Dimensional Scoring (0-100):
- **Completeness Score**: Chunk boundary integrity, sentence completeness
- **Reliability Score**: OCR accuracy, formatting preservation, encoding quality  
- **Legal Structure Score**: Regulatory format compliance, section coherence
- **Overall Validation Score**: Composite quality metric

### Materiality Assessment:
- **LOW**: Administrative or procedural content
- **MEDIUM**: Standard compliance requirements  
- **HIGH**: Critical compliance or risk management
- **CRITICAL**: Fundamental to banking operations or systemic risk

## Error Handling

Failed extractions are logged and reported in:
- Log file: `regulatory_processor.log`
- Error report: `{output_name}_errors.txt`