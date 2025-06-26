# Regulatory Document Processor - Architecture Guide

## 🏗️ System Overview

The processor follows a modular pipeline architecture with optional AI enhancement:

```
PDF Input → Extraction → Chunking → AI Validation → Article Extraction → Excel Export
```

## 📋 Processing Pipeline

### 1. **Document Intake & Validation**
```
📁 Input: PDF files from directories
├── File validation (size, format, readability)
├── Document type identification (INSTRUCTION, REGLEMENT, CODE_PENAL)
└── Metadata extraction (pages, creation date, file hash)
```

### 2. **Text Extraction (Multi-Method)**
```
📄 PDF Processing
├── Primary: pdfplumber extraction
├── Fallback: PyPDF2 extraction
├── Page-by-page text capture
└── Error handling & logging
```

### 3. **Text Processing & Chunking**
```
📝 Text Processing
├── Text cleaning (encoding, formatting)
├── Semantic chunking (sections, sentences)
├── Chunk overlap management
└── Metadata preservation per chunk
```

### 4. **AI Validation Chain** (Optional)
```
🤖 AI Analysis
├── Multi-dimensional scoring
├── Article extraction
├── Materiality assessment
└── Quality validation
```

### 5. **Export & Reporting**
```
📊 Excel Export
├── Multiple worksheets
├── Validation scores
├── Article breakdown
└── Comprehensive statistics
```

## 🔧 Component Details

### Configuration (`config.py`)
- **Purpose**: Centralized settings management
- **Key Settings**:
  - Chunk size/overlap
  - AI validation toggle
  - API credentials
  - Export options

### Extractors (`extractors.py`)
- **Purpose**: PDF text extraction with fallbacks
- **Methods**:
  - `pdfplumber` for complex layouts
  - `PyPDF2` for simple documents
  - Metadata extraction
  - Validation of extracted content

### Chunkers (`chunkers.py`)
- **Purpose**: Intelligent text segmentation
- **Strategies**:
  - Sentence-based chunking
  - Section-aware splitting
  - Semantic boundary detection
  - Overlap management

### Validators (`validators.py`)
- **Purpose**: AI-powered quality assessment
- **Features**:
  - Multi-dimensional scoring
  - Article extraction
  - Materiality assessment
  - Prompt engineering templates

### Exporters (`exporters.py`)
- **Purpose**: Structured data export
- **Outputs**:
  - Excel with multiple sheets
  - CSV for simple data
  - Formatted reports
  - Validation summaries

## 🚀 Usage Walkthrough

### Step 1: Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (for AI features)
export ANTHROPIC_API_KEY="your-key-here"
```

### Step 2: Basic Configuration
```python
from regulatory_processor import ProcessorConfig

config = ProcessorConfig(
    chunk_size=1000,          # Text chunk size
    chunk_overlap=200,        # Overlap between chunks
    enable_ai_validation=True, # AI validation
    extract_articles=True     # Article extraction
)
```

### Step 3: Initialize Processor
```python
from regulatory_processor import RegulatoryDocumentProcessor

processor = RegulatoryDocumentProcessor(config)
```

### Step 4: Process Documents
```python
# Single document
result = processor.process_document("document.pdf")

# Entire directory
results = processor.process_directory("regulations/")
```

### Step 5: Export Results
```python
processor.export_results(
    "analysis.xlsx",
    include_validation=True,
    include_articles=True
)
```

## 📊 Data Flow Details

### Input Processing
1. **File Discovery**: Scan directories for PDF files
2. **Validation**: Check file integrity and format
3. **Type Detection**: Identify document category
4. **Metadata Extraction**: Gather file information

### Text Processing
1. **Extraction**: Pull text from PDF pages
2. **Cleaning**: Remove artifacts and normalize
3. **Chunking**: Split into manageable segments
4. **Annotation**: Add metadata to each chunk

### AI Enhancement (Optional)
1. **Validation Scoring**: Assess quality dimensions
2. **Article Detection**: Find regulatory articles
3. **Materiality Assessment**: Evaluate importance
4. **Reasoning Chain**: Validate results

### Export Generation
1. **Data Aggregation**: Combine all results
2. **Sheet Creation**: Generate Excel worksheets
3. **Formatting**: Apply styling and structure
4. **Validation**: Ensure data integrity

## 🎯 AI Validation Deep Dive

### Multi-Dimensional Scoring

#### Completeness Score (0-100)
- **Evaluates**: Chunk boundary integrity
- **Checks**: 
  - Sentences not cut mid-way
  - Regulatory references intact
  - Semantic coherence maintained

#### Reliability Score (0-100)
- **Evaluates**: Text quality and accuracy
- **Checks**:
  - OCR error detection
  - Encoding issues
  - Formatting preservation

#### Legal Structure Score (0-100)
- **Evaluates**: Regulatory compliance
- **Checks**:
  - Article numbering consistency
  - Hierarchical structure
  - Legal language appropriateness

### Article Extraction Process

1. **Regex Detection**: Find article patterns
2. **AI Enhancement**: Refine and validate
3. **Content Extraction**: Pull complete text
4. **Materiality Analysis**: Assess importance

### Materiality Assessment Criteria

#### LOW Materiality
- Administrative procedures
- Minor operational details
- Informational content

#### MEDIUM Materiality
- Standard compliance requirements
- Operational guidelines
- Routine reporting

#### HIGH Materiality
- Critical compliance requirements
- Risk management provisions
- Significant penalties

#### CRITICAL Materiality
- Systemic risk factors
- Fundamental banking operations
- Severe regulatory consequences

## 📈 Output Structure

### Excel Worksheets Generated

1. **Document_Metadata**
   - File information
   - Processing timestamps
   - Page counts and statistics

2. **Articles**
   - Context (COBAC, Code Pénal, etc.)
   - Regulation name
   - Article numbers
   - Full content
   - Materiality assessment

3. **Text_Chunks**
   - Segmented text
   - Validation scores
   - Chunk metadata

4. **Validation_Results**
   - AI quality scores
   - Processing metrics
   - Issue identification

5. **Statistics**
   - Processing summary
   - Document type breakdown
   - Validation statistics

## 🔄 Error Handling

### Graceful Degradation
- AI unavailable → Rules-based validation
- Extraction fails → Log and continue
- API limits → Batch processing

### Logging & Monitoring
- Detailed processing logs
- Error tracking
- Performance metrics
- Validation reports

## 🛠️ Customization Points

### Extending Validation
```python
# Add custom validation criteria
class CustomValidator(AIValidator):
    def custom_validation(self, text):
        # Your validation logic
        pass
```

### Custom Article Patterns
```python
# Add new article detection patterns
patterns = [
    r'Custom\s+(\d+)\s*[:\-.]?\s*([^\n]*)',
    # Your patterns here
]
```

### Export Customization
```python
# Customize Excel output
class CustomExporter(ExcelExporter):
    def _custom_sheet(self, data, writer):
        # Your export logic
        pass
```

## 🎯 Best Practices

### Performance Optimization
- Process in batches for large datasets
- Use appropriate chunk sizes
- Monitor API usage for AI features

### Quality Assurance
- Review validation scores
- Check materiality assessments
- Validate article extraction

### Data Management
- Organize output files by date
- Maintain processing logs
- Archive original documents