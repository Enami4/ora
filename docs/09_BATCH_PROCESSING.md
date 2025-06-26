# Batch Processing & Database Building Guide

## 🗂️ How Multiple Documents Are Processed

The tool automatically processes **ALL** documents in your folders and combines them into a **single comprehensive Excel database**.

### 📁 Current Folder Structure (Your Setup)
```
/mnt/c/Users/doupa/Desktop/Ventures/Orabank/
├── Code Pénal Gabonais.pdf                    (1 document)
├── Instruction_COBAC/                          (11 documents)
│   ├── I-2004_01.pdf
│   ├── INSTRUCTION-COBAC-I-2006-01-ASTROLAB.pdf
│   ├── INSTRUCTION-COBAC-I-2008-01.pdf
│   └── ... (8 more)
└── Reglement_COBAC/                           (60+ documents)
    ├── R-2001-07.pdf
    ├── R-2018-01.pdf
    ├── REGLEMENT-N-04-18-CEMAC-UMAC-COBAC.pdf
    └── ... (57+ more)
```

**Total: ~72 documents → 1 comprehensive Excel database**

## 🔄 Automatic Batch Processing

### When you run:
```bash
python process_regulations_ai.py
```

### The tool automatically:
1. **Scans all folders** recursively
2. **Finds all PDF files** (72 documents in your case)
3. **Processes each document** individually
4. **Appends results** to the same Excel file
5. **Creates a unified database** with all documents

### 📊 Single Excel Output Contains ALL Documents

The generated Excel file (e.g., `regulatory_ai_processed_20241218_143022.xlsx`) contains:

#### **Articles Sheet** - All Articles from All Documents
| Context | Regulation Name | Article Number | Content | Materiality |
|---------|----------------|----------------|---------|-------------|
| CODE_PENAL | Code Pénal Gabonais | Article 1 | Les infractions... | HIGH |
| CODE_PENAL | Code Pénal Gabonais | Article 2 | Les sanctions... | CRITICAL |
| INSTRUCTION | I-2004_01 | Article 1 | Les établissements... | MEDIUM |
| INSTRUCTION | I-2006-01-ASTROLAB | Article 1 | Les procédures... | HIGH |
| REGLEMENT | R-2001-07 | Article 1 | Capital minimum... | HIGH |
| REGLEMENT | R-2018-01 | Article 1 | Ratios prudentiels... | CRITICAL |
| ... | ... | ... | ... | ... |

#### **Document_Metadata Sheet** - All 72+ Documents
| File Name | Document Type | Page Count | Total Words | Total Chunks |
|-----------|---------------|------------|-------------|--------------|
| Code Pénal Gabonais.pdf | CODE_PENAL | 156 | 45,231 | 87 |
| I-2004_01.pdf | INSTRUCTION | 12 | 3,456 | 8 |
| R-2018-01.pdf | REGLEMENT | 28 | 8,923 | 19 |
| ... | ... | ... | ... | ... |

#### **Text_Chunks Sheet** - All Chunks from All Documents
| File Name | Chunk Index | Text | Validation Score |
|-----------|-------------|------|------------------|
| Code Pénal Gabonais.pdf | 1 | Article 1. Les infractions... | 87.5 |
| Code Pénal Gabonais.pdf | 2 | Les sanctions applicables... | 92.1 |
| I-2004_01.pdf | 1 | Les établissements de crédit... | 85.3 |
| ... | ... | ... | ... |

## 🎯 Processing Examples

### Example 1: Process Everything (Default)
```bash
python process_regulations_ai.py
```
**Result**: 1 Excel file with all 72+ documents combined

### Example 2: Process Specific Folder Only
```python
from regulatory_processor import RegulatoryDocumentProcessor, ProcessorConfig

config = ProcessorConfig(enable_ai_validation=True, extract_articles=True)
processor = RegulatoryDocumentProcessor(config)

# Process only COBAC Regulations
processor.process_directory("Reglement_COBAC/")
processor.export_results("cobac_regulations_only.xlsx")
```
**Result**: 1 Excel file with ~60 COBAC regulations

### Example 3: Process Multiple Folders Separately
```python
processor = RegulatoryDocumentProcessor(config)

# Process Instructions
processor.process_directory("Instruction_COBAC/")
processor.export_results("instructions_database.xlsx")

# Clear and process Regulations  
processor.processed_documents.clear()
processor.process_directory("Reglement_COBAC/")
processor.export_results("regulations_database.xlsx")
```
**Result**: 2 separate Excel files

### Example 4: Incremental Processing (Add New Documents)
```python
# Process existing documents
processor = RegulatoryDocumentProcessor(config)
processor.process_directory("Reglement_COBAC/")

# Add new documents from another folder
processor.process_directory("New_Regulations_2024/")

# Export combined database
processor.export_results("complete_database.xlsx")
```
**Result**: 1 Excel file with old + new documents

## 📈 Database Building Statistics

### Your Current Setup Will Generate:

#### Estimated Totals (72 documents):
- **Articles**: 800-1,500 regulatory articles
- **Text Chunks**: 3,000-6,000 text segments  
- **Metadata Records**: 72 document records
- **Validation Scores**: 72 document scores + chunk scores
- **Processing Time**: 1-3 hours (with AI validation)

#### Excel File Size:
- **Basic Processing**: 15-25 MB
- **AI Processing**: 25-40 MB (with validation data)

## 🔧 Advanced Batch Configuration

### Custom Batch Processing
```python
from regulatory_processor import RegulatoryDocumentProcessor, ProcessorConfig
import os

config = ProcessorConfig(
    enable_ai_validation=True,
    extract_articles=True,
    assess_materiality=True,
    chunk_size=800,
    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
)

processor = RegulatoryDocumentProcessor(config)

# Process all folders automatically
base_path = "/mnt/c/Users/doupa/Desktop/Ventures/Orabank"

print("🏛️ Building Comprehensive Regulatory Database...")

# Process Code Pénal
print("📖 Processing Code Pénal Gabonais...")
processor.process_document(os.path.join(base_path, "Code Pénal Gabonais.pdf"))

# Process all COBAC Instructions  
print("📋 Processing COBAC Instructions...")
processor.process_directory(os.path.join(base_path, "Instruction_COBAC"))

# Process all COBAC Regulations
print("📊 Processing COBAC Regulations...")
processor.process_directory(os.path.join(base_path, "Reglement_COBAC"))

# Export unified database
print("💾 Exporting unified database...")
processor.export_results("unified_regulatory_database.xlsx", 
                        include_validation=True, 
                        include_articles=True)

# Show summary
summary = processor.get_summary()
print(f"\n✅ Database Complete!")
print(f"📄 Documents processed: {summary['total_documents']}")
print(f"📋 Articles extracted: {len([a for doc in processor.processed_documents for a in doc.get('articles', [])])}")
print(f"🔍 Validation scores available: {summary['successful_extractions']}")
```

## 🗃️ Database Query Examples

Once you have the Excel database, you can query it:

### Filter by Document Type
```python
import pandas as pd

# Load the database
df_articles = pd.read_excel("regulatory_ai_processed.xlsx", sheet_name="Articles")

# Get only COBAC Regulations
regulations = df_articles[df_articles['Context'] == 'REGLEMENT']
print(f"Total COBAC Regulations: {len(regulations)}")

# Get only Instructions
instructions = df_articles[df_articles['Context'] == 'INSTRUCTION']
print(f"Total COBAC Instructions: {len(instructions)}")
```

### Filter by Materiality
```python
# Get all CRITICAL articles
critical_articles = df_articles[df_articles['Materiality'] == 'CRITICAL']
print(f"Critical articles requiring immediate attention: {len(critical_articles)}")

# Get HIGH materiality articles
high_articles = df_articles[df_articles['Materiality'] == 'HIGH']
print(f"High importance articles: {len(high_articles)}")
```

### Search by Content
```python
# Find all articles about capital requirements
capital_articles = df_articles[df_articles['Content'].str.contains('capital|ratio', case=False, na=False)]
print(f"Articles about capital/ratios: {len(capital_articles)}")
```

## 🎯 Best Practices for Database Building

### 1. **Process in Logical Order**
```python
# Recommended processing order:
1. Code Pénal (foundation legal framework)
2. COBAC Regulations (main regulatory framework)  
3. COBAC Instructions (implementation guidance)
```

### 2. **Monitor Progress**
```python
# Add progress tracking
import os
total_files = len([f for f in os.listdir("Reglement_COBAC") if f.endswith('.pdf')])
for i, file in enumerate(pdf_files):
    print(f"Processing {i+1}/{total_files}: {file}")
    processor.process_document(file)
```

### 3. **Handle Large Batches**
```python
# For very large document sets, process in chunks
def process_large_batch(directory, batch_size=20):
    files = get_pdf_files(directory)
    
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}")
        
        for file in batch:
            processor.process_document(file)
        
        # Optional: Save intermediate results
        processor.export_results(f"batch_{i//batch_size + 1}.xlsx")
```

## ✅ Verification Steps

### After building your database:

1. **Check Total Counts**
   ```python
   summary = processor.get_summary()
   print(f"Expected ~72 documents, got: {summary['total_documents']}")
   ```

2. **Verify All Document Types**
   ```python
   doc_types = summary['document_types']
   print("Document types found:", doc_types)
   # Should see: REGLEMENT, INSTRUCTION, CODE_PENAL
   ```

3. **Check Article Distribution**
   ```python
   all_articles = [a for doc in processor.processed_documents for a in doc.get('articles', [])]
   materiality_dist = {}
   for article in all_articles:
       level = article.materiality.value
       materiality_dist[level] = materiality_dist.get(level, 0) + 1
   print("Materiality distribution:", materiality_dist)
   ```

The tool is specifically designed to create comprehensive regulatory databases by processing all documents in your folders and combining them into a single, queryable Excel file. This is perfect for regulatory compliance analysis, risk assessment, and legal research.