# PyQt GUI User Guide

## 🖥️ Regulatory Document Processor GUI

A user-friendly graphical interface for processing regulatory documents with AI validation and article extraction.

## 🚀 Getting Started

### Installation

1. **Install Dependencies**:
```bash
cd /mnt/c/Users/doupa/Desktop/Ventures/Orabank
pip install -r requirements.txt
```

2. **Launch the GUI**:
```bash
python regulatory_gui.py
```

### First Launch

The application will open with a modern interface featuring four main tabs:
- **📋 Document Input**: User info and file selection
- **⚙️ Configuration**: Processing settings
- **🔄 Processing**: Real-time progress monitoring  
- **📊 Results**: Summary and download options

---

## 📋 Tab 1: Document Input

### User Information (Required)
- **First Name**: Enter your first name
- **Last Name**: Enter your last name
- **Filename Preview**: Shows how your output file will be named

**Example Output Filename**: `regulatory_analysis_John_Doe_20241218_143022.xlsx`

### Document Selection
- **📄 Add PDF Files**: Select individual PDF files
- **📂 Add Folder**: Select entire folders (processes all PDFs recursively)
- **🗑️ Clear All**: Remove all selected paths

**Supported Input Types**:
- Individual PDF files
- Folders containing PDFs (auto-scans subfolders)
- Mixed selection of files and folders

### Output Settings
- **Output Directory**: Choose where to save the Excel file
- **File Name**: Base name (user info and timestamp auto-added)
- **Preview**: Real-time preview of final filename

### Validation & Start
- **✅ Validate Inputs**: Check all inputs before processing
- **🚀 Start Processing**: Begin document processing (enabled after validation)

---

## ⚙️ Tab 2: Configuration

### Processing Settings
- **Chunk Size**: Text segment size (200-2000 characters)
- **Chunk Overlap**: Overlap between segments (50-500 characters)  
- **Max File Size**: Maximum PDF size limit (10-500 MB)

### AI Validation Settings
- **Enable AI Validation**: Toggle AI-powered quality assessment
- **API Key**: Anthropic API key (or set `ANTHROPIC_API_KEY` environment variable)
- **AI Model**: Choose Claude model:
  - `claude-3-haiku-20240307` (Fast, cost-effective)
  - `claude-3-sonnet-20240229` (Balanced)
  - `claude-3-opus-20240229` (Most capable)

### Feature Settings
- **Extract Regulatory Articles**: Find and extract article text
- **Assess Article Materiality**: AI-powered importance classification
- **Clean Extracted Text**: Remove formatting artifacts

### Presets
- **Basic Processing**: Fast processing without AI
- **AI Enhanced**: Full AI validation and analysis
- **Save as Preset**: Save current settings as default

---

## 🔄 Tab 3: Processing

### Progress Monitoring
- **Progress Bar**: Visual progress indicator (0-100%)
- **Status Label**: Current processing step
- **Processing Log**: Detailed real-time logging

### Controls
- **⏸️ Pause**: Temporarily halt processing (future feature)
- **⏹️ Cancel**: Stop processing and return to input

### Log Features
- **🗑️ Clear Log**: Empty the log display
- **💾 Save Log**: Export log to text file
- **Auto-scroll**: Automatically scrolls to show latest messages

### Log Examples
```
[14:30:22] 🚀 Processing started...
[14:30:23] Processing: R-2018-01.pdf
[14:30:45] ✓ Successfully processed: R-2018-01.pdf
[14:30:46] Running AI validation and article extraction...
[14:30:58] Document validation score: 87.5/100
[14:30:59] Extracted 12 articles
```

---

## 📊 Tab 4: Results

### Processing Summary
- **Total Documents**: Number of files processed
- **Successful**: Files processed without errors
- **Failed**: Files that encountered errors
- **Total Articles**: Articles extracted across all documents
- **Average Score**: Mean AI validation score

### Document Details Table
| Document | Type | Articles | Chunks | Validation Score | Status |
|----------|------|----------|---------|------------------|--------|
| R-2018-01.pdf | REGLEMENT | 12 | 25 | 87.5 | ✅ Success |

### Download Options
- **📄 File Preview**: Shows generated filename
- **📂 Open File**: Launch Excel file directly
- **📁 Open Folder**: Open containing directory

---

## 🎯 Step-by-Step Workflow

### Basic Processing (No AI)

1. **Enter User Info**:
   - First Name: `John`
   - Last Name: `Doe`

2. **Select Documents**:
   - Click **📂 Add Folder**
   - Navigate to: `/mnt/c/Users/doupa/Desktop/Ventures/Orabank`
   - Select the main folder (includes all subfolders)

3. **Set Output**:
   - Click **📂 Browse** for output directory
   - Keep default filename: `regulatory_analysis`

4. **Configure (Tab 2)**:
   - Click **Basic Processing** preset
   - Verify **Enable AI Validation** is unchecked

5. **Process (Tab 1)**:
   - Click **✅ Validate Inputs**
   - Click **🚀 Start Processing**

6. **Monitor (Tab 3)**:
   - Watch progress bar and logs
   - Processing time: ~10-20 minutes for 72 documents

7. **Download (Tab 4)**:
   - Review summary statistics
   - Click **📂 Open File** to view Excel results

### AI-Enhanced Processing

1. **Get API Key**:
   - Sign up at https://console.anthropic.com
   - Generate API key: `sk-ant-api03-...`

2. **Configure AI (Tab 2)**:
   - Click **AI Enhanced** preset
   - Enter API key in **API Key** field
   - Or set environment variable: `export ANTHROPIC_API_KEY="your-key"`

3. **Follow Steps 1-2 from Basic Processing**

4. **Process with AI**:
   - Processing time: 1-3 hours for 72 documents
   - Monitor validation scores in logs

5. **Enhanced Results**:
   - Articles with materiality assessment
   - Multi-dimensional validation scores
   - Quality metrics per document

---

## 📁 Output File Structure

### Generated Filename Format
```
{base_name}_{first_name}_{last_name}_{YYYYMMDD_HHMMSS}.xlsx
```

**Examples**:
- `regulatory_analysis_John_Doe_20241218_143022.xlsx`
- `cobac_review_Marie_Dubois_20241218_150315.xlsx`

### Excel Worksheets

#### 1. **Document_Metadata**
| File Name | Document Type | Page Count | Processed By | Processing Date |
|-----------|---------------|------------|--------------|-----------------|
| R-2018-01.pdf | REGLEMENT | 28 | John Doe | 2024-12-18 14:30:22 |

#### 2. **Articles** (AI Processing)
| Context | Regulation Name | Article Number | Content | Materiality | Materiality Reasoning |
|---------|----------------|----------------|---------|-------------|---------------------|
| REGLEMENT | R-2018-01 | Article 1 | Les établissements... | HIGH | Contains critical capital requirements |

#### 3. **Text_Chunks**
| File Name | Chunk Index | Text | Completeness Score | Reliability Score | Overall Score |
|-----------|-------------|------|-------------------|-------------------|---------------|
| R-2018-01.pdf | 1 | Article 1. Les... | 90 | 85 | 87.5 |

#### 4. **Validation_Results** (AI Processing)
| File Name | Overall Score | Completeness | Reliability | Legal Structure | Issues |
|-----------|---------------|--------------|-------------|-----------------|--------|
| R-2018-01.pdf | 87.5 | 90 | 85 | 87 | Minor formatting issues |

#### 5. **Summary**
| Processing Date | Total Documents | Processed By | User First Name | User Last Name |
|----------------|----------------|--------------|------------------|-----------------|
| 2024-12-18 14:30:22 | 72 | John Doe | John | Doe |

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. **"Module 'PyQt5' not found"**
```bash
pip install PyQt5>=5.15.0
```

#### 2. **"No PDF files found"**
- Verify folder contains `.pdf` files
- Check file permissions
- Ensure PDFs are not corrupted

#### 3. **"AI validation failed"**
- Check API key format: `sk-ant-api03-...`
- Verify internet connection
- Check API rate limits

#### 4. **"Processing cancelled unexpectedly"**
- Check available disk space
- Verify output directory permissions
- Review processing logs for errors

#### 5. **"Excel export failed"**
- Close Excel if output file is open
- Check output directory permissions
- Ensure sufficient disk space

### Performance Tips

#### For Large Document Sets (100+ files):
1. **Use Basic Processing first** to test
2. **Process in smaller batches** if memory issues occur
3. **Close other applications** during processing
4. **Use SSD storage** for faster I/O

#### For Better AI Performance:
1. **Use claude-3-haiku** for faster processing
2. **Process during off-peak hours** for better API response
3. **Monitor API usage** to avoid rate limits

### Log Analysis

#### Success Indicators:
```
✓ Successfully processed: filename.pdf
✓ AI validator initialized
✓ Extracted X articles
✅ Processing completed successfully!
```

#### Warning Indicators:
```
⚠️ No ANTHROPIC_API_KEY found
⚠️ AI validator not available (using fallback)
```

#### Error Indicators:
```
✗ Failed to process: filename.pdf
✗ Error processing filename: [error details]
❌ Critical error: [error details]
```

---

## 🎨 Interface Features

### Modern Design
- **Clean, professional appearance**
- **Intuitive tab-based navigation**
- **Real-time feedback and validation**
- **Responsive progress indicators**

### User Experience
- **Auto-save settings** between sessions
- **Live filename preview** updates
- **Comprehensive validation** before processing
- **Detailed logging** for transparency

### Accessibility
- **Clear visual hierarchy**
- **Descriptive button labels with emojis**
- **Logical tab flow**
- **Error messages with actionable guidance**

---

## 📧 Support & Feedback

### Getting Help
- **Check logs** for specific error details
- **Review troubleshooting section** above
- **Test with single files** before processing large batches

### Reporting Issues
When reporting problems, please include:
1. **Error messages** from the log
2. **System information** (Windows/Linux/Mac)
3. **File types and sizes** being processed
4. **Configuration settings** used

The GUI provides a complete, user-friendly solution for regulatory document processing with professional-grade output suitable for compliance and regulatory analysis.