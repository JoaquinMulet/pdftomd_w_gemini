# PDFtoMD

Extract PDF content to well-structured Markdown using **Google Gemini 3**.

A powerful Python library that leverages Gemini's advanced multimodal capabilities to extract complete information from PDF documents and convert them to clean, readable Markdown.

## Features

- **Complete Extraction**: Captures all document elements - text, tables, images, equations, code, references
- **Structured Output**: Uses Pydantic models for validated, consistent extraction
- **Gemini 3 Flash Preview**: Leverages the latest Gemini model for best results
- **Google Search Integration**: Optional web context for enhanced understanding
- **Streaming Support**: Progress feedback for large documents
- **CLI & Library**: Use from command line or integrate into your Python projects

## Installation

```bash
# Clone the repository
git clone https://github.com/JoaquinMulet/pdftomd_w_gemini.git
cd pdftomd_w_gemini

# Create virtual environment
uv venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv pip install -e .
```

## Setup

Create a `.env` file with your Google API key:

```env
GOOGLE_API_KEY=your-api-key-here
```

## Usage

### Command Line

```bash
# Basic usage
python -m pdftomd document.pdf

# Specify output file
python -m pdftomd document.pdf -o output.md

# Use Pro model for complex documents
python -m pdftomd document.pdf --model gemini-3-pro-preview

# Enable Google Search for additional context
python -m pdftomd document.pdf --search

# Output as JSON
python -m pdftomd document.pdf --json

# Verbose mode
python -m pdftomd document.pdf -v
```

### Python Library

```python
from pdftomd import PDFExtractor

# Initialize extractor
extractor = PDFExtractor()

# Extract to Markdown
markdown = extractor.extract_to_markdown("document.pdf")
print(markdown)

# Or get structured data
document = extractor.extract("document.pdf")
print(f"Title: {document.metadata.title}")
print(f"Sections: {len(document.sections)}")
print(f"Tables: {len(document.tables)}")

# Save to file
extractor.extract_to_file("document.pdf", "output.md")
```

### Advanced Usage

```python
from pdftomd import PDFExtractor

# Use Gemini 3 Pro with Google Search
extractor = PDFExtractor(
    model="gemini-3-pro-preview",
    use_search=True,
    use_url_context=True,
    temperature=0.1,
)

# Extract with web context
document = extractor.extract("research_paper.pdf", include_web_context=True)

# Access extracted data
print(document.metadata.title)
print(document.summary)

for section in document.sections:
    print(f"{'#' * section.level} {section.title}")

for table in document.tables:
    print(f"Table: {table.caption}")

for image in document.images:
    print(f"Figure: {image.figure_number} - {image.description}")
```

### Streaming for Large Documents

```python
from pdftomd import PDFExtractor

extractor = PDFExtractor()

def progress_callback(chunk):
    print(".", end="", flush=True)

document = extractor.extract_with_streaming(
    "large_document.pdf",
    callback=progress_callback
)
```

## Extracted Data Structure

The library extracts the following elements:

| Element | Description |
|---------|-------------|
| **Metadata** | Title, authors, date, document type, language |
| **Summary** | Executive summary of the document |
| **Key Points** | Main takeaways and important points |
| **Sections** | Full text content with heading hierarchy |
| **Tables** | All tables with headers, rows, and captions |
| **Images** | Descriptions of figures, charts, diagrams |
| **Equations** | Mathematical equations in LaTeX format |
| **Code** | Code blocks and algorithms |
| **References** | Bibliography and citations |
| **Glossary** | Key terms and definitions |

## Models

| Model | Description |
|-------|-------------|
| `gemini-3-flash-preview` | **Default**. Fast, feature-rich, best for most uses |
| `gemini-3-pro-preview` | More capable, better for complex documents |
| `gemini-2.0-flash` | Production stable, fallback option |

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `model` | `gemini-3-flash-preview` | Gemini model to use |
| `temperature` | `0.1` | Generation temperature (0.0-1.0) |
| `use_search` | `False` | Enable Google Search tool |
| `use_url_context` | `False` | Enable URL context tool |

## License

MIT License
