# pdftomd

Ultra-simple PDF to Markdown extractor using Google Gemini.

**KISS principle**: One function, one query, complete extraction.

## Installation

```bash
pip install -e .
```

## Usage

### Python API

```python
from pdftomd import extract, extract_to_file

# Get markdown as string
markdown = extract("document.pdf")
print(markdown)

# Save directly to file
output_path = extract_to_file("document.pdf")
print(f"Created: {output_path}")

# Custom output path
extract_to_file("document.pdf", "output.md")
```

### CLI

```bash
# Basic usage
pdftomd document.pdf

# Custom output
pdftomd document.pdf -o output.md

# Different model
pdftomd document.pdf -m gemini-2.0-flash
```

## Configuration

Set your Gemini API key:

```bash
export GOOGLE_API_KEY="your-api-key"
```

Or create a `.env` file:

```
GOOGLE_API_KEY=your-api-key
```

## How it works

1. Upload PDF to Gemini File API
2. Send extraction prompt
3. Return complete markdown

That's it. No chunking, no agents, no complexity.

## License

MIT
