"""
pdftomd - Ultra-simple PDF to Markdown extractor using Gemini.

KISS: One function, one query, complete extraction.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

__version__ = "2.0.0"
__all__ = ["extract", "extract_to_file"]


def extract(pdf_path: str | Path, model: str = "gemini-3-flash-preview") -> str:
    """
    Extract complete markdown from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        model: Gemini model to use (default: gemini-3-flash-preview)
        
    Returns:
        Complete markdown content as string
        
    Example:
        >>> from pdftomd import extract
        >>> markdown = extract("document.pdf")
        >>> print(markdown)
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Initialize client
    client = genai.Client()
    
    # Upload PDF to File API
    with open(pdf_path, "rb") as f:
        uploaded = client.files.upload(
            file=f, 
            config={"display_name": pdf_path.name, "mime_type": "application/pdf"}
        )
    
    # Simple but effective extraction prompt
    prompt = """Extract ALL content from this PDF to Markdown format.

Structure your output as:
1. YAML frontmatter with title, type, language, pages
2. Brief summary (2-3 sentences)  
3. Key points (5-10 bullets)
4. Table of contents
5. Full content with proper heading hierarchy
6. All tables with headers and data
7. Glossary of key terms (if present)

CRITICAL: Include EVERY word and paragraph. Do NOT summarize or skip any content.
Use proper Markdown: # for headings, | for tables, > for quotes, **bold**, *italic*.
Return ONLY the markdown."""

    # Generate content
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=uploaded.uri, mime_type="application/pdf"),
                    types.Part.from_text(text=prompt)
                ]
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,  # Low temperature for accuracy
            max_output_tokens=65536,  # Maximum output for complete documents
        )
    )
    
    # Handle empty response
    if not response.text:
        # Try to get text from candidates if direct access fails
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        raise RuntimeError("Empty response from Gemini API")
    
    return response.text


def extract_to_file(
    pdf_path: str | Path,
    output_path: str | Path = None,
    model: str = "gemini-3-flash-preview"
) -> Path:
    """
    Extract PDF to markdown and save to file.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Output path (default: same name with .md extension)
        model: Gemini model to use
        
    Returns:
        Path to the created markdown file
        
    Example:
        >>> from pdftomd import extract_to_file
        >>> output = extract_to_file("chapter.pdf")
        >>> print(f"Created: {output}")
    """
    pdf_path = Path(pdf_path)
    
    if output_path is None:
        output_path = pdf_path.with_suffix(".md")
    else:
        output_path = Path(output_path)
    
    markdown = extract(pdf_path, model=model)
    output_path.write_text(markdown, encoding="utf-8")
    
    return output_path
