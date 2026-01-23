"""
pdftomd - Ultra-simple PDF to Markdown extractor using Gemini.

KISS: One function, one query, complete extraction.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

__version__ = "2.1.0"
__all__ = ["extract", "extract_to_file", "extract_with_stats", "ExtractionResult"]


# Gemini 3 Flash pricing (USD per 1M tokens) - Paid Tier
GEMINI_PRICING = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3-pro-preview": {"input": 2.50, "output": 15.00},
    # Fallback for unknown models
    "default": {"input": 0.50, "output": 3.00}
}


@dataclass
class ExtractionResult:
    """Result of PDF extraction with token usage and cost stats."""
    markdown: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str
    
    def format_stats(self) -> str:
        """Format stats for CLI display."""
        return f"""
--- Token Usage ---
Input tokens:  {self.input_tokens:,}
Output tokens: {self.output_tokens:,}
Total tokens:  {self.total_tokens:,}

--- Cost ({self.model}) ---
Input:  ${self.input_cost:.6f}
Output: ${self.output_cost:.6f}
Total:  ${self.total_cost:.6f}
"""


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


def extract_with_stats(pdf_path: str | Path, model: str = "gemini-3-flash-preview") -> ExtractionResult:
    """
    Extract complete markdown from a PDF file with token usage stats.
    
    Args:
        pdf_path: Path to the PDF file
        model: Gemini model to use (default: gemini-3-flash-preview)
        
    Returns:
        ExtractionResult with markdown content, token counts, and costs
        
    Example:
        >>> from pdftomd import extract_with_stats
        >>> result = extract_with_stats("document.pdf")
        >>> print(result.markdown)
        >>> print(result.format_stats())
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
        if response.candidates and response.candidates[0].content.parts:
            markdown = response.candidates[0].content.parts[0].text
        else:
            raise RuntimeError("Empty response from Gemini API")
    else:
        markdown = response.text
    
    # Extract token usage from response
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else 0
    output_tokens = usage.candidates_token_count if usage else 0
    total_tokens = usage.total_token_count if usage else 0
    
    # Calculate costs
    pricing = GEMINI_PRICING.get(model, GEMINI_PRICING["default"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return ExtractionResult(
        markdown=markdown,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
        model=model
    )


def extract_to_file(
    pdf_path: str | Path,
    output_path: str | Path = None,
    model: str = "gemini-3-flash-preview",
    return_stats: bool = False
) -> Path | tuple[Path, ExtractionResult]:
    """
    Extract PDF to markdown and save to file.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Output path (default: same name with .md extension)
        model: Gemini model to use
        return_stats: If True, also return ExtractionResult with token usage
        
    Returns:
        Path to the created markdown file, or (Path, ExtractionResult) if return_stats=True
        
    Example:
        >>> from pdftomd import extract_to_file
        >>> output = extract_to_file("chapter.pdf")
        >>> print(f"Created: {output}")
        
        # With stats
        >>> output, stats = extract_to_file("chapter.pdf", return_stats=True)
        >>> print(stats.format_stats())
    """
    pdf_path = Path(pdf_path)
    
    if output_path is None:
        output_path = pdf_path.with_suffix(".md")
    else:
        output_path = Path(output_path)
    
    if return_stats:
        result = extract_with_stats(pdf_path, model=model)
        output_path.write_text(result.markdown, encoding="utf-8")
        return output_path, result
    else:
        markdown = extract(pdf_path, model=model)
        output_path.write_text(markdown, encoding="utf-8")
        return output_path

