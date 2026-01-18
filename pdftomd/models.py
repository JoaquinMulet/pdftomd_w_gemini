"""
Pydantic models for structured PDF extraction.

These models define the schema for Gemini's structured output,
ensuring consistent and validated extraction results.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata extracted from the PDF document."""
    
    title: str = Field(
        description="Main title of the document"
    )
    subtitle: Optional[str] = Field(
        default=None,
        description="Subtitle if present"
    )
    authors: Optional[List[str]] = Field(
        default=None,
        description="List of authors if identified"
    )
    date: Optional[str] = Field(
        default=None,
        description="Publication or creation date"
    )
    total_pages: Optional[int] = Field(
        default=None,
        description="Total number of pages in the document"
    )
    language: Optional[str] = Field(
        default=None,
        description="Primary language of the document"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Type of document (e.g., academic paper, report, manual, book chapter)"
    )


class Section(BaseModel):
    """A section or subsection of the document."""
    
    title: str = Field(
        description="Section heading/title"
    )
    level: int = Field(
        description="Heading level (1-6, where 1 is the main heading)",
        ge=1,
        le=6
    )
    content: str = Field(
        description="Full content of the section in Markdown format, preserving formatting"
    )


class Table(BaseModel):
    """A table extracted from the document."""
    
    caption: Optional[str] = Field(
        default=None,
        description="Table caption or title if present"
    )
    headers: List[str] = Field(
        description="Column headers of the table"
    )
    rows: List[List[str]] = Field(
        description="Table data rows, each row is a list of cell values"
    )
    context: Optional[str] = Field(
        default=None,
        description="Brief context about what the table represents"
    )


class Image(BaseModel):
    """Description of an image or figure in the document."""
    
    figure_number: Optional[str] = Field(
        default=None,
        description="Figure number if labeled (e.g., 'Figure 3.1')"
    )
    caption: Optional[str] = Field(
        default=None,
        description="Image caption if present"
    )
    description: str = Field(
        description="Detailed description of what the image shows"
    )
    context: str = Field(
        description="How this image relates to the surrounding text"
    )
    alt_text: str = Field(
        description="Concise alt-text suitable for accessibility"
    )


class CodeBlock(BaseModel):
    """A code block or algorithm in the document."""
    
    language: Optional[str] = Field(
        default=None,
        description="Programming language if identifiable"
    )
    code: str = Field(
        description="The code content"
    )
    context: Optional[str] = Field(
        default=None,
        description="What the code demonstrates or implements"
    )


class Equation(BaseModel):
    """A mathematical equation in the document."""
    
    equation_number: Optional[str] = Field(
        default=None,
        description="Equation number if labeled"
    )
    latex: str = Field(
        description="The equation in LaTeX format"
    )
    description: Optional[str] = Field(
        default=None,
        description="What the equation represents"
    )


class Reference(BaseModel):
    """A bibliographic reference."""
    
    number: Optional[str] = Field(
        default=None,
        description="Reference number or key"
    )
    citation: str = Field(
        description="Full citation text"
    )


class ExtractedDocument(BaseModel):
    """Complete extracted content from a PDF document."""
    
    metadata: DocumentMetadata = Field(
        description="Document metadata"
    )
    summary: str = Field(
        description="Executive summary of the document content (2-3 paragraphs)"
    )
    key_points: List[str] = Field(
        description="Main key points or takeaways from the document"
    )
    sections: List[Section] = Field(
        description="All sections of the document in order"
    )
    tables: List[Table] = Field(
        default_factory=list,
        description="All tables found in the document"
    )
    images: List[Image] = Field(
        default_factory=list,
        description="All images/figures found in the document"
    )
    code_blocks: List[CodeBlock] = Field(
        default_factory=list,
        description="All code blocks or algorithms found"
    )
    equations: List[Equation] = Field(
        default_factory=list,
        description="Important mathematical equations"
    )
    references: List[Reference] = Field(
        default_factory=list,
        description="Bibliography/references if present"
    )
    glossary: Optional[List[str]] = Field(
        default=None,
        description="Key terms and definitions if present"
    )
