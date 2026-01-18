"""
PDFtoMD - Extract PDF content to Markdown using Google Gemini

A powerful library to extract complete information from PDFs
and convert them to well-structured Markdown documents.
"""

from pdftomd.extractor import PDFExtractor
from pdftomd.models import (
    ExtractedDocument,
    DocumentMetadata,
    Section,
    Table,
    Image,
)

__version__ = "0.1.0"
__all__ = [
    "PDFExtractor",
    "ExtractedDocument",
    "DocumentMetadata",
    "Section",
    "Table",
    "Image",
]
