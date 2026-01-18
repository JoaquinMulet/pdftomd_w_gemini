"""
Document Converter - Convert ExtractedDocument to Markdown.

Handles formatting of all document elements including
YAML frontmatter, sections, tables, images, equations, and code.
"""

from typing import List
from pdftomd.models import (
    ExtractedDocument,
    Section,
    Table,
    Image,
    CodeBlock,
    Equation,
    Reference,
)


class DocumentConverter:
    """
    Convert an ExtractedDocument to well-formatted Markdown.
    
    Produces clean, readable Markdown with:
    - YAML frontmatter for metadata
    - Proper heading hierarchy
    - Formatted tables
    - Image descriptions with alt-text
    - Code blocks with syntax highlighting
    - LaTeX equations
    - Bibliography section
    """
    
    def convert(self, document: ExtractedDocument) -> str:
        """
        Convert an ExtractedDocument to Markdown.
        
        Args:
            document: The extracted document data.
            
        Returns:
            Formatted Markdown string.
        """
        parts = []
        
        # YAML Frontmatter
        parts.append(self._render_frontmatter(document))
        
        # Summary
        parts.append(self._render_summary(document))
        
        # Key Points
        if document.key_points:
            parts.append(self._render_key_points(document.key_points))
        
        # Table of Contents (optional - for long documents)
        if len(document.sections) > 5:
            parts.append(self._render_toc(document.sections))
        
        # Main Sections
        for section in document.sections:
            parts.append(self._render_section(section))
        
        # Tables (if not already inline)
        if document.tables:
            parts.append(self._render_tables_section(document.tables))
        
        # Figures (descriptions)
        if document.images:
            parts.append(self._render_images_section(document.images))
        
        # Equations
        if document.equations:
            parts.append(self._render_equations_section(document.equations))
        
        # Code Blocks
        if document.code_blocks:
            parts.append(self._render_code_section(document.code_blocks))
        
        # References
        if document.references:
            parts.append(self._render_references(document.references))
        
        # Glossary
        if document.glossary:
            parts.append(self._render_glossary(document.glossary))
        
        return "\n\n".join(filter(None, parts))
    
    def _render_frontmatter(self, document: ExtractedDocument) -> str:
        """Render YAML frontmatter with metadata."""
        meta = document.metadata
        lines = ["---"]
        
        lines.append(f"title: \"{meta.title}\"")
        
        if meta.subtitle:
            lines.append(f"subtitle: \"{meta.subtitle}\"")
        
        if meta.authors:
            if len(meta.authors) == 1:
                lines.append(f"author: \"{meta.authors[0]}\"")
            else:
                lines.append("authors:")
                for author in meta.authors:
                    lines.append(f"  - \"{author}\"")
        
        if meta.date:
            lines.append(f"date: \"{meta.date}\"")
        
        if meta.document_type:
            lines.append(f"type: \"{meta.document_type}\"")
        
        if meta.language:
            lines.append(f"language: \"{meta.language}\"")
        
        if meta.total_pages:
            lines.append(f"pages: {meta.total_pages}")
        
        lines.append("---")
        
        return "\n".join(lines)
    
    def _render_summary(self, document: ExtractedDocument) -> str:
        """Render the document summary."""
        parts = ["## Summary", "", document.summary]
        return "\n".join(parts)
    
    def _render_key_points(self, key_points: List[str]) -> str:
        """Render key points as a bullet list."""
        lines = ["## Key Points", ""]
        for point in key_points:
            lines.append(f"- {point}")
        return "\n".join(lines)
    
    def _render_toc(self, sections: List[Section]) -> str:
        """Render a table of contents."""
        lines = ["## Table of Contents", ""]
        
        for section in sections:
            indent = "  " * (section.level - 1)
            # Create anchor link
            anchor = section.title.lower().replace(" ", "-")
            anchor = "".join(c for c in anchor if c.isalnum() or c == "-")
            lines.append(f"{indent}- [{section.title}](#{anchor})")
        
        return "\n".join(lines)
    
    def _render_section(self, section: Section) -> str:
        """Render a section with proper heading level."""
        heading = "#" * section.level
        return f"{heading} {section.title}\n\n{section.content}"
    
    def _render_tables_section(self, tables: List[Table]) -> str:
        """Render all tables."""
        if not tables:
            return ""
        
        lines = ["## Tables", ""]
        
        for i, table in enumerate(tables, 1):
            if table.caption:
                lines.append(f"### {table.caption}")
            else:
                lines.append(f"### Table {i}")
            
            if table.context:
                lines.append(f"\n*{table.context}*\n")
            
            lines.append("")
            lines.append(self._render_table(table))
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_table(self, table: Table) -> str:
        """Render a single table in Markdown format."""
        if not table.headers and not table.rows:
            return ""
        
        lines = []
        
        # Headers
        if table.headers:
            lines.append("| " + " | ".join(table.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")
        
        # Rows
        for row in table.rows:
            # Ensure row has same number of cells as headers
            while len(row) < len(table.headers):
                row.append("")
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return "\n".join(lines)
    
    def _render_images_section(self, images: List[Image]) -> str:
        """Render image descriptions."""
        if not images:
            return ""
        
        lines = ["## Figures and Images", ""]
        
        for image in images:
            if image.figure_number:
                lines.append(f"### {image.figure_number}")
            
            if image.caption:
                lines.append(f"**{image.caption}**")
            
            lines.append("")
            lines.append(f"*Description:* {image.description}")
            lines.append("")
            lines.append(f"*Context:* {image.context}")
            lines.append("")
            lines.append(f"*Alt-text:* {image.alt_text}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_equations_section(self, equations: List[Equation]) -> str:
        """Render equations in LaTeX format."""
        if not equations:
            return ""
        
        lines = ["## Equations", ""]
        
        for eq in equations:
            if eq.equation_number:
                lines.append(f"### Equation {eq.equation_number}")
            
            lines.append("")
            lines.append(f"$$\n{eq.latex}\n$$")
            
            if eq.description:
                lines.append(f"\n*{eq.description}*")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_code_section(self, code_blocks: List[CodeBlock]) -> str:
        """Render code blocks."""
        if not code_blocks:
            return ""
        
        lines = ["## Code and Algorithms", ""]
        
        for i, block in enumerate(code_blocks, 1):
            if block.context:
                lines.append(f"### {block.context}")
            else:
                lines.append(f"### Code Block {i}")
            
            lines.append("")
            lang = block.language or ""
            lines.append(f"```{lang}")
            lines.append(block.code)
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_references(self, references: List[Reference]) -> str:
        """Render bibliography/references."""
        if not references:
            return ""
        
        lines = ["## References", ""]
        
        for ref in references:
            if ref.number:
                lines.append(f"[{ref.number}] {ref.citation}")
            else:
                lines.append(f"- {ref.citation}")
        
        return "\n".join(lines)
    
    def _render_glossary(self, glossary: List[str]) -> str:
        """Render glossary terms."""
        if not glossary:
            return ""
        
        lines = ["## Glossary", ""]
        
        for term in glossary:
            lines.append(f"- {term}")
        
        return "\n".join(lines)
