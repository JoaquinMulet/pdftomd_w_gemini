"""
PDF Extractor - Core extraction logic using Google Gemini 3.

Uses structured outputs with Pydantic models for consistent,
validated extraction results. Leverages Gemini 3's advanced features
including integrated tools for enhanced extraction.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Literal

from google import genai
from google.genai import types
from dotenv import load_dotenv

from pdftomd.models import ExtractedDocument, TokenUsage, ExtractionResult


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extract content from PDF documents using Google Gemini 3.
    
    Uses structured outputs to ensure consistent, validated extraction
    of all document elements including text, tables, images, and more.
    
    Leverages Gemini 3's advanced features:
    - Native PDF understanding
    - Structured JSON outputs with Pydantic schema
    - Integrated tools (Google Search for context)
    
    Example:
        >>> extractor = PDFExtractor()
        >>> document = extractor.extract("document.pdf")
        >>> markdown = extractor.to_markdown(document)
    """
    
    # Gemini 3 Flash Preview - latest with best features
    DEFAULT_MODEL = "gemini-3-flash-preview"
    
    # Alternative models
    MODELS = {
        "flash-preview": "gemini-3-flash-preview",  # Fast, feature-rich
        "pro-preview": "gemini-3-pro-preview",      # More capable, tools
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        use_search: bool = False,
        use_url_context: bool = False,
    ):
        """
        Initialize the PDF extractor.
        
        Args:
            api_key: Google API key. If None, loads from GOOGLE_API_KEY env var.
            model: Gemini model to use. Default: gemini-3-flash-preview
            temperature: Generation temperature (0.0-1.0). Lower = more deterministic.
            use_search: Enable Google Search tool for additional context (Gemini 3 only).
            use_url_context: Enable URL context tool for web references (Gemini 3 only).
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Provide api_key parameter or set GOOGLE_API_KEY env var."
            )
        
        self.model = model
        self.temperature = temperature
        self.use_search = use_search
        self.use_url_context = use_url_context
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info(f"PDFExtractor initialized with model: {self.model}")
        if use_search:
            logger.info("Google Search integration enabled")
        if use_url_context:
            logger.info("URL Context integration enabled")
    
    def _build_tools_config(self) -> Optional[list]:
        """Build tools configuration for Gemini 3 features."""
        tools = []
        
        if self.use_search:
            tools.append({"google_search": {}})
        
        if self.use_url_context:
            tools.append({"url_context": {}})
        
        return tools if tools else None
    
    def _build_extraction_prompt(self, include_web_context: bool = False) -> str:
        """Build the extraction prompt for Gemini."""
        base_prompt = """You are an expert document analyzer with deep expertise in technical content extraction. Extract ALL information from this PDF document with maximum completeness and accuracy.

Your task is to create a comprehensive structured extraction that captures:

## METADATA
- Title, subtitle, authors, date, document type, language
- Identify the type: academic paper, textbook chapter, technical manual, report, etc.

## SUMMARY
- Write a thorough executive summary (2-3 detailed paragraphs)
- Cover the main topic, methodology, findings, and conclusions

## KEY POINTS
- List 5-10 main takeaways and important points
- These should be actionable insights or critical information

## SECTIONS
- Extract ALL text content, preserving the exact document structure
- Use appropriate heading levels (1-6) matching the original hierarchy
- Keep the original text formatting (bold, italic, lists, bullet points)
- Include ALL paragraphs - do NOT summarize or skip any content
- Preserve technical terminology exactly as written

## TABLES
- Extract every table with complete headers and all data rows
- Include table captions and numbers
- Preserve the original formatting and alignment intent

## IMAGES/FIGURES
- Describe EVERY image, diagram, chart, or figure in detail
- Include figure numbers and captions exactly as they appear
- Explain what the visual shows and its significance
- For diagrams, describe the components and their relationships
- For charts, describe the data trends and key values

## CODE & ALGORITHMS
- Extract all code blocks, pseudocode, or algorithms
- Identify the programming language when possible
- Include any code comments

## EQUATIONS
- Capture all mathematical equations in LaTeX format
- Include equation numbers if present
- Briefly describe what each equation represents

## REFERENCES
- List all bibliographic references completely
- Preserve the original citation format

## GLOSSARY
- Extract key terms and their definitions if present

CRITICAL RULES:
- Be EXHAUSTIVE - capture every piece of information, no matter how small
- Preserve the EXACT original structure and hierarchy
- Use proper Markdown formatting in section content
- For images, your descriptions must convey ALL visual information
- NEVER skip, summarize, or paraphrase content - include it verbatim
- Technical accuracy is paramount - preserve all formulas, numbers, and terminology exactly"""
        
        if include_web_context:
            base_prompt += """

ADDITIONAL CONTEXT:
- If you encounter references to external resources, papers, or concepts, you may use Google Search to provide additional context
- When technical terms would benefit from clarification, include brief explanations"""
        
        base_prompt += "\n\nNow extract everything from this document:"
        
        return base_prompt
    
    def _load_pdf(self, pdf_path: Union[str, Path]) -> types.Part:
        """
        Load a PDF file as a Gemini Part.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            A Gemini Part containing the PDF data.
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"File must be a PDF: {pdf_path}")
        
        file_size = pdf_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Loading PDF: {pdf_path.name} ({file_size:.2f} MB)")
        
        return types.Part.from_bytes(
            data=pdf_path.read_bytes(),
            mime_type="application/pdf",
        )
    
    def extract(
        self,
        pdf_path: Union[str, Path],
        include_web_context: bool = False,
        max_retries: int = 3,
    ) -> ExtractedDocument:
        """
        Extract content from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file.
            include_web_context: If True and search is enabled, adds web context.
            max_retries: Maximum number of retry attempts on failure.
            
        Returns:
            ExtractedDocument with all extracted content.
        """
        import time
        
        pdf_part = self._load_pdf(pdf_path)
        prompt = self._build_extraction_prompt(
            include_web_context=include_web_context and self.use_search
        )
        
        logger.info(f"Sending PDF to {self.model} for extraction...")
        
        # Build generation config with explicit output token limit
        config = {
            "response_mime_type": "application/json",
            "response_schema": ExtractedDocument,
            "temperature": self.temperature,
            "max_output_tokens": 65536,  # Maximum to avoid truncation
        }
        
        # Add Gemini 3 tools if enabled
        tools = self._build_tools_config()
        if tools:
            config["tools"] = tools
        
        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[pdf_part, prompt],
                    config=config,
                )
                
                logger.info("Extraction complete. Validating response...")
                
                # Check finish_reason for diagnostics
                finish_reason = None
                if response.candidates and len(response.candidates) > 0:
                    finish_reason = response.candidates[0].finish_reason
                    logger.info(f"Finish reason: {finish_reason}")
                    
                    # Detect truncation
                    if finish_reason and str(finish_reason) == "MAX_TOKENS":
                        logger.warning("Response was TRUNCATED due to max_output_tokens limit!")
                    
                    # Handle RECITATION - retry may help
                    if finish_reason and "RECITATION" in str(finish_reason):
                        logger.warning("Model stopped due to RECITATION detection. Retrying...")
                        raise ValueError(f"RECITATION detected - model stopped to avoid copying content.")
                    
                    # Handle SAFETY - content filtered
                    if finish_reason and "SAFETY" in str(finish_reason):
                        raise ValueError(f"Content blocked due to safety filters.")
                
                # Check for empty response
                if response.text is None:
                    raise ValueError(f"API returned empty response (None). Finish reason: {finish_reason}")
                
                # Log response size for debugging
                response_len = len(response.text)
                logger.info(f"Response size: {response_len:,} characters")
                
                # Log token usage for cost analysis
                if response.usage_metadata:
                    usage = response.usage_metadata
                    logger.info(
                        f"Tokens - In: {usage.prompt_token_count:,} | "
                        f"Out: {usage.candidates_token_count:,} | "
                        f"Total: {usage.total_token_count:,}"
                    )
                
                # Parse and validate the response
                document = ExtractedDocument.model_validate_json(response.text)
                
                logger.info(
                    f"Extracted: {len(document.sections)} sections, "
                    f"{len(document.tables)} tables, "
                    f"{len(document.images)} images, "
                    f"{len(document.equations)} equations"
                )
                
                return document
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Extraction failed after {max_retries} attempts: {e}")
                    raise
    
    def extract_with_stats(
        self,
        pdf_path: Union[str, Path],
        include_web_context: bool = False,
        max_retries: int = 3,
    ) -> ExtractionResult:
        """
        Extract content from a PDF document with token usage statistics.
        
        This method is useful for cost analysis and monitoring API usage.
        
        Args:
            pdf_path: Path to the PDF file.
            include_web_context: If True and search is enabled, adds web context.
            max_retries: Maximum number of retry attempts on failure.
            
        Returns:
            ExtractionResult with document, token usage, and metadata.
        """
        import time
        
        pdf_part = self._load_pdf(pdf_path)
        prompt = self._build_extraction_prompt(
            include_web_context=include_web_context and self.use_search
        )
        
        logger.info(f"Sending PDF to {self.model} for extraction...")
        
        config = {
            "response_mime_type": "application/json",
            "response_schema": ExtractedDocument,
            "temperature": self.temperature,
            "max_output_tokens": 65536,
        }
        
        tools = self._build_tools_config()
        if tools:
            config["tools"] = tools
        
        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[pdf_part, prompt],
                    config=config,
                )
                
                # Get finish reason
                finish_reason = "UNKNOWN"
                was_truncated = False
                if response.candidates and len(response.candidates) > 0:
                    finish_reason = str(response.candidates[0].finish_reason or "STOP")
                    was_truncated = finish_reason == "MAX_TOKENS"
                    if was_truncated:
                        logger.warning("Response was TRUNCATED due to max_output_tokens limit!")
                    
                    # Handle RECITATION - model stopped due to potential content copying
                    if "RECITATION" in finish_reason:
                        logger.warning(f"Model stopped due to RECITATION detection. Retrying...")
                        raise ValueError(f"RECITATION detected - model stopped to avoid copying content. Retry may help.")
                    
                    # Handle SAFETY - content filtered
                    if "SAFETY" in finish_reason:
                        raise ValueError(f"Content was blocked due to safety filters. Finish reason: {finish_reason}")
                
                if response.text is None:
                    raise ValueError(f"API returned empty response. Finish reason: {finish_reason}")
                
                # Get token usage
                token_usage = TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )
                if response.usage_metadata:
                    usage = response.usage_metadata
                    token_usage = TokenUsage(
                        prompt_tokens=usage.prompt_token_count or 0,
                        completion_tokens=usage.candidates_token_count or 0,
                        total_tokens=usage.total_token_count or 0,
                    )
                    logger.info(f"Tokens - {token_usage}")
                
                # Parse document
                document = ExtractedDocument.model_validate_json(response.text)
                
                logger.info(
                    f"Extracted: {len(document.sections)} sections, "
                    f"{len(document.tables)} tables, "
                    f"{len(document.images)} images"
                )
                
                return ExtractionResult(
                    document=document,
                    token_usage=token_usage,
                    finish_reason=finish_reason,
                    was_truncated=was_truncated,
                )
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Extraction failed after {max_retries} attempts: {e}")
                    raise
    
    def to_markdown(self, document: ExtractedDocument) -> str:
        """
        Convert an ExtractedDocument to Markdown format.
        
        Args:
            document: The extracted document data.
            
        Returns:
            Formatted Markdown string.
        """
        from pdftomd.converter import DocumentConverter
        converter = DocumentConverter()
        return converter.convert(document)
    
    def extract_to_markdown(
        self,
        pdf_path: Union[str, Path],
        include_web_context: bool = False,
    ) -> str:
        """
        Extract a PDF and convert directly to Markdown.
        
        Convenience method that combines extract() and to_markdown().
        
        Args:
            pdf_path: Path to the PDF file.
            include_web_context: If True and search is enabled, adds web context.
            
        Returns:
            Formatted Markdown string.
        """
        document = self.extract(pdf_path, include_web_context=include_web_context)
        return self.to_markdown(document)
    
    def extract_to_file(
        self,
        pdf_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        include_web_context: bool = False,
    ) -> Path:
        """
        Extract a PDF and save as a Markdown file.
        
        Args:
            pdf_path: Path to the PDF file.
            output_path: Output file path. If None, uses PDF name with .md extension.
            include_web_context: If True and search is enabled, adds web context.
            
        Returns:
            Path to the created Markdown file.
        """
        pdf_path = Path(pdf_path)
        
        if output_path is None:
            output_path = pdf_path.with_suffix(".md")
        else:
            output_path = Path(output_path)
        
        markdown = self.extract_to_markdown(
            pdf_path,
            include_web_context=include_web_context
        )
        output_path.write_text(markdown, encoding="utf-8")
        
        logger.info(f"Saved Markdown to: {output_path}")
        
        return output_path
    
    def extract_with_streaming(
        self,
        pdf_path: Union[str, Path],
        callback=None,
    ) -> ExtractedDocument:
        """
        Extract content with streaming output for progress feedback.
        
        This method streams the response for large documents,
        calling the callback function with progress updates.
        
        Args:
            pdf_path: Path to the PDF file.
            callback: Optional function called with (chunk_text) for each chunk.
            
        Returns:
            ExtractedDocument with all extracted content.
        """
        pdf_part = self._load_pdf(pdf_path)
        prompt = self._build_extraction_prompt()
        
        logger.info(f"Streaming extraction from {self.model}...")
        
        # For streaming, we can't use structured output directly
        # We stream first, then parse at the end
        full_response = ""
        
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=[pdf_part, prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": ExtractedDocument,
                "temperature": self.temperature,
            },
        )
        
        for chunk in stream:
            if chunk.text:
                full_response += chunk.text
                if callback:
                    callback(chunk.text)
        
        logger.info("Streaming complete. Validating response...")
        
        document = ExtractedDocument.model_validate_json(full_response)
        
        logger.info(
            f"Extracted: {len(document.sections)} sections, "
            f"{len(document.tables)} tables, "
            f"{len(document.images)} images"
        )
        
        return document
