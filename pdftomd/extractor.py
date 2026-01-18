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
    
    # ===========================================
    # Context Caching Methods for Large Documents
    # ===========================================
    
    def _upload_to_file_api(self, pdf_path: Union[str, Path]) -> str:
        """
        Upload a PDF to Gemini's File API.
        
        Returns:
            The file URI for use in caching.
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Uploading {pdf_path.name} to File API...")
        
        file = self.client.files.upload(file=str(pdf_path))
        
        logger.info(f"Uploaded: {file.name} (URI: {file.uri})")
        return file.uri
    
    def _create_cache(self, file_uri: str, cache_ttl: str = "3600s"):
        """
        Create a cache with the uploaded PDF for multiple queries.
        
        Args:
            file_uri: URI from File API upload.
            cache_ttl: Time-to-live for cache (e.g., "3600s" = 1 hour).
            
        Returns:
            The cached content object.
        """
        logger.info(f"Creating cache with TTL={cache_ttl}...")
        
        cached_content = self.client.caches.create(
            model=self.model,
            config=types.CreateCachedContentConfig(
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=file_uri,
                                mime_type="application/pdf"
                            )
                        ]
                    )
                ],
                system_instruction="You are an expert document analyzer. Extract information accurately and completely.",
                display_name="PDF Extraction Cache",
                ttl=cache_ttl
            )
        )
        
        logger.info(f"Cache created: {cached_content.name}")
        return cached_content
    
    def _query_cached(self, prompt: str, cached_content, response_schema=None, max_retries: int = 5, json_mode: bool = False) -> dict:
        """
        Make a query against cached content with retry logic.
        
        Args:
            prompt: The extraction prompt.
            cached_content: The cache object.
            response_schema: Optional Pydantic model for structured output.
            max_retries: Maximum retry attempts (default: 5 for robustness).
            json_mode: If True, forces JSON response format.
            
        Returns:
            Dict with response text, token usage, and finish reason.
        """
        import time
        
        config = {
            "cached_content": cached_content.name,
            "temperature": self.temperature,
            "max_output_tokens": 65536,
        }
        
        # Enable JSON mode if requested or if using response_schema
        if json_mode or response_schema:
            config["response_mime_type"] = "application/json"
        
        if response_schema:
            config["response_schema"] = response_schema
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt
                    logger.info(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                
                # Check finish reason
                finish_reason = "UNKNOWN"
                if response.candidates and len(response.candidates) > 0:
                    finish_reason = str(response.candidates[0].finish_reason or "STOP")
                    
                    # Handle RECITATION - retry
                    if "RECITATION" in finish_reason:
                        logger.warning(f"  RECITATION detected, retrying...")
                        continue
                    
                    # Handle SAFETY - retry
                    if "SAFETY" in finish_reason:
                        logger.warning(f"  SAFETY filter triggered, retrying...")
                        continue
                
                # Check for empty response
                if response.text is None or response.text.strip() == "":
                    if attempt < max_retries - 1:
                        logger.warning(f"  Empty response, retrying...")
                        continue
                    else:
                        logger.warning(f"  Empty response after {max_retries} attempts")
                
                # Get token usage
                token_usage = {"prompt": 0, "completion": 0, "total": 0, "cached": 0}
                if response.usage_metadata:
                    usage = response.usage_metadata
                    token_usage = {
                        "prompt": usage.prompt_token_count or 0,
                        "completion": usage.candidates_token_count or 0,
                        "total": usage.total_token_count or 0,
                        "cached": getattr(usage, 'cached_content_token_count', 0) or 0,
                    }
                
                return {
                    "text": response.text or "",
                    "token_usage": token_usage,
                    "finish_reason": finish_reason,
                }
                
            except Exception as e:
                logger.warning(f"  Query attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        
        # Fallback - should not reach here
        return {"text": "", "token_usage": {"prompt": 0, "completion": 0, "total": 0, "cached": 0}, "finish_reason": "FAILED"}
    
    def extract_large_document(
        self,
        pdf_path: Union[str, Path],
        cache_ttl: str = "3600s",
    ) -> ExtractionResult:
        """
        Extract content from a large PDF using Intelligent Context-Aware Chunking.
        
        This method uses a 3-phase approach:
        1. STRUCTURAL ANALYSIS: Analyze document structure and get chunking suggestions
        2. CONTEXT-AWARE EXTRACTION: Extract each chunk with global context
        3. INTELLIGENT ASSEMBLY: Combine chunks maintaining coherence
        
        Args:
            pdf_path: Path to the PDF file.
            cache_ttl: Cache time-to-live (default: 1 hour).
            
        Returns:
            ExtractionResult with document and combined token usage.
        """
        from pdftomd.models import (
            DocumentMetadata, Section, Table, Image,
            CodeBlock, Equation, Reference, DocumentAnalysis
        )
        import json
        
        pdf_path = Path(pdf_path)
        logger.info(f"[LARGE DOC] Starting intelligent extraction for: {pdf_path.name}")
        
        # Step 1: Upload to File API
        file_uri = self._upload_to_file_api(pdf_path)
        
        # Step 2: Create cache
        cached_content = self._create_cache(file_uri, cache_ttl)
        
        total_tokens = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        
        # ========================================
        # PHASE 1: STRUCTURAL ANALYSIS
        # ========================================
        logger.info("[PHASE 1] Analyzing document structure...")
        
        analysis_result = self._query_cached(
            """Analyze this document and return a JSON object for chunked extraction.

Return this exact JSON structure (fill in the values):
{
    "title": "The document title",
    "document_type": "textbook chapter",
    "total_pages": 20,
    "language": "English",
    "sections_outline": ["Section 1 Title", "Section 2 Title", "Section 3 Title"],
    "global_context": "A summary paragraph describing the document purpose and main topics.",
    "suggested_chunks": [
        {"chunk_id": "chunk_1", "start_page": 1, "end_page": 7, "content_type": "introduction", "section_titles": ["Section 1"], "has_tables": false, "has_figures": false, "has_equations": false, "has_code": false},
        {"chunk_id": "chunk_2", "start_page": 8, "end_page": 15, "content_type": "main_content", "section_titles": ["Section 2"], "has_tables": true, "has_figures": true, "has_equations": false, "has_code": false},
        {"chunk_id": "chunk_3", "start_page": 16, "end_page": 20, "content_type": "conclusion", "section_titles": ["Section 3"], "has_tables": false, "has_figures": false, "has_equations": false, "has_code": false}
    ]
}

Create 2-5 chunks covering all pages. Each chunk should cover logically related sections.""",
            cached_content,
            json_mode=True  # Force JSON response format
        )
        
        total_tokens.prompt_tokens += analysis_result["token_usage"]["prompt"]
        total_tokens.completion_tokens += analysis_result["token_usage"]["completion"]
        logger.info(f"  Tokens: {analysis_result['token_usage']}")
        
        # Parse analysis
        try:
            analysis_raw = json.loads(analysis_result["text"])
            analysis = DocumentAnalysis.model_validate(analysis_raw)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse structural analysis: {e}")
            raise ValueError(f"Structural analysis failed: {e}")
        
        logger.info(f"  Document: {analysis.title}")
        logger.info(f"  Type: {analysis.document_type}, Pages: {analysis.total_pages}")
        logger.info(f"  Chunks suggested: {len(analysis.suggested_chunks)}")
        
        # ========================================
        # PHASE 2: CONTEXT-AWARE CHUNK EXTRACTION
        # ========================================
        all_sections = []
        all_tables = []
        all_images = []
        all_code_blocks = []
        all_equations = []
        all_references = []
        
        for i, chunk in enumerate(analysis.suggested_chunks):
            logger.info(f"[PHASE 2] Extracting chunk {i+1}/{len(analysis.suggested_chunks)}: {chunk.chunk_id}")
            logger.info(f"  Pages {chunk.start_page}-{chunk.end_page}: {chunk.content_type}")
            logger.info(f"  Sections: {', '.join(chunk.section_titles[:3])}...")
            
            # Build context-aware prompt
            chunk_prompt = f"""DOCUMENT GLOBAL CONTEXT:
{analysis.global_context}

FULL DOCUMENT STRUCTURE:
{chr(10).join(analysis.sections_outline)}

YOUR TASK:
Extract ALL content from pages {chunk.start_page} to {chunk.end_page}.
This chunk covers: {', '.join(chunk.section_titles)}
Content type: {chunk.content_type}

Extract and return as JSON:
{{
    "sections": [
        {{"title": "Section Title", "level": 1-6, "content": "Complete markdown text..."}}
    ],
    "tables": [
        {{"caption": "Table caption", "headers": ["Col1", "Col2"], "rows": [["data", "data"]], "context": "What this table shows"}}
    ],
    "images": [
        {{"figure_number": "Figure 1", "caption": "Caption", "description": "Detailed description", "alt_text": "Alt text"}}
    ],
    "code_blocks": [
        {{"language": "python", "code": "code here", "context": "What this code does"}}
    ],
    "equations": [
        {{"equation_number": "Eq. 1", "latex": "LaTeX formula", "description": "What this equation represents"}}
    ],
    "references": [
        {{"number": "1", "citation": "Full citation text"}}
    ]
}}

IMPORTANT:
- Extract COMPLETE text for each section, not summaries
- Preserve all formatting (headers, lists, emphasis)
- Include ALL tables, figures, equations in this page range
- If an element type doesn't exist in these pages, return empty array"""
            
            chunk_result = self._query_cached(chunk_prompt, cached_content, json_mode=True)
            
            total_tokens.prompt_tokens += chunk_result["token_usage"]["prompt"]
            total_tokens.completion_tokens += chunk_result["token_usage"]["completion"]
            logger.info(f"  Tokens: {chunk_result['token_usage']}")
            
            # Parse chunk results
            try:
                if chunk_result["text"]:
                    chunk_data = json.loads(chunk_result["text"])
                    
                    # Accumulate results
                    for s in chunk_data.get("sections", []):
                        all_sections.append(Section(**s))
                    for t in chunk_data.get("tables", []):
                        all_tables.append(Table(**t))
                    for img in chunk_data.get("images", []):
                        all_images.append(Image(**img))
                    for c in chunk_data.get("code_blocks", []):
                        all_code_blocks.append(CodeBlock(**c))
                    for eq in chunk_data.get("equations", []):
                        all_equations.append(Equation(**eq))
                    for ref in chunk_data.get("references", []):
                        all_references.append(Reference(**ref))
                    
                    logger.info(f"  Extracted: {len(chunk_data.get('sections', []))} sections")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"  Failed to parse chunk {chunk.chunk_id}: {e}")
                # Continue with other chunks
        
        # ========================================
        # PHASE 3: INTELLIGENT ASSEMBLY
        # ========================================
        logger.info("[PHASE 3] Assembling final document...")
        
        # Calculate total tokens
        total_tokens.total_tokens = total_tokens.prompt_tokens + total_tokens.completion_tokens
        
        # Build metadata from analysis
        metadata = DocumentMetadata(
            title=analysis.title,
            document_type=analysis.document_type,
            total_pages=analysis.total_pages,
            language=analysis.language,
        )
        
        # Build document
        document = ExtractedDocument(
            metadata=metadata,
            summary=analysis.global_context,  # Use global context as summary
            key_points=[],  # Will be populated from sections
            sections=all_sections,
            tables=all_tables,
            images=all_images,
            code_blocks=all_code_blocks,
            equations=all_equations,
            references=all_references,
        )
        
        logger.info(
            f"[COMPLETE] Large document extraction finished: "
            f"{len(document.sections)} sections, {len(document.tables)} tables, "
            f"{len(document.images)} images, {len(document.equations)} equations"
        )
        logger.info(f"Total tokens - {total_tokens}")
        
        return ExtractionResult(
            document=document,
            token_usage=total_tokens,
            finish_reason="STOP",
            was_truncated=False,
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
