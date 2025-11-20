"""
Knowledge Base Management Module for Disaster Literacy RAG System
Handles PDF processing (text extraction + OCR), chunking, metadata extraction
References: Lines 25-41 (KB structure), Lines 209-231 (Admin KB Updates)
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib

# PDF Processing - References: Lines 29-33
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

from config import (
    KB_DOCUMENTS_DIR,
    METADATA_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    TESSERACT_CMD,
    POPPLER_PATH,
    OCR_LANGUAGE,
    DPI,
    ENABLE_VERSION_TRACKING,
    VERSION_TIMESTAMP_FORMAT
)
from error_handler import error_handler


# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class KBManager:
    """
    Manages knowledge base document processing and metadata
    References: Lines 25-41, 209-231
    """
    
    def __init__(self):
        self.kb_dir = Path(KB_DOCUMENTS_DIR)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = Path(METADATA_FILE)
        self.metadata = self._load_metadata()
    
    def check_duplicate(self, file_path: Path) -> Dict[str, Any]:
        """
        Check if a document is a duplicate based on filename and content hash
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with:
                - is_duplicate: bool
                - reason: str (if duplicate)
                - doc_id: str
                - existing_metadata: Dict (if duplicate)
        """
        # Generate document ID from filename
        doc_id = hashlib.md5(file_path.name.encode()).hexdigest()
        
        # Calculate content hash upfront
        content_hash = self._calculate_file_hash(file_path)
        
        # Check if document with same filename exists
        if doc_id in self.metadata.get("documents", {}):
            existing_doc = self.metadata["documents"][doc_id]
            
            if "content_hash" not in existing_doc:
                # Old documents without hash - consider duplicate by filename only
                return {
                    "is_duplicate": True,
                    "reason": f"Document with filename '{file_path.name}' already exists",
                    "doc_id": doc_id,
                    "existing_metadata": existing_doc
                }
            
            # Compare content hashes
            if existing_doc.get("content_hash") == content_hash:
                return {
                    "is_duplicate": True,
                    "reason": f"Document with same content already exists (uploaded as '{file_path.name}')",
                    "doc_id": doc_id,
                    "existing_metadata": existing_doc
                }
        
        # Check all other documents for content hash match (different filename, same content)
        for existing_doc_id, existing_doc in self.metadata.get("documents", {}).items():
            if existing_doc_id == doc_id:
                continue  # Already checked this one above
            
            if existing_doc.get("content_hash") == content_hash:
                return {
                    "is_duplicate": True,
                    "reason": f"Document with same content already exists as '{existing_doc['source']}'",
                    "doc_id": doc_id,
                    "existing_metadata": existing_doc
                }
        
        return {
            "is_duplicate": False,
            "doc_id": doc_id
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file content for duplicate detection
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load existing KB metadata
        References: Lines 36, 40 (Metadata tracking)
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                error_handler.logger.warning(f"Failed to load metadata: {e}")
                return {"documents": {}, "last_updated": None}
        return {"documents": {}, "last_updated": None}
        
    def _save_metadata(self) -> None:
        """
        Save KB metadata with version tracking
        References: Lines 225-227 (Version control with timestamps)
        """
        if ENABLE_VERSION_TRACKING:
            self.metadata["last_updated"] = datetime.now().strftime(VERSION_TIMESTAMP_FORMAT)
            
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
    def process_document(
        self,
        file_path: str,
        disaster_type: Optional[str] = None,
        region: Optional[str] = None,
        date: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point for document processing
        References: Lines 215-224 (Manual upload and automatic processing)
        
        Args:
            file_path: Path to the document file
            disaster_type: Type of disaster (tsunami, flood, cyclone, etc.)
            region: Geographic region
            date: Document date
            force: If True, bypass duplicate detection and force reprocessing
            
        Returns:
            Dictionary with:
                - success: bool
                - chunks: List of processed chunks (if success)
                - message: str (status message)
                - is_duplicate: bool
                - doc_id: str (document ID)
        """
        file_path = Path(file_path)
        
        # Check for duplicates unless force is True
        if not force:
            duplicate_check = self.check_duplicate(file_path)
            if duplicate_check["is_duplicate"]:
                error_handler.logger.warning(
                    f"Duplicate document detected: {file_path.name} - {duplicate_check['reason']}"
                )
                return {
                    "success": False,
                    "is_duplicate": True,
                    "doc_id": duplicate_check["doc_id"],
                    "message": f"Duplicate document: {duplicate_check['reason']}",
                    "existing_metadata": duplicate_check.get("existing_metadata")
                }
        
        error_handler.logger.info(f"Processing document: {file_path.name}")
        
        # Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text, is_scanned = self._extract_pdf_text(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            text = self._extract_text_file(file_path)
            is_scanned = False
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        # Clean extracted text - References: Lines 32-33
        text = self._clean_text(text)
        
        # Create chunks - References: Lines 34-36
        chunks = self._chunk_text(text, file_path.name)
        
        # Auto-detect disaster type if not provided
        if not disaster_type:
            disaster_type = self._detect_disaster_type(text, file_path.name)
        
        # Calculate content hash for duplicate detection
        content_hash = self._calculate_file_hash(file_path)
        
        # Add metadata - References: Line 36
        doc_metadata = {
            "source": file_path.name,
            "disaster_type": disaster_type or "general",
            "region": region or "not specified",
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "processed_at": datetime.now().strftime(VERSION_TIMESTAMP_FORMAT),
            "is_scanned": is_scanned,
            "chunk_count": len(chunks),
            "content_hash": content_hash
        }
        
        # Attach metadata to each chunk
        for chunk in chunks:
            chunk.update({
                "source": doc_metadata["source"],
                "disaster_type": doc_metadata["disaster_type"],
                "region": doc_metadata["region"],
                "date": doc_metadata["date"]
            })
            
        # Update global metadata
        doc_id = hashlib.md5(file_path.name.encode()).hexdigest()
        self.metadata["documents"][doc_id] = doc_metadata
        self._save_metadata()
        
        error_handler.logger.info(
            f"Successfully processed {file_path.name}: {len(chunks)} chunks created"
        )
        
        return {
            "success": True,
            "is_duplicate": False,
            "doc_id": doc_id,
            "chunks": chunks,
            "message": f"Successfully processed {len(chunks)} chunks"
        }
        
    def _extract_pdf_text(self, pdf_path: Path) -> Tuple[str, bool]:
        """
        Extract text from PDF with OCR fallback
        References: Lines 29-33 (PDF handling with text extraction and OCR)
        
        Returns:
            Tuple of (extracted_text, is_scanned)
        """
        text = ""
        is_scanned = False
        
        # First, try direct text extraction - References: Line 30
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            error_handler.logger.warning(f"pdfplumber failed: {e}")
            
        # If no text extracted, PDF might be scanned - apply OCR
        # References: Lines 31-32 (OCR for scanned PDFs)
        if len(text.strip()) < 100:  # Threshold for considering PDF as scanned
            error_handler.logger.info(f"Applying OCR to {pdf_path.name}")
            is_scanned = True
            text = self._ocr_pdf(pdf_path)
            
        return text, is_scanned
        
    def _ocr_pdf(self, pdf_path: Path) -> str:
        """
        Apply OCR to scanned PDF page by page to save memory and provide feedback
        References: Line 31 (Tesseract via pytesseract for OCR)
        """
        text = ""
        
        try:
            error_handler.logger.info(f"Starting OCR for {pdf_path.name}...")
            
            # Check if Poppler path exists
            if not Path(POPPLER_PATH).exists():
                error_handler.logger.error(f"Poppler path not found at: {POPPLER_PATH}")
                raise FileNotFoundError(f"Poppler tools not found at {POPPLER_PATH}. Please install Poppler or update config.")

            # Get total pages first using pdfplumber (fast)
            total_pages = 0
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    total_pages = len(pdf.pages)
            except Exception as e:
                error_handler.logger.warning(f"Could not get page count with pdfplumber: {e}. Fallback to processing without count.")
            
            if total_pages > 0:
                error_handler.logger.info(f"PDF has {total_pages} pages. Starting page-by-page OCR...")
                
                for i in range(total_pages):
                    page_num = i + 1
                    try:
                        error_handler.logger.info(f"Processing page {page_num}/{total_pages} for {pdf_path.name}...")
                        
                        # Convert single page to image
                        # first_page and last_page are 1-indexed in pdf2image
                        images = convert_from_path(
                            str(pdf_path),
                            dpi=DPI,
                            first_page=page_num,
                            last_page=page_num,
                            poppler_path=POPPLER_PATH
                        )
                        
                        if not images:
                            error_handler.logger.warning(f"No image generated for page {page_num}")
                            continue
                            
                        image = images[0]
                        
                        # Apply OCR
                        page_text = pytesseract.image_to_string(
                            image,
                            lang=OCR_LANGUAGE,
                            config='--psm 1'
                        )
                        text += f"\n--- Page {page_num} ---\n{page_text}\n"
                        
                        # Explicitly clear image from memory
                        del image
                        del images
                        
                    except Exception as page_error:
                        error_handler.logger.error(f"OCR failed for page {page_num} of {pdf_path.name}: {page_error}")
                        text += f"\n--- Page {page_num} (OCR Failed) ---\n[Error extracting text]\n"
            else:
                # Fallback if page count failed (original behavior but with better logging)
                error_handler.logger.info(f"Converting entire PDF to images (page count unknown)...")
                images = convert_from_path(
                    str(pdf_path),
                    dpi=DPI,
                    poppler_path=POPPLER_PATH
                )
                
                total_pages = len(images)
                error_handler.logger.info(f"PDF converted to {total_pages} images. Starting text extraction...")
                
                for i, image in enumerate(images):
                    try:
                        error_handler.logger.info(f"OCR Processing page {i+1}/{total_pages} for {pdf_path.name}")
                        page_text = pytesseract.image_to_string(
                            image,
                            lang=OCR_LANGUAGE,
                            config='--psm 1'
                        )
                        text += f"\n--- Page {i+1} ---\n{page_text}\n"
                    except Exception as page_error:
                        error_handler.logger.error(f"OCR failed for page {i+1} of {pdf_path.name}: {page_error}")
                        text += f"\n--- Page {i+1} (OCR Failed) ---\n[Error extracting text]\n"

        except Exception as e:
            error_handler.logger.error(f"OCR failed for {pdf_path.name}: {e}")
            raise
            
        return text
        
    def _extract_text_file(self, file_path: Path) -> str:
        """
        Extract text from plain text files
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
            
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text
        References: Lines 32-33 (Remove headers, footers, page numbers, whitespace)
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\b(?:Page|PAGE)\s+\d+\b', '', text)
        text = re.sub(r'\b\d+\s+of\s+\d+\b', '', text)
        
        # Remove common header/footer patterns
        text = re.sub(r'^\s*[-=_]+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove URLs and emails (optional - might want to keep some)
        # text = re.sub(r'http[s]?://\S+', '', text)
        # text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
        
    def _chunk_text(
        self,
        text: str,
        source_name: str
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        References: Lines 34-36 (Chunking with 200-800 tokens, 50-100 overlap)
        
        Note: Using word-based approximation (1 token ≈ 0.75 words)
        """
        # Approximate token count (1 token ≈ 0.75 words for English)
        words_per_chunk = int(CHUNK_SIZE * 0.75)
        words_overlap = int(CHUNK_OVERLAP * 0.75)
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_word_count + sentence_words > words_per_chunk and current_chunk:
                # Save chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= int(MIN_CHUNK_SIZE * 0.75):
                    chunks.append({
                        "chunk_id": f"{source_name}_chunk_{chunk_id}",
                        "text": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text)
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                # Keep last few sentences for overlap
                overlap_words = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_words = len(s.split())
                    if overlap_words + s_words <= words_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_words += s_words
                    else:
                        break
                        
                current_chunk = overlap_sentences
                current_word_count = overlap_words
                
            current_chunk.append(sentence)
            current_word_count += sentence_words
            
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= int(MIN_CHUNK_SIZE * 0.75):
                chunks.append({
                    "chunk_id": f"{source_name}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text)
                })
                
        return chunks
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get current KB metadata
        """
        return self.metadata
        
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all processed documents
        """
        return [
            {"id": doc_id, **doc_info}
            for doc_id, doc_info in self.metadata.get("documents", {}).items()
        ]

    def delete_document(self, doc_id: str) -> Optional[str]:
        """
        Delete a document from metadata
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            str: Source filename of the deleted document, or None if not found
        """
        if doc_id in self.metadata.get("documents", {}):
            doc_info = self.metadata["documents"].pop(doc_id)
            self._save_metadata()
            error_handler.logger.info(f"Deleted document metadata: {doc_info['source']} ({doc_id})")
            return doc_info["source"]
        
        error_handler.logger.warning(f"Document ID not found for deletion: {doc_id}")
        return None
    
    def _detect_disaster_type(self, text: str, filename: str = "") -> str:
        """
        Auto-detect disaster type from document content and filename using keyword matching
        """
        # Convert to lowercase for matching
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Define keywords for each disaster type (more specific first)
        disaster_keywords = {
            "tsunami": ["tsunami", "tidal wave", "seismic sea wave"],
            "flood": ["flood", "flooding", "inundation", "overflow", "deluge"],
            "cyclone": ["cyclone", "hurricane", "typhoon", "tropical storm"],
            "fire": ["fire", "wildfire", "forest fire", "conflagration"],
            "landslide": ["landslide", "mudslide", "rockfall", "avalanche"],
            "earthquake": ["earthquake", "tremor", "quake"]  # Less weight for "seismic"
        }
        
        # Count keyword occurrences for each disaster type
        scores = {}
        for disaster, keywords in disaster_keywords.items():
            # Check filename first (weighted heavily)
            filename_matches = sum(keyword in filename_lower for keyword in keywords)
            # Check text content
            text_matches = sum(text_lower.count(keyword) for keyword in keywords)
            
            # Filename matches get 10x weight
            total_score = (filename_matches * 10) + text_matches
            
            if total_score > 0:
                scores[disaster] = total_score
        
        # Return disaster type with highest score, or "general" if none found
        if scores:
            detected_type = max(scores, key=scores.get)
            error_handler.logger.info(f"Auto-detected disaster type: {detected_type} (filename: {filename})")
            return detected_type
        
        error_handler.logger.info("Could not auto-detect disaster type, using 'general'")
        return "general"
