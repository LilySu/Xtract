from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

class Chunker:
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize chunker with configuration from settings file."""
        self.config = self._load_config(config_path)
        chunking_config = self.config.get("chunking", {})
        self.strategy = chunking_config.get("strategy", "page")
        self.max_chars = chunking_config.get("max_chars", 1000)
        self.overlap = chunking_config.get("overlap", 100)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            # Fallback to default config if file doesn't exist
            return {
                "chunking": {
                    "strategy": "page",
                    "max_chars": 1000,
                    "overlap": 100
                }
            }
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def chunk(self, paragraphs: List[Dict], tables: Optional[List[Dict]] = None) -> List[str]:
        """Chunk documents based on configured strategy."""
        if self.strategy == "page":
            return self._chunk_by_page(paragraphs, tables)
        return self._chunk_by_chars(paragraphs)
    
    def _chunk_by_page(self, paragraphs: List[Dict], tables: Optional[List[Dict]] = None) -> List[str]:
        """Chunk by page boundaries with signature page handling."""
        if not paragraphs:
            return []
        
        chunks = []
        pages = {}
        signature_pages = set()
        
        # First pass: identify signature pages
        signature_keywords = [
            "IN WITNESS WHEREOF",
            "AGREED TO AND ACCEPTED",
            "executed this",
            "have caused this",
            "/s/",
            "Signature:",
            "By:",
            "Name:",
            "Title:"
        ]
        
        for para in paragraphs:
            text = para.get("text", "")
            page_num = para.get("page_number", 0)
            
            # Check if this paragraph contains signature keywords
            if any(keyword in text for keyword in signature_keywords):
                signature_pages.add(page_num)
        
        # Group paragraphs by page
        for para in paragraphs:
            page_num = para.get("page_number", 0)
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(para.get("text", ""))
        
        # Add tables if provided
        if tables:
            for table in tables:
                page_num = table.get("page_number", 0)
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append(table.get("text", ""))
        
        # Create chunks, combining first page with signature pages if found
        page_nums = sorted(pages.keys())
        
        if signature_pages and self.config.get("chunking", {}).get("include_signatures", True):
            # Combine first page (usually has party definitions) with signature pages
            first_page_text = " ".join(pages.get(page_nums[0], []))
            signature_text = " ".join(
                " ".join(pages.get(sig_page, [])) 
                for sig_page in sorted(signature_pages)
            )
            
            # Create a combined chunk for extraction
            combined_chunk = first_page_text + " " + signature_text
            if combined_chunk.strip():
                chunks.append(combined_chunk)
            
            # Add remaining pages as separate chunks
            for page_num in page_nums[1:]:
                if page_num not in signature_pages:
                    chunk_text = " ".join(pages[page_num])
                    if chunk_text.strip():
                        chunks.append(chunk_text)
        else:
            # Standard page-by-page chunking
            for page_num in page_nums:
                chunk_text = " ".join(pages[page_num])
                if chunk_text.strip():
                    chunks.append(chunk_text)
        
        return chunks
    
    def _chunk_by_chars(self, paragraphs: List[Dict]) -> List[str]:
        """Chunk by character count with overlap."""
        if not paragraphs:
            return []
        
        # Combine all text
        full_text = " ".join([p.get("text", "") for p in paragraphs])
        
        if not full_text:
            return []
        
        chunks = []
        start = 0
        text_length = len(full_text)
        
        while start < text_length:
            end = min(start + self.max_chars, text_length)
            chunk = full_text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            # Move start position with overlap
            if end < text_length:
                start = end - self.overlap
            else:
                start = end
        
        return chunks