import os
from typing import List, Dict, Optional
import logging
import pymupdf

class DocumentLoader:
    def __init__(self):
        self.supported_extensions = ['.pdf']  # We'll add more later
        self.logger = logging.getLogger(__name__)
    
    def load_directory(self, directory_path: str) -> List[Dict]:
        """Load all supported documents from a directory"""
        # TODO: Implement
        pass
    
    def _load_pdf(self, file_path: str) -> Optional[str]:
        """Extract text from a PDF file"""
        # TODO: Implement using PyMuPDF
        pass
    
    def _get_file_extension(self, filename: str) -> str:
        """Helper to get lowercase file extension"""
        # TODO: Implement
        pass