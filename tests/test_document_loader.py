import pytest
import os
from src.document_loader import DocumentLoader

def test_get_file_extension():
    loader = DocumentLoader()
    assert loader._get_file_extension("notes.pdf") == ".pdf"
    assert loader._get_file_extension("SLIDES.PPTX") == ".pptx"
    # TODO: Add more test cases

def test_supported_extensions():
    loader = DocumentLoader()
    assert '.pdf' in loader.supported_extensions