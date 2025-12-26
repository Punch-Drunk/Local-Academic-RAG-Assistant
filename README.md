# Academic RAG Assistant

## Description
A Retrieval-Augmented Generation (RAG) system that allows me to query my machine learning course materials (PDFs, slides, Jupyter notebooks) using local LLMs.

## Why I'm Building This
- To gain hands-on experience with RAG systems
- To create a practical tool for reviewing ML concepts
- To learn about document processing, embeddings, and local LLMs
- To assist me in reviewing topics and material

## Tech Stack
- **Local LLM**: Ollama + Mistral 7B (TBD)
- **Embeddings**: Sentence Transformers 
- **Vector Store**: ChromaDB
- **Document Processing**: PyMuPDF4LLM, PyMuPDF Layout, LaTeXOCR
- **Environment**: Python 3.10, Linux, RTX 3060

## Current Status
🚧 In Development - Phase 5: Integration with local LLM 

## Getting Started
1. Clone the repository
2. Set up virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Roadmap
- [X] Phase 1: Basic document loading (PDFs)
- [X] Phase 2: Intelligent LaTeX extraction
- [X] Phase 3: Chunking strategy
- [X] Phase 4: Embedding and vector store setup
- [ ] Phase 5: Integration with local LLM 
- [ ] Phase 6: Web interface with Streamlit

## Learning Goals
- [ ] Implement a complete RAG pipeline from scratch
- [ ] Work with different document formats
- [ ] Deploy a local LLM application
- [ ] Create a usable tool for my own learning

## Challenges & Solutions
