# Academic RAG Assistant

## Description
A Retrieval-Augmented Generation (RAG) system that allows me to query my machine learning course materials (PDFs, slides, Jupyter notebooks) using local LLMs.

## Why I'm Building This
- To gain hands-on experience with RAG systems
- To create a practical tool for reviewing ML concepts
- To learn about document processing, embeddings, and local LLMs
- Portfolio project post-bachelor's degree

## Tech Stack
- **Local LLM**: Ollama + Mistral 7B
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **Document Processing**: PyMuPDF, custom chunking strategies
- **UI**: Streamlit (planned)
- **Environment**: Python 3.10, Linux, RTX 4060

## Current Status
🚧 In Development - Phase 0: Project Setup

## Getting Started
1. Clone the repository
2. Set up virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Roadmap
- [ ] Phase 1: Basic document loading (PDFs)
- [ ] Phase 2: Chunking strategy for academic content
- [ ] Phase 3: Embedding and vector store setup
- [ ] Phase 4: Integration with local LLM (Ollama)
- [ ] Phase 5: Web interface with Streamlit
- [ ] Phase 6: Special handling for equations and code

## Learning Goals
- [ ] Implement a complete RAG pipeline from scratch
- [ ] Work with different document formats
- [ ] Optimize chunking for academic content
- [ ] Deploy a local LLM application
- [ ] Create a usable tool for my own learning

## Challenges & Solutions
