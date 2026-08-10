# Advanced RAG Application

This workspace contains a complete advanced RAG Q&A prototype built with Python, custom chunking and retrieval components, and a Streamlit interface.

## Features

- Document ingestion from the documents folder
- Context-aware chunking with metadata
- Metadata-aware retrieval and question transformation
- Simple answer generation with a grounded context passage
- Streamlit web UI with an evaluation report panel
- Regression test coverage for the core pipeline

## Project structure

- app.py: Streamlit UI
- src/advanced_rag/pipeline.py: orchestration of ingestion, chunking, retrieval, and evaluation
- src/advanced_rag/components/: modular components for loading, chunking, retrieval, LLM response generation, and evaluation
- tests/test_pipeline.py: regression test for the ingestion and answering flow
- documents/: place your source documents here
- run_app.bat: convenience launcher for Windows

## Getting started

1. Activate the virtual environment in env_ad_rag
2. Install the required dependencies from requirements.txt
3. Put your documents in the documents folder
4. Launch the app with: streamlit run app.py
5. Click Build index, then ask a question
