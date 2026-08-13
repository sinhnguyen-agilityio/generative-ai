# Agent RAG System

A Retrieval-Augmented Generation (RAG) system implementation comparing naive and agentic approaches for question answering over HuggingFace documentation.

## Project Overview

This project implements and evaluates two different RAG strategies:

- **Naive RAG**: Single-pass retrieval followed by answer generation
- **Agentic RAG**: Multi-turn agent-based approach with dynamic document retrieval

The system uses BM25 retrieval, OpenAI's language models, and RAGAS framework for comprehensive evaluation.

## Goals of This Practice

1. **Implement Multi-Modal RAG**: Build both simple and complex RAG pipelines
2. **Learn Agent Design Patterns**: Understand how agents can iteratively refine answers through multiple retrieval cycles
3. **Practice Evaluation Frameworks**: Use RAGAS to systematically measure RAG quality metrics (correctness, relevance, etc.)
4. **Work with LLM APIs**: Integrate OpenAI API for both retrieval and answer generation
5. **Handle Asynchronous Operations**: Implement async/await patterns for efficient API calls

## Project Structure

```
agent_rag/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                   # Main entry point - runs experiments and generates comparison
├── experiment.py             # Experiment runner with naive/agentic modes
│
├── data/
│   ├── compare.csv          # Comparison results (naive vs agentic)
│   └── hf_qa_doc_eval.csv   # Dataset
│
└── rag/
    ├── __init__.py
    ├── rag.py               # RAG system implementation (naive & agentic modes)
    ├── retriever.py         # BM25 retriever for document search
    ├── dataset.py           # Dataset creation and preparation
    └── evaluation.py        # RAGAS-based evaluation metrics
```

### File Descriptions

- **main.py**: Entry point that runs both naive and agentic experiments, compares results, and saves to CSV
- **experiment.py**: Contains `run_experiment()` function that orchestrates the RAG system with dataset and evaluation
- **rag/rag.py**: Core RAG class supporting both naive and agentic modes with configurable LLM
- **rag/retriever.py**: BM25-based document retriever using LangChain and HuggingFace datasets
- **rag/dataset.py**: Creates RAGAS dataset from HuggingFace QA benchmarks
- **rag/evaluation.py**: Implements RAGAS evaluation framework for quality assessment
- **data/**: Output directory for comparison results and evaluation metrics

## Running the Project

### Prerequisites

1. **Python 3.9+**
2. **OpenAI API Key**: Required for LLM operations
3. **Dependencies**: Install packages from requirements.txt

### Setup Instructions

1. **Clone/Navigate to Project**

   ```bash
   cd agent_rag
   ```
2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Set OpenAI API Key**

   ```bash
   # On Windows (PowerShell)
   $env:OPENAI_API_KEY = "your-api-key-here"

   # On Windows (Command Prompt)
   set OPENAI_API_KEY=your-api-key-here

   # On macOS/Linux
   export OPENAI_API_KEY="your-api-key-here"
   ```

   Alternatively, create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=your-api-key-here
   ```
4. **Run the Experiment**

   ```bash
   python main.py
   ```

### What Happens When You Run

1. **Initializes RAG System**: Loads HuggingFace documentation and creates a BM25 retriever
2. **Runs Naive Mode**:
   - Retrieves relevant documents once
   - Generates answer based on retrieved context
3. **Runs Agentic Mode**:
   - Agent dynamically retrieves documents based on query
   - May perform multiple retrieval rounds for refinement
   - Generates final answer using agent reasoning
4. **Compares Results**: Saves side-by-side comparison to `data/compare.csv`

### Output

After running, check:

- `data/compare.csv`: Contains questions, expected answers, and responses from both approaches with correctness scores
