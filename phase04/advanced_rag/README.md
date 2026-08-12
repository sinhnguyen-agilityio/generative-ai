# Advanced RAG Project

This project implements an advanced retrieval-augmented generation (RAG) pipeline for answering questions using local tourism and historical knowledge. It loads documents from the Cilento dataset, chunks them for retrieval, rewrites queries, filters sensitive content, and evaluates the answer quality with a Ragas-based experiment.

## What this project does

- Ingests source documents from the local data folder
- Builds a Chroma vector store with parent/child document chunking
- Rewrites user questions before retrieval for better matching
- Uses an OpenAI model to answer using the retrieved context only
- Masks PII before sending context to the language model
- Traces the retrieval and generation flow with Langfuse
- Runs baseline evaluation against a dataset and exports results to CSV

## Project structure

- `main.py` - CLI entry point that builds the retriever and runs the evaluation experiment
- `generation/rag_chatbot.py` - chatbot logic, question rewriting, secure context processing, and answer generation
- `ingestion/ingest.py` - creates or reuses the Chroma retriever and ingests documents
- `ingestion/loader.py` - loads .txt, .pdf, and .docx files from a folder
- `ingestion/chunker.py` - chunking logic used for retrieval
- `retrieval/query_rewriter.py` - query rewrite step for more effective retrieval
- `retrieval/security.py` - context sanitization and security checks
- `evaluations/experiment.py` - evaluation experiment setup
- `evaluations/dataset.json` - evaluation questions and reference answers
- `data/` - local documents and Chroma storage
- `reports/` - generated evaluation CSV results
- `env_ad_rag/` - local Python virtual environment

## Requirements

- Python 3.12
- Access to an OpenAI API key
- Optional: Langfuse keys for tracing in the cloud dashboard
- A Windows environment is assumed here, but the same flow works on macOS/Linux with minor path changes

## Setup

1. Open a terminal in the project root.
2. Activate the virtual environment:

   PowerShell:

   ```powershell
   .\env_ad_rag\Scripts\Activate.ps1
   ```

   Command Prompt:

   ```bat
   env_ad_rag\Scripts\activate.bat
   ```

3. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root if it does not already exist, and add your secrets:

   ```env
   OPENAI_API_KEY=your_openai_key_here
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
   ```

5. Make sure the source documents are available under:

   ```text
   data/CilentoTouristInfo/
   ```

## Run the project

### 1) Build or refresh the vector store

This will ingest the documents into Chroma and create the retriever index:

```bash
python main.py --reingest
```

- `--reingest` forces a full reload of the dataset into the vector database.
- If you omit it, the project will reuse the existing Chroma index if it already exists.

### 2) Run the evaluation pipeline

The main script executes the retrieval and evaluation flow:

```bash
python main.py
```

This does the following:

- loads the evaluation dataset from `evaluations/dataset.json`
- builds the retriever
- creates a chatbot using `RAGChatbot`
- runs the baseline experiment
- saves the results to `reports/ragas_experiment_report.csv`

## Expected output

After a successful run, you should see:

- the number of evaluation samples loaded
- the experiment row count
- a CSV file created in `reports/`

Example:

```text
Testset size: 10
Experiment rows: 10
```

## Data and storage locations

- Document source folder: `data/CilentoTouristInfo/`
- Chroma persistence directory: `data/chroma/`
- Local document store: `data/docstore.db`
- Evaluation dataset: `evaluations/dataset.json`
- Generated report: `reports/ragas_experiment_report.csv`

## Common notes

- If you change the data files in `data/CilentoTouristInfo/`, run the project with `--reingest` so the vector store is rebuilt.
- If the OpenAI model or Langfuse connection fails, verify that the `.env` file has valid keys and that the required packages are installed.
- The project is designed as a research/evaluation pipeline rather than a browser app, so the main execution path is via the Python CLI.

## Useful commands

```bash
# Activate virtual environment
.\env_ad_rag\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Rebuild the Chroma index
python main.py --reingest

# Run the baseline RAG evaluation
python main.py
```

## License

This project is intended for educational and experimental use within the training repository.
