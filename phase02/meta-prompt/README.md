# Meta-Prompt: Prompt Optimization Pipeline

A two-stage LLM pipeline that improves user queries into high-quality prompts, then executes them for optimal results.

**How it works:**

1. **Refiner Stage**: Analyzes your question and rewrites it into a structured, clear prompt using prompt engineering best practices
2. **Executor Stage**: Executes the refined prompt to get a concise, high-quality answer

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Create `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Run the Tool

```bash
python main.py
```

**Usage:**

```
User question: What are the benefits of machine learning in healthcare?

[Refined prompt and answer output]

User question: exit
```

## Project Structure

```
meta-prompt/
├── main.py           # Entry point - interactive CLI
├── pipeline.py       # Orchestrates refiner and executor
├── meta_refiner.py   # Improves queries into prompts
├── executor.py       # Executes refined prompts
├── prompt.py         # Meta-prompt engineering template
├── config.py         # Configuration management
├── tracer.py         # Langfuse observability (optional)
├── llm.py            # OpenAI integration
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Workflow

```
User Input
    ↓
MetaRefiner (Prompt Enhancement)
    • Analyzes the question
    • Adds structure and clarity
    • Applies prompt engineering best practices
    ↓
Executor (Answer Generation)
    • Receives refined prompt
    • Generates concise answer (max 200 words)
    ↓
Final Output
```

## Key Components

| Component        | Role                                                    |
| ---------------- | ------------------------------------------------------- |
| **MetaRefiner**  | Transforms vague queries into structured, clear prompts |
| **Executor**     | Runs the refined prompt and generates concise answers   |
| **MetaPipeline** | Coordinates the two stages seamlessly                   |
| **Config**       | Manages API keys and model settings                     |
