# Self-Correcting Research Assistant

An intelligent research assistant powered by OpenAI's GPT models and LangChain that performs autonomous web research using the ReAct (Reasoning + Acting) framework. The assistant can iteratively refine its research, validate information, and provide sources for its answers.

## Features

- **ReAct Framework**: Uses chain-of-thought reasoning combined with tool execution for transparent decision-making
- **Web Search Integration**: Leverages DuckDuckGo for real-time web searches
- **Streaming Output**: Real-time visualization of agent reasoning and execution steps
- **Langfuse Tracing**: Complete observability with token usage, latency, and execution traces
- **Self-Correcting**: Iteratively improves answers through reasoning and tool feedback
- **Source Attribution**: Tracks and returns sources used in the research

## Prerequisites

Before you begin, ensure you have:

- Python 3.8 or higher
- An OpenAI API key (for GPT models)
- Internet connection (for web searches and LLM API calls)
- Basic knowledge of Python and command-line usage

## Setup Instructions

### 1. Clone/Download the Project

```bash
cd /path/to/phase02/self_correcting_research_assistant
```

### 2. Create a Virtual Environment

Create an isolated Python environment to manage dependencies:

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**

- `langchain-core`: Core LangChain framework for building agents
- `langchain-community`: Community tools and integrations
- `langchain-openai`: OpenAI LLM integration
- `langfuse`: Observability and tracing
- `python-dotenv`: Environment variable management

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

**To get your OpenAI API key:**

1. Visit https://platform.openai.com/api-keys
2. Create a new API key
3. Copy the key and paste it in the `.env` file

> ⚠️ **Important**: Never commit the `.env` file to version control. It's already in `.gitignore`.

## Running the Tool

### Interactive Mode

Start the assistant in interactive mode:

```bash
python -m main
```

**Usage Example:**

```
============================================================
Self-Correcting Research Assistant
============================================================

Question (or 'exit'): What are the latest developments in quantum computing?

============================================================
Question: What are the latest developments in quantum computing?
============================================================

------------------------------
Action
------------------------------
duckduckgo_search

Action Input
------------------------------
{'query': 'latest developments quantum computing 2024'}

Observation
------------------------------
[Search results...]

==============================
Final Answer
==============================
[The assistant's answer with findings]

Question (or 'exit'): exit
```

## Workflow Documentation

### Agent Execution Flow

The Self-Correcting Research Assistant follows the ReAct (Reasoning + Acting) framework:

```
1. User Question
   ↓
2. Agent Thinks
   - Analyzes the question
   - Determines if additional information is needed
   ↓
3. Agent Acts (Tool Execution)
   - Executes search using DuckDuckGo
   - Gathers research data
   ↓
4. Agent Observes (Feedback)
   - Reviews search results
   - Evaluates completeness of information
   ↓
5. Iterate (Self-Correction)
   - If more information needed → Go to Step 2
   - Otherwise → Go to Step 6
   ↓
6. Final Answer
   - Provides comprehensive answer
   - Includes sources and citations
```

### Key Components

#### **agent.py** - Core Research Agent

- `ResearchAgent` class: Manages the agent lifecycle and execution
- `invoke()`: Single execution returning final answer
- `stream()`: Streaming execution for real-time visualization
- `build_agent()`: Factory function to create configured agent

#### **llm.py** - Language Model Configuration

- Initializes OpenAI ChatGPT model
- Configuration: `gpt-5.4-nano` with temperature=0 (deterministic)
- API authentication via environment variables

#### **tool.py** - Tool Integration

- Provides DuckDuckGo search tool for web research
- Extensible for additional tools (calculators, APIs, databases)

#### **prompt.py** - ReAct Prompt Template

- Defines the system prompt for agent behavior
- Instructs the agent to use Thought/Action/Observation pattern
- Guides final answer generation

#### **callbacks.py** - Event Logging

- `ReActLogger`: Pretty-prints agent execution events
- Formats tool calls, observations, and final answers
- Enables real-time monitoring of agent reasoning

#### **trace.py** - Observability

- Integrates Langfuse for complete execution tracing
- Tracks: LLM calls, tool usage, token counts, latency
- Provides debugging and performance analysis

#### **config.py** - Configuration Management

- Loads environment variables via python-dotenv
- Centralizes configuration (API keys, model settings)
- Validates required settings on startup

#### **schemas.py** - Data Models

- `Source`: Represents information sources (URLs, citations)
- `AgentResponse`: Structured response with answer and sources

### Execution Steps Explained

1. **Input Reception**: User submits a research question
2. **ReAct Thinking**: Agent analyzes the question and decides on actions
3. **Tool Invocation**: Agent calls DuckDuckGo search with relevant query
4. **Observation Processing**: Results are evaluated and processed
5. **Iteration Decision**: Agent determines if additional searches are needed
6. **Answer Generation**: Final comprehensive answer is synthesized
7. **Output Streaming**: Results are printed in real-time with proper formatting

## Project Structure

```
self_correcting_research_assistant/
├── main.py                 # Entry point - interactive CLI
├── agent.py               # Core ResearchAgent class
├── llm.py                 # LLM initialization (OpenAI)
├── tool.py                # Tool definitions (DuckDuckGo)
├── prompt.py              # ReAct system prompt
├── callbacks.py           # Event logging and formatting
├── trace.py               # Langfuse observability setup
├── config.py              # Configuration and settings
├── schemas.py             # Pydantic data models
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
└── README.md              # This file
```
