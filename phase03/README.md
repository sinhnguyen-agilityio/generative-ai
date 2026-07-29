# Customer Support Analyzer

An AI-powered customer support ticket analysis pipeline built with **LangChain LCEL**.

The application receives raw customer tickets and processes them through a secure, observable AI workflow:

- Detect and mask sensitive information (PII)
- Analyze customer intent, sentiment, urgency, and entities
- Generate recommended responses
- Support multiple LLM providers with automatic fallback
- Trace workflow execution with Langfuse

---

# Workflow Overview

The complete workflow:

```text
Customer Ticket (raw text)
            |
            v
+--------------------------+
|    Security Pipeline     |
|                          |
| - PII Detection          |
| - Content Moderation     |
| - Sanitization           |
+--------------------------+
            |
            v
+--------------------------+
|   Initial Analysis       |
|                          |
| RunnableParallel:        |
|                          |
| - Summary                |
| - Intent                 |
| - Sentiment              |
| - Entity Extraction      |
+--------------------------+
            |
            v
+--------------------------+
| Response Analysis        |
|                          |
| - Urgency                |
| - Escalation Decision    |
| - Suggested Reply        |
+--------------------------+
            |
            v
+--------------------------+
|    Final AnalysisReport  |
+--------------------------+
```

Each phase is implemented as an LCEL workflow component and traced through Langfuse.

---

# Requirements

- Python 3.11+
- OpenAI API Key
- Gemini API Key
- Langfuse account (optional)

---

# Installation

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Mac/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Create `.env` file:

```env
OPENAI_API_KEY=<your-openai-key>
OPENAI_MODEL=gpt-4o-mini

GENAI_API_KEY=<your-gemini-key>
GENAI_MODEL=gemini-3.6-flash

LANGFUSE_PUBLIC_KEY=<your-public-key>
LANGFUSE_SECRET_KEY=<your-secret-key>
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

# Run Pipeline Test

The project does not include a CLI.

Use a test script to execute the workflow with sample tickets.

Example:

```bash
python -m tests.customer_support_workflow
```
