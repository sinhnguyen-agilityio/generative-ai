# Multi-Part Planner

This project uses an LLM to generate, evaluate, and select software design approaches. It is built around a small Python workflow that sends prompts to OpenAI and optionally records traces with Langfuse.

## Prerequisites

Before you begin, make sure you have:

- Python 3.10 or newer
- A valid OpenAI API key
- Optional: Langfuse credentials if you want tracing enabled

## 1. Open the project folder

From your terminal, change to the project directory:

```powershell
cd path\projectfolder\phase02\multi-part-planner
```

## 2. Create and activate a virtual environment

On Windows PowerShell, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run this once in the same terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 3. Install dependencies

Install the required Python packages:

```powershell
pip install -r requirements.txt
```

## 4. Configure environment variables

Create a file named `.env` in the project root and add your settings:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-nano
```

Optional Langfuse settings:

```env
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

The application reads these values automatically when it starts.

## 5. Run the project

Start the main workflow:

```powershell
python -m main
```

The script will:

1. Generate several design approaches
2. Evaluate each approach
3. Choose a final recommended design
4. Print the results to the terminal

## Expected output

You should see three sections in the terminal:

- `=== Brainstorm ===`
- `=== Evaluation ===`
- `=== Final Design ===`

## Troubleshooting

### `ModuleNotFoundError`

This usually means the virtual environment is not active or dependencies were not installed correctly.

Run:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### `Missing required environment variable: OPENAI_API_KEY`

Make sure the `.env` file exists in the project folder and contains a valid `OPENAI_API_KEY`.

### API request errors

Check that:

- Your OpenAI API key is valid
- Your account has access to the selected model
- You have internet access from the terminal

## Project structure

- `main.py` — entry point for the workflow
- `planner.py` — brainstorm, evaluate, and select logic
- `llm.py` — OpenAI client wrapper
- `config.py` — environment variable loading and settings
- `prompts.py` — prompt templates
- `models.py` — data models for approaches and evaluations

If you want, this README can also be expanded with a section describing how to customize the prompts or change the model used by the planner.
