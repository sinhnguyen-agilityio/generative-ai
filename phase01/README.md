# Phase 01 - Classifier and Strange Commands

This project provides two interactive command-line tools for working with a secure LLM-based text pipeline:

- `classifier`: classifies user input text and returns a structured safety/classification result.
- `strange`: analyzes input text for suspicious or unusual patterns and is useful for testing how temperature affects the output.

These commands help you explore the security and classification logic in a simple terminal workflow.

## 1. Prerequisites

Make sure you have the following installed:

- Python 3.10 or newer
- pip
- An OpenAI API key

## 2. Prepare the environment

Open a terminal in the project root folder and change to the project directory:

```powershell
cd path\to\your\project
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

- Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

- Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

- macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required Python packages:

```powershell
pip install -r phase01\requirements.txt
```

Copy the sample environment file and update it with your own values:

```powershell
copy .env.example .env
```

Then edit `.env` and set your credentials:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-mini
OPENAI_TEMPERATURE=0.0
```

If you prefer to set the variables directly in PowerShell, you can run:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
$env:OPENAI_MODEL="gpt-5-mini"
$env:OPENAI_TEMPERATURE="0.0"
```

> Tip: The `strange` command is a good way to compare outputs when you change `OPENAI_TEMPERATURE` in `.env`.

## 3. Run the classifier command

From the project root, run:

```powershell
python -m phase01.command.classifier
```

You will see a prompt like this:

```text
Enter text (or 'quit'):
```

Type a sentence or paragraph and press Enter. The command will process the input and print a JSON result showing the classification outcome.

To exit, type:

```text
quit
```

## 4. Run the strange command

From the project root, run:

```powershell
python -m phase01.command.strange
```

You will again be prompted with:

```text
Enter text (or 'quit'):
```

Type your input and press Enter. The command will analyze the text through the strange-sequence pipeline and display the result.

To test how temperature changes the result, try different values in `.env` and run the command again.

To exit, type:

```text
quit
```

## 5. Example usage

Example for the classifier:

```powershell
python -m phase01.command.classifier
Enter text (or 'quit'): Hello world
quit
```

Example for the strange command:

```powershell
python -m phase01.command.strange
Enter text (or 'quit'): 1 4 7 10 13
quit
```

## 6. Troubleshooting

If the command fails, check the following:

- Python is installed and the virtual environment is activated.
- The required packages were installed successfully.
- `OPENAI_API_KEY` is set correctly.
- `OPENAI_TEMPERATURE` is valid and set in `.env` if you want to test different outputs.
- You are running the command from the project root folder.

If you encounter an API error, verify that your OpenAI key is valid and has access to the selected model.
