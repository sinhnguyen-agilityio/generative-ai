# Reasoning Benchmark

## Run the benchmark

From the project directory, run:

```bash
cd reasoning-benchmark
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Set your OpenAI API key before running the benchmark. You can either export it in the shell or place it in a `.env` file:

```bash
export OPENAI_API_KEY="your_api_key"
```

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

Then start the benchmark:

```bash
python main.py
```

The script will run the benchmark using the configured reasoning strategies and print the results to the console.
