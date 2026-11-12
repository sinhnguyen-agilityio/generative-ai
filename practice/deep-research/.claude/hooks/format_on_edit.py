"""PostToolUse hook: apply ruff fixes and formatting to the Python file just edited."""
import json
import subprocess
import sys

path = (json.load(sys.stdin).get("tool_input") or {}).get("file_path", "")
if path.endswith(".py"):
    subprocess.run(["uvx", "ruff", "check", "--fix", "--quiet", path])
    subprocess.run(["uvx", "ruff", "format", "--quiet", path])
