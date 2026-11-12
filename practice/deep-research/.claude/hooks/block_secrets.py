"""PreToolUse hook: deny access to real .env files and edits to uv.lock."""
import json
import os
import sys

data = json.load(sys.stdin).get("tool_input") or {}
target = data.get("file_path") or data.get("path") or ""
name = os.path.basename(target)
if (name.startswith(".env") and name != ".env.example") or (name == "uv.lock" and "file_path" in data):
    print(f"Blocked: {name} is protected.", file=sys.stderr)
    sys.exit(2)
