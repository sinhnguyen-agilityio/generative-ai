"""Launch the Streamlit app and drive it in headless Chromium, taking a screenshot per step.

    uv run --no-sync --with playwright python .claude/skills/run-deep-research/driver.py offline
    uv run --no-sync --with playwright python .claude/skills/run-deep-research/driver.py live
    uv run --no-sync --with playwright python .claude/skills/run-deep-research/driver.py live --question "..."

offline  scripted models (tests/fakes.py): full flow ask -> approve plan -> approve report -> published.
live     real app with .env: landing page only, unless --question is given. With --question it
         runs the full flow on real models, Tavily and Langfuse -- this costs money.

The driver runs in its own Python (with playwright); the app runs on the project's .venv.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

SKILL = Path(__file__).resolve().parent
UNIT = SKILL.parents[2]
VENV_PYTHON = UNIT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def healthy(port: int) -> bool:
    try:
        return urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2).read() == b"ok"
    except OSError:
        return False


def start_server(entry: Path, port: int) -> subprocess.Popen:
    if healthy(port):  # otherwise we would silently drive a stale server
        sys.exit(f"Port {port} already serves a Streamlit app; stop it or pass --port")
    log = open(SKILL / f"streamlit-{port}.log", "w", encoding="utf-8")
    server = subprocess.Popen(
        [str(VENV_PYTHON), "-m", "streamlit", "run", str(entry), "--server.headless", "true",
         "--server.port", str(port), "--browser.gatherUsageStats", "false"],
        cwd=UNIT, stdout=log, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        if healthy(port):
            return server
        time.sleep(0.5)
    server.kill()
    sys.exit(f"Streamlit did not become healthy on :{port}; see {log.name}")


def shot(page, shots: Path, name: str) -> None:
    page.wait_for_timeout(800)  # let Streamlit finish its rerun paint
    path = shots / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    print(f"screenshot {path}")


def wait_text(page, text: str, timeout_s: float) -> None:
    page.get_by_text(text).first.wait_for(timeout=timeout_s * 1000)
    print(f"saw '{text}'")


def full_flow(page, shots: Path, question: str, timeout_s: float) -> None:
    box = page.get_by_placeholder("What should I research?")
    box.fill(question)
    box.press("Enter")
    wait_text(page, "Review the research plan", timeout_s)
    shot(page, shots, "2-plan-review")
    page.get_by_role("button", name="Approve and start research").click()
    wait_text(page, "Review the report", timeout_s)
    shot(page, shots, "3-report-review")
    page.get_by_role("button", name="Approve and publish").click()
    wait_text(page, "Report published to", timeout_s)
    shot(page, shots, "4-published")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["offline", "live"])
    parser.add_argument("--question", help="live only: run the full flow on real models (costs money)")
    parser.add_argument("--port", type=int, default=8599)
    parser.add_argument("--shots", type=Path, default=SKILL / "shots")
    parser.add_argument("--timeout", type=float, default=None, help="seconds per step")
    args = parser.parse_args()

    entry = SKILL / "offline_app.py" if args.mode == "offline" else UNIT / "app.py"
    timeout_s = args.timeout or (30 if args.mode == "offline" else 900)
    args.shots.mkdir(parents=True, exist_ok=True)

    server = start_server(entry, args.port)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            errors: list[str] = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(f"http://localhost:{args.port}")
            wait_text(page, "New research", 60)
            shot(page, args.shots, "1-landing")
            if args.mode == "offline":
                full_flow(page, args.shots, "Compare LangGraph and CrewAI", timeout_s)
            elif args.question:
                full_flow(page, args.shots, args.question, timeout_s)
            exceptions = page.locator("[data-testid='stException']").all_inner_texts()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    print("console errors:", errors or "none")
    print("streamlit exceptions:", exceptions or "none")
    if errors or exceptions:
        sys.exit(1)


if __name__ == "__main__":
    main()
