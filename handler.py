"""FastAPI request handler for prompt-to-Selenium execution."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent
INTENTFINDER_DIR = REPO_ROOT / "intentfinder"
ARTIFACT_DIR = INTENTFINDER_DIR / "artifacts_attributes"

if str(INTENTFINDER_DIR) not in sys.path:
    sys.path.insert(0, str(INTENTFINDER_DIR))

from infer import predict
from selenium.script_generator import generate_selenium_script


app = FastAPI(title="Test Script Function")


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Natural language test prompt")


def infer(prompt: str) -> dict[str, Any]:
    """Run intent inference for a prompt and return the generated test JSON."""
    inference = predict(prompt, artifact_dir=str(ARTIFACT_DIR))
    result_json = inference.get("json")

    if not isinstance(result_json, dict):
        raise ValueError("Inference did not return a JSON object")

    return result_json


def execute_selenium_script(script_path: Path) -> dict[str, Any]:
    """Execute a generated Selenium script and return process/report details."""
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    report_path = script_path.parent / "reports" / "latest_result.json"
    report: Any = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    return {
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report": report,
    }


@app.post("/")
def handle_prompt(request: PromptRequest) -> dict[str, Any]:
    """
    Accept a POST body containing `prompt`, generate a Selenium script, execute it,
    and return execution results.
    """
    try:
        result_json = infer(request.prompt)
        script_path = generate_selenium_script(result_json)
        execution = execute_selenium_script(script_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "prompt": request.prompt,
        "plan": result_json,
        "script_path": str(script_path.relative_to(REPO_ROOT)),
        "execution": execution,
    }
