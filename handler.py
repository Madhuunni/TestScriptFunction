"""FastAPI request handler for prompt-to-Selenium execution."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from fastapi.middleware.cors import CORSMiddleware


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

origins = [
    "http://localhost:4300",
    "http://127.0.0.1:4300",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Natural language test prompt")


class InvalidPromptError(ValueError):
    """Raised when a prompt cannot be converted into a supported test plan."""


def infer(prompt: str) -> dict[str, Any]:
    """Run intent inference for a prompt and return the generated test JSON."""
    inference = predict(prompt, artifact_dir=str(ARTIFACT_DIR))

    if not inference.get("is_valid"):
        reason = inference.get("validation_error") or "Prompt is invalid or out of context"
        raise InvalidPromptError(reason)

    result_json = inference.get("json")

    if not isinstance(result_json, dict):
        raise ValueError("Inference did not return a JSON object")

    return result_json


def resolve_repo_path(path: Path) -> Path:
    """Return an absolute path, resolving relative paths from the repo root."""
    if path.is_absolute():
        return path

    return REPO_ROOT / path


def repo_relative_path(path: Path) -> str:
    """Return a repo-relative path for response payloads when possible."""
    absolute_path = resolve_repo_path(path).resolve()

    try:
        return str(absolute_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(absolute_path)


def execute_selenium_script(script_path: Path) -> dict[str, Any]:
    """Execute a generated Selenium script and return process/report details."""
    absolute_script_path = resolve_repo_path(script_path)

    completed = subprocess.run(
        [sys.executable, str(absolute_script_path)],
        cwd=absolute_script_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    report_path = absolute_script_path.parent / "reports" / "latest_result.json"
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
        script_path = generate_selenium_script(
            result_json,
            output_dir=str(REPO_ROOT / "generated_tests"),
        )
        execution = execute_selenium_script(script_path)
    except InvalidPromptError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "prompt": request.prompt,
        "plan": result_json,
        "script_path": repo_relative_path(script_path),
        "execution": execution,
    }
