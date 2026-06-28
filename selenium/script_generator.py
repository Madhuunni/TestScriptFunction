import json
import re
from pathlib import Path
from typing import Any, Mapping


PERSISTENT_BROWSER_HOST = "127.0.0.1"
PERSISTENT_BROWSER_PORT = 9222
PERSISTENT_BROWSER_USER_DATA_DIR = ".selenium_chrome_profile"


BY_ALIASES = {
    "css selector": "css",
    "css": "css",
    "xpath": "xpath",
    "id": "id",
    "name": "name",
}


def safe_filename(name: str) -> str:
    """
    Convert test case name into a safe Python file name.
    """
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "generated_test"


def normalize_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize inference JSON into the structure used by generated scripts."""
    normalized_steps = []

    for step in plan.get("steps", []):
        normalized_step = dict(step)
        by = normalized_step.get("by")
        selector = normalized_step.get("selector")

        if by:
            normalized_step["by"] = BY_ALIASES.get(str(by).lower(), by)

        if selector and not normalized_step.get("target"):
            normalized_step["target"] = selector

        if normalized_step.get("action") == "navigate" and not normalized_step.get("target"):
            normalized_step["target"] = normalized_step.get("value") or plan.get("base_url")

        normalized_steps.append(normalized_step)

    return {
        "name": plan.get("name") or plan.get("test_name") or "Generated Selenium Test",
        "base_url": plan.get("base_url"),
        "steps": normalized_steps,
    }


def generate_selenium_script(
    plan: Mapping[str, Any],
    output_dir: str = "generated_tests"
) -> Path:
    """
    Generate a Selenium Python script from inference result JSON.

    The generated script:
    - Executes test steps one by one
    - Supports locator fallback using locator_candidates
    - Saves failure screenshots
    - Writes result JSON to reports/latest_result.json
    """

    Path(output_dir).mkdir(exist_ok=True)

    normalized_plan = normalize_plan(plan)
    safe_name = safe_filename(normalized_plan["name"])
    file_path = Path(output_dir) / f"{safe_name}.py"

    steps_json = json.dumps(normalized_plan, indent=2)

    script = f'''
import os
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


PLAN = {steps_json!r}
PERSISTENT_BROWSER_HOST = {PERSISTENT_BROWSER_HOST!r}
PERSISTENT_BROWSER_PORT = int(os.getenv("SELENIUM_REMOTE_DEBUGGING_PORT", "{PERSISTENT_BROWSER_PORT}"))
PERSISTENT_BROWSER_USER_DATA_DIR = os.getenv(
    "SELENIUM_CHROME_USER_DATA_DIR",
    str(Path(__file__).resolve().parent / {PERSISTENT_BROWSER_USER_DATA_DIR!r})
)
PERSISTENT_BROWSER_HEADLESS = os.getenv("SELENIUM_HEADLESS", "false").lower() in {{
    "1",
    "true",
    "yes",
}}


def get_by(by_name):
    """
    Convert our plan locator type into Selenium By type.
    """
    mapping = {{
        "css": By.CSS_SELECTOR,
        "xpath": By.XPATH,
        "id": By.ID,
        "name": By.NAME,
    }}

    if by_name not in mapping:
        raise RuntimeError(f"Unsupported locator strategy: {{by_name}}")

    return mapping[by_name]


def get_locators(step):
    """
    Build ordered locator list.

    First use the primary by/target from the step.
    Then try locator_candidates as fallback.
    """
    locators = []

    by = step.get("by")
    target = step.get("target")

    if by and target:
        locators.append({{
            "by": by,
            "target": target
        }})

    for locator in step.get("locator_candidates", []):
        locator_by = locator.get("by")
        locator_target = locator.get("target")

        if not locator_by or not locator_target:
            continue

        candidate = {{
            "by": locator_by,
            "target": locator_target
        }}

        if candidate not in locators:
            locators.append(candidate)

    return locators


def find_element(driver, wait, step, condition_type="visible"):
    """
    Find an element using primary locator and fallback locator candidates.
    """
    locators = get_locators(step)

    if not locators:
        raise RuntimeError(
            f"No locator found for step: {{step.get('description') or step.get('action')}}"
        )

    errors = []

    for locator in locators:
        by_name = locator["by"]
        target = locator["target"]

        try:
            selenium_by = get_by(by_name)

            if condition_type == "clickable":
                return wait.until(
                    EC.element_to_be_clickable((selenium_by, target))
                )

            if condition_type == "present":
                return wait.until(
                    EC.presence_of_element_located((selenium_by, target))
                )

            return wait.until(
                EC.visibility_of_element_located((selenium_by, target))
            )

        except Exception as error:
            errors.append({{
                "locator": locator,
                "error": str(error)
            }})

    raise RuntimeError(
        "Unable to find element using locator candidates: "
        + json.dumps(errors, indent=2)
    )


def wait_for_page_ready(driver, timeout=15):
    """
    Wait until document.readyState is complete.
    Useful after navigate and click.
    """
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


def save_screenshot(driver, step_index):
    """
    Save screenshot for failed step.
    """
    Path("reports").mkdir(exist_ok=True)
    screenshot_path = f"reports/failure_step_{{step_index}}.png"
    driver.save_screenshot(screenshot_path)
    return screenshot_path


def add_result(results, step_index, action, status, message, screenshot=None):
    """
    Add one step execution result.
    """
    item = {{
        "step": step_index,
        "action": action,
        "status": status,
        "message": message
    }}

    if screenshot:
        item["screenshot"] = screenshot

    results.append(item)


def is_persistent_chrome_running():
    """
    Return True when the shared Chrome remote-debugging endpoint is available.
    """
    url = f"http://{{PERSISTENT_BROWSER_HOST}}:{{PERSISTENT_BROWSER_PORT}}/json/version"

    try:
        with urllib.request.urlopen(url, timeout=1):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


def find_chrome_binary():
    """
    Find a Chrome/Chromium executable for the persistent browser process.
    """
    configured_binary = os.getenv("CHROME_BINARY")

    if configured_binary:
        return configured_binary

    for candidate in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ):
        binary = shutil.which(candidate)

        if binary:
            return binary

    raise RuntimeError(
        "Unable to find Chrome/Chromium. Set CHROME_BINARY to the browser executable."
    )


def start_persistent_chrome():
    """
    Start a Chrome process that is intentionally reused by later prompt requests.
    """
    if is_persistent_chrome_running():
        return

    Path(PERSISTENT_BROWSER_USER_DATA_DIR).mkdir(parents=True, exist_ok=True)

    command = [
        find_chrome_binary(),
        f"--remote-debugging-address={{PERSISTENT_BROWSER_HOST}}",
        f"--remote-debugging-port={{PERSISTENT_BROWSER_PORT}}",
        f"--user-data-dir={{PERSISTENT_BROWSER_USER_DATA_DIR}}",
        "--start-maximized",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--window-size=1920,1080",
    ]

    if PERSISTENT_BROWSER_HEADLESS:
        command.append("--headless=new")

    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.time() + 15

    while time.time() < deadline:
        if is_persistent_chrome_running():
            return

        time.sleep(0.25)

    raise RuntimeError("Timed out waiting for persistent Chrome to start")


def create_driver():
    """
    Attach Selenium to the shared browser instead of creating a one-off browser.
    """
    start_persistent_chrome()

    options = Options()
    options.add_experimental_option(
        "debuggerAddress",
        f"{{PERSISTENT_BROWSER_HOST}}:{{PERSISTENT_BROWSER_PORT}}",
    )

    return webdriver.Chrome(options=options)


def run():
    results = []

    Path("reports").mkdir(exist_ok=True)

    driver = create_driver()
    wait = WebDriverWait(driver, 15)

    try:
        plan = json.loads(PLAN)
        base_url = plan.get("base_url")

        for index, step in enumerate(plan["steps"], start=1):
            action = step.get("action")
            description = step.get("description") or action
            value = step.get("value")
            value_from_env = step.get("value_from_env")

            try:
                if value_from_env:
                    value = os.getenv(value_from_env)

                    if value is None:
                        raise RuntimeError(
                            f"Missing environment variable: {{value_from_env}}"
                        )

                if action == "navigate":
                    target = step.get("target") or base_url

                    if not target:
                        raise RuntimeError(
                            "No target URL found for navigate step"
                        )

                    driver.get(target)
                    wait_for_page_ready(driver)

                    add_result(
                        results,
                        index,
                        action,
                        "PASS",
                        f"Navigated to {{target}}"
                    )

                elif action == "type":
                    if value is None:
                        raise RuntimeError(
                            f"No value found for type step: {{description}}"
                        )

                    element = find_element(
                        driver,
                        wait,
                        step,
                        condition_type="visible"
                    )

                    element.clear()
                    element.send_keys(value)

                    add_result(
                        results,
                        index,
                        action,
                        "PASS",
                        description
                    )

                elif action == "click":
                    element = find_element(
                        driver,
                        wait,
                        step,
                        condition_type="clickable"
                    )

                    element.click()

                    time.sleep(1)

                    add_result(
                        results,
                        index,
                        action,
                        "PASS",
                        description
                    )

                elif action == "assert_text":
                    if value is None:
                        raise RuntimeError(
                            "assert_text requires value"
                        )

                    wait.until(
                        lambda d: value in d.find_element(By.TAG_NAME, "body").text
                    )

                    add_result(
                        results,
                        index,
                        action,
                        "PASS",
                        f"Text found: {{value}}"
                    )

                elif action == "assert_visible":
                    find_element(
                        driver,
                        wait,
                        step,
                        condition_type="visible"
                    )

                    add_result(
                        results,
                        index,
                        action,
                        "PASS",
                        description or "Element visible"
                    )

                elif action == "assert_url_contains":
                    if value is None:
                        raise RuntimeError(
                            "assert_url_contains requires value"
                        )

                    wait.until(lambda d: value in d.current_url)

                    add_result(
                        results,
                        index,
                        action,
                        "PASS",
                        f"URL contains: {{value}}"
                    )

                else:
                    raise RuntimeError(f"Unsupported action: {{action}}")

            except Exception as step_error:
                screenshot_path = save_screenshot(driver, index)

                add_result(
                    results,
                    index,
                    action,
                    "FAIL",
                    str(step_error),
                    screenshot=screenshot_path
                )

                break

    finally:
        # Keep the shared Chrome browser open so the next prompt request can
        # continue in the same browser/session. Only stop this request's
        # ChromeDriver service to avoid leaving an extra driver process behind.
        driver.service.stop()

    with open("reports/latest_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
'''

    file_path.write_text(script, encoding="utf-8")
    return file_path
