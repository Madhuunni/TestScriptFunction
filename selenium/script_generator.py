import json
import re
from pathlib import Path

from schemas import TestPlan


def safe_filename(name: str) -> str:
    """
    Convert test case name into a safe Python file name.
    """
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "generated_test"


def generate_selenium_script(
    plan: TestPlan,
    output_dir: str = "generated_tests"
) -> Path:
    """
    Generate a Selenium Python script from a validated TestPlan.

    The generated script:
    - Executes test steps one by one
    - Supports locator fallback using locator_candidates
    - Saves failure screenshots
    - Writes result JSON to reports/latest_result.json
    """

    Path(output_dir).mkdir(exist_ok=True)

    safe_name = safe_filename(plan.name)
    file_path = Path(output_dir) / f"{safe_name}.py"

    steps_json = json.dumps(plan.model_dump(), indent=2)

    script = f'''
import os
import json
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


PLAN = {steps_json!r}


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


def run():
    results = []

    Path("reports").mkdir(exist_ok=True)

    options = Options()
    options.add_argument("--start-maximized")

    # Uncomment these for Linux server / Azure container execution:
    # options.add_argument("--headless=new")
    # options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage")
    # options.add_argument("--disable-gpu")
    # options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)
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

                    # Give SPA/router a moment if click causes navigation/rendering.
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
                    element = find_element(
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
                        description or f"Element visible"
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
        driver.quit()

    with open("reports/latest_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
'''

    file_path.write_text(script, encoding="utf-8")
    return file_path