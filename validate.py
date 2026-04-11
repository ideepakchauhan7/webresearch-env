#!/usr/bin/env python3
"""Pre-submission validation for the WebResearch OpenEnv hackathon repo."""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True

    print(f"✗ {description} MISSING: {filepath}")
    return False


def check_required_files() -> bool:
    """Check for required project files."""
    print("\n=== Checking Required Files ===\n")

    required = [
        ("inference.py", "Inference script"),
        ("openenv.yaml", "OpenEnv specification"),
        ("Dockerfile", "Docker configuration"),
        ("requirements.txt", "Python dependencies"),
        ("README.md", "Documentation"),
        ("environment.py", "Environment implementation"),
        ("models.py", "Pydantic models"),
        ("tasks.py", "Task definitions"),
        ("graders.py", "Grader functions"),
        ("server/app.py", "Canonical FastAPI app"),
        ("server/__init__.py", "Server package init"),
        (".dockerignore", "Docker ignore file"),
    ]

    return all(check_file_exists(path, description) for path, description in required)


def check_server_layout() -> bool:
    """Ensure the canonical import path is unambiguous."""
    print("\n=== Checking Server Layout ===\n")

    all_pass = True
    if Path("server.py").exists():
        print("✗ server.py should not exist at the repo root because it shadows the server package")
        all_pass = False
    else:
        print("✓ No root-level server.py shadowing the package")

    try:
        module = importlib.import_module("server.app")
        print(f"✓ server.app import works: {module.__file__}")
    except Exception as exc:
        print(f"✗ server.app import failed: {exc}")
        all_pass = False

    return all_pass


def check_inference_script() -> bool:
    """Check inference.py for required structure and defaults."""
    print("\n=== Checking inference.py Source ===\n")

    content = Path("inference.py").read_text()
    checks = [
        ("API_BASE_URL", "API_BASE_URL environment variable"),
        ("MODEL_NAME", "MODEL_NAME environment variable"),
        ("HF_TOKEN", "HF_TOKEN environment variable"),
        ('os.getenv("API_BASE_URL"', "API_BASE_URL with default"),
        ('os.getenv("MODEL_NAME"', "MODEL_NAME with default"),
        ("ENV_URL = os.getenv(", "Optional ENV_URL override"),
        ("from openai import OpenAI", "OpenAI client import"),
        ("[START]", "START marker"),
        ("[STEP]", "STEP marker"),
        ("[END]", "END marker"),
    ]

    all_pass = True
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} MISSING")
            all_pass = False

    if "ENV_URL environment variable is required" in content:
        print("✗ ENV_URL is still treated as mandatory")
        all_pass = False
    else:
        print("✓ ENV_URL is not mandatory")

    return all_pass


def check_inference_execution() -> bool:
    """Run the inference script with a dummy token and validate strict stdout formatting."""
    print("\n=== Checking inference.py Execution ===\n")

    env = os.environ.copy()
    env["HF_TOKEN"] = "dummy"
    env.pop("ENV_URL", None)

    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("✗ inference.py timed out")
        return False

    if result.returncode != 0:
        print(f"✗ inference.py exited with code {result.returncode}")
        if result.stderr.strip():
            print(result.stderr.strip())
        return False

    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        print("✗ inference.py produced no stdout")
        return False

    start_pattern = re.compile(r"^\[START\] task=[^ ]+ env=[^ ]+ model=.+$")
    step_pattern = re.compile(
        r"^\[STEP\] step=\d+ action=.* reward=\d+\.\d{2} done=(true|false) error=.*$"
    )
    end_pattern = re.compile(
        r"^\[END\] success=(true|false) steps=\d+ rewards=(\d+\.\d{2}(,\d+\.\d{2})*)?$"
    )

    start_count = 0
    end_count = 0
    all_pass = True

    for line in stdout_lines:
        if start_pattern.match(line):
            start_count += 1
            continue
        if step_pattern.match(line):
            continue
        if end_pattern.match(line):
            end_count += 1
            continue
        print(f"✗ Unexpected stdout line: {line}")
        all_pass = False

    if start_count != 3 or end_count != 3:
        print(f"✗ Expected 3 task runs, found {start_count} START lines and {end_count} END lines")
        all_pass = False
    else:
        print("✓ inference.py completed all 3 tasks with strict stdout formatting")

    if all_pass and result.stderr.strip():
        print("✓ inference.py stderr only contains non-fatal diagnostics")

    return all_pass


def check_openenv_yaml() -> bool:
    """Check openenv.yaml structure and deployment settings."""
    print("\n=== Checking openenv.yaml ===\n")

    try:
        config = yaml.safe_load(Path("openenv.yaml").read_text())
    except Exception as exc:
        print(f"✗ Error parsing openenv.yaml: {exc}")
        return False

    required_keys = ["spec_version", "name", "description", "type", "runtime", "tasks", "app", "port"]
    all_pass = True
    for key in required_keys:
        if key in config:
            print(f"✓ {key}: {config.get(key)}")
        else:
            print(f"✗ {key} MISSING")
            all_pass = False

    if config.get("app") != "server.app:app":
        print(f"✗ app should be server.app:app, found {config.get('app')}")
        all_pass = False
    else:
        print("✓ Canonical app entrypoint is server.app:app")

    if config.get("port") != 7860:
        print(f"✗ port should be 7860, found {config.get('port')}")
        all_pass = False
    else:
        print("✓ openenv.yaml uses port 7860")

    tasks = config.get("tasks", [])
    if len(tasks) < 3:
        print(f"✗ Has only {len(tasks)} tasks (minimum 3 required)")
        all_pass = False
    else:
        print(f"✓ Has {len(tasks)} tasks")

    return all_pass


def check_readme_frontmatter() -> bool:
    """Verify Hugging Face Space metadata is present."""
    print("\n=== Checking README Frontmatter ===\n")

    content = Path("README.md").read_text()
    if not content.startswith("---\n"):
        print("✗ README.md is missing HF Space frontmatter")
        return False

    _, frontmatter, _ = content.split("---", 2)
    try:
        metadata = yaml.safe_load(frontmatter)
    except Exception as exc:
        print(f"✗ Failed to parse README frontmatter: {exc}")
        return False

    expected = {"sdk": "docker", "app_port": 7860}
    all_pass = True
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            print(f"✗ README frontmatter {key} should be {expected_value}, found {metadata.get(key)}")
            all_pass = False
        else:
            print(f"✓ README frontmatter {key}: {expected_value}")
    return all_pass


def check_models() -> bool:
    """Check Pydantic models instantiate correctly."""
    print("\n=== Checking Pydantic Models ===\n")

    try:
        from models import EnvironmentState, WebAction, WebObservation

        WebAction(action_type="scrape", action_arg="https://example.com")
        WebObservation(content="Test content")
        EnvironmentState()
        print("✓ WebAction model valid")
        print("✓ WebObservation model valid")
        print("✓ EnvironmentState model valid")
        return True
    except Exception as exc:
        print(f"✗ Model validation error: {exc}")
        return False


def check_dockerfile() -> bool:
    """Check Dockerfile deployment choices."""
    print("\n=== Checking Dockerfile ===\n")

    content = Path("Dockerfile").read_text()
    checks = [
        ("FROM", "Base image"),
        ("requirements.txt", "Requirements installation"),
        ("COPY . .", "Full project copy"),
        ("EXPOSE 7860", "Port exposure"),
        ("server.app:app", "Canonical app command"),
    ]

    all_pass = True
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} MISSING")
            all_pass = False
    return all_pass


def check_tasks() -> bool:
    """Check task metadata and difficulty coverage."""
    print("\n=== Checking Tasks ===\n")

    try:
        import tasks

        all_tasks = tasks.get_all_tasks()
        if len(all_tasks) < 3:
            print(f"✗ Found only {len(all_tasks)} tasks")
            return False

        difficulties = {task.get("difficulty") for task in all_tasks}
        for task in all_tasks:
            print(f"✓ {task['name']} ({task['difficulty']})")

        missing = {"easy", "medium", "hard"} - difficulties
        if missing:
            print(f"✗ Missing difficulty levels: {sorted(missing)}")
            return False

        print("✓ Task difficulty ladder is complete")
        return True
    except Exception as exc:
        print(f"✗ Task validation error: {exc}")
        return False


def check_graders() -> bool:
    """Check grader outputs stay inside the validator-safe range."""
    print("\n=== Checking Graders ===\n")

    try:
        import graders

        samples = [
            graders.grade_company_founding_year("2021"),
            graders.grade_product_price_comparison(
                "Product A: $99.99, Product B: $129.99, Product C: $89.99. Cheapest: Product C. Most expensive: Product B."
            ),
            graders.grade_research_synthesis(
                "Renewable energy grew strongly in 2023, with solar growth of 30 percent and wind growth of 15 percent. "
                "Capacity rose across solar and offshore wind markets, and investment reached $500 billion. "
                "Policy support helped the market grow, while grid integration and supply chain issues remained challenges."
            ),
        ]

        all_pass = True
        for index, (score, reason) in enumerate(samples, start=1):
            if not (0.0 <= score <= 1.0):
                print(f"✗ Grader {index} produced out-of-range score {score}")
                all_pass = False
            else:
                print(f"✓ Grader {index} score={score:.2f} reason={reason}")
        return all_pass
    except Exception as exc:
        print(f"✗ Grader validation error: {exc}")
        return False


def check_environment() -> bool:
    """Smoke test the environment implementation directly."""
    print("\n=== Checking Environment ===\n")

    try:
        from environment import create_environment

        env = create_environment()
        reset_result = env.reset(task="company_info_lookup")
        if "observation" not in reset_result or "info" not in reset_result:
            print("✗ reset() returns incorrect structure")
            return False
        print("✓ reset() returns correct structure")

        search_result = env.step({"action_type": "search", "action_arg": "Anthropic founding year"})
        scrape_result = env.step({"action_type": "scrape", "action_arg": "https://anthropic.com/about"})
        submit_result = env.step({"action_type": "submit", "action_arg": "2021"})

        for name, result in [("search", search_result), ("scrape", scrape_result), ("submit", submit_result)]:
            if not all(key in result for key in ["observation", "reward", "done", "info"]):
                print(f"✗ {name} step returns incorrect structure")
                return False
            reward = float(result["reward"])
            if not (0.0 <= reward <= 1.0):
                print(f"✗ {name} step reward out of range: {reward}")
                return False
            print(f"✓ {name} step reward={reward:.2f}")

        state = env.state()
        if "current_task" not in state or "scraped_content" not in state:
            print("✗ state() returns incorrect structure")
            return False
        print("✓ state() returns correct structure")

        return True
    except Exception as exc:
        print(f"✗ Environment validation error: {exc}")
        return False


def main() -> int:
    """Run all validation checks."""
    print("=" * 60)
    print("WebResearch OpenEnv Pre-Submission Validation")
    print("=" * 60)

    checks = [
        ("Required Files", check_required_files),
        ("Server Layout", check_server_layout),
        ("Inference Script Source", check_inference_script),
        ("Inference Script Execution", check_inference_execution),
        ("OpenEnv YAML", check_openenv_yaml),
        ("README Frontmatter", check_readme_frontmatter),
        ("Pydantic Models", check_models),
        ("Dockerfile", check_dockerfile),
        ("Tasks", check_tasks),
        ("Graders", check_graders),
        ("Environment", check_environment),
    ]

    results = [(name, fn()) for name, fn in checks]

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for name, passed in results:
        symbol = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"{symbol} {name}: {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready for submission!")
    else:
        print("✗ SOME CHECKS FAILED - Fix issues before submission")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
