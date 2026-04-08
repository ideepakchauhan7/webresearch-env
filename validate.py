#!/usr/bin/env python3
"""
Pre-submission validation script for WebResearch OpenEnv.
Checks compliance with OpenEnv specification.
"""

import os
import sys
import yaml
import json
import subprocess
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} MISSING: {filepath}")
        return False


def check_required_files() -> bool:
    """Check for required files."""
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
        ("server.py", "FastAPI server"),
    ]

    all_exist = True
    for filepath, description in required:
        if not check_file_exists(filepath, description):
            all_exist = False

    return all_exist


def check_inference_script() -> bool:
    """Check inference.py for required elements."""
    print("\n=== Checking inference.py ===\n")

    with open("inference.py", "r") as f:
        content = f.read()

    checks = [
        ("API_BASE_URL", "API_BASE_URL environment variable"),
        ("MODEL_NAME", "MODEL_NAME environment variable"),
        ("HF_TOKEN", "HF_TOKEN environment variable"),
        ('os.getenv("API_BASE_URL"', "API_BASE_URL with default"),
        ('os.getenv("MODEL_NAME"', "MODEL_NAME with default"),
        ("[START]", "[START] log marker"),
        ("[STEP]", "[STEP] log marker"),
        ("[END]", "[END] log marker"),
        ("from openai import OpenAI", "OpenAI client import"),
        ("openai", "OpenAI usage"),
    ]

    all_pass = True
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} MISSING")
            all_pass = False

    return all_pass


def check_openenv_yaml() -> bool:
    """Check openenv.yaml structure."""
    print("\n=== Checking openenv.yaml ===\n")

    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)

        required_keys = [
            "spec_version",
            "name",
            "description",
            "type",
            "runtime",
            "tasks",
        ]

        all_pass = True
        for key in required_keys:
            if key in config:
                print(f"✓ {key}: {config.get(key)}")
            else:
                print(f"✗ {key} MISSING")
                all_pass = False

        # Check tasks
        if "tasks" in config:
            tasks = config["tasks"]
            if len(tasks) >= 3:
                print(f"✓ Has {len(tasks)} tasks (minimum 3)")
                for task in tasks:
                    if "name" in task and "difficulty" in task:
                        print(f"  - {task['name']} ({task['difficulty']})")
            else:
                print(f"✗ Has only {len(tasks)} tasks (minimum 3 required)")
                all_pass = False

        return all_pass

    except Exception as e:
        print(f"✗ Error parsing openenv.yaml: {e}")
        return False


def check_models() -> bool:
    """Check Pydantic models."""
    print("\n=== Checking Pydantic Models ===\n")

    try:
        import models
        from models import WebAction, WebObservation

        # Try to instantiate models
        action = WebAction(action_type="scrape", action_arg="https://example.com")
        observation = WebObservation(content="Test content")

        print("✓ WebAction model valid")
        print("✓ WebObservation model valid")

        return True
    except Exception as e:
        print(f"✗ Model validation error: {e}")
        return False


def check_dockerfile() -> bool:
    """Check Dockerfile."""
    print("\n=== Checking Dockerfile ===\n")

    with open("Dockerfile", "r") as f:
        content = f.read()

    checks = [
        ("FROM", "Base image"),
        ("requirements.txt", "Requirements installation"),
        ("EXPOSE", "Port exposure"),
        ("CMD", "Command"),
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
    """Check task definitions."""
    print("\n=== Checking Tasks ===\n")

    try:
        import tasks
        all_tasks = tasks.get_all_tasks()

        if len(all_tasks) >= 3:
            print(f"✓ Found {len(all_tasks)} tasks")
        else:
            print(f"✗ Found only {len(all_tasks)} tasks (minimum 3)")
            return False

        difficulties = set()
        for task in all_tasks:
            name = task.get("name", "unknown")
            difficulty = task.get("difficulty", "unknown")
            difficulties.add(difficulty)
            print(f"  - {name}: {difficulty}")

        # Check for gradle of difficulties
        if "easy" in difficulties:
            print("✓ Has easy task")
        else:
            print("✗ Missing easy task")
            return False

        if "medium" in difficulties:
            print("✓ Has medium task")
        else:
            print("✗ Missing medium task")
            return False

        if "hard" in difficulties:
            print("✓ Has hard task")
        else:
            print("✗ Missing hard task")
            return False

        return True

    except Exception as e:
        print(f"✗ Task validation error: {e}")
        return False


def check_graders() -> bool:
    """Check grader functions."""
    print("\n=== Checking Graders ===\n")

    try:
        import graders

        test_answer = "2021"
        score, reason = graders.grade_company_founding_year(test_answer, "2021")

        if 0.0 <= score <= 1.0:
            print(f"✓ Grader returns score in range [0.0, 1.0]: {score}")
        else:
            print(f"✗ Grader returns score out of range: {score}")
            return False

        print(f"  - Test score: {score}, reason: {reason}")

        return True

    except Exception as e:
        print(f"✗ Grader validation error: {e}")
        return False


def check_environment() -> bool:
    """Check environment implementation."""
    print("\n=== Checking Environment ===\n")

    try:
        import environment
        from environment import create_environment

        env = create_environment()

        # Test reset
        result = env.reset(task="company_info_lookup")
        if "observation" in result and "info" in result:
            print("✓ reset() returns correct structure")
        else:
            print("✗ reset() returns incorrect structure")
            return False

        # Test step
        action = {"action_type": "scrape", "action_arg": "https://anthropic.com/about"}
        result = env.step(action)
        if all(k in result for k in ["observation", "reward", "done", "info"]):
            print("✓ step() returns correct structure")
        else:
            print("✗ step() returns incorrect structure")
            return False

        # Check reward range
        reward = result.get("reward", 0)
        if -1.0 <= reward <= 1.0:
            print(f"✓ Reward in valid range: {reward}")
        else:
            print(f"✗ Reward out of range: {reward}")
            return False

        # Test state
        state = env.state()
        if "current_task" in state:
            print("✓ state() returns correct structure")
        else:
            print("✗ state() returns incorrect structure")
            return False

        return True

    except Exception as e:
        print(f"✗ Environment validation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("WebResearch OpenEnv Pre-Submission Validation")
    print("=" * 60)

    results = []

    results.append(("Required Files", check_required_files()))
    results.append(("Inference Script", check_inference_script()))
    results.append(("OpenEnv YAML", check_openenv_yaml()))
    results.append(("Pydantic Models", check_models()))
    results.append(("Dockerfile", check_dockerfile()))
    results.append(("Tasks", check_tasks()))
    results.append(("Graders", check_graders()))
    results.append(("Environment", check_environment()))

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")

    all_passed = all(p for _, p in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready for submission!")
    else:
        print("✗ SOME CHECKS FAILED - Fix issues before submission")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
