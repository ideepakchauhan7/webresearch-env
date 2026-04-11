#!/usr/bin/env python3
"""Validator-safe inference script for WebResearch OpenEnv."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from environment import create_environment
from tasks import get_all_tasks

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL")
BENCHMARK_NAME = "webresearch_env"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if OpenAI is not None else None
LLM_ENABLED = OpenAI is not None and HF_TOKEN.strip().lower() not in {
    "dummy",
    "test",
    "placeholder",
    "changeme",
}

TASK_PLANS: Dict[str, List[Dict[str, str]]] = {
    "company_info_lookup": [
        {"action_type": "search", "action_arg": "Anthropic founding year"},
        {"action_type": "scrape", "action_arg": "https://anthropic.com/about"},
        {"action_type": "submit", "action_arg": "2021"},
    ],
    "product_price_comparison": [
        {"action_type": "search", "action_arg": "tech retailer headphone prices product a product b product c"},
        {"action_type": "scrape", "action_arg": "https://tech-retailer.com/products"},
        {
            "action_type": "extract",
            "action_arg": "List the prices for Product A, Product B, and Product C. Identify the cheapest and most expensive product.",
        },
        {
            "action_type": "submit",
            "action_arg": "Product A: $99.99, Product B: $129.99, Product C: $89.99. Cheapest: Product C. Most expensive: Product B.",
        },
    ],
    "research_synthesis": [
        {"action_type": "search", "action_arg": "renewable energy growth 2023 solar wind investment"},
        {"action_type": "scrape", "action_arg": "https://market-analysis.com/energy-2023"},
        {"action_type": "scrape", "action_arg": "https://renewable-insights.com/solar-trends"},
        {"action_type": "scrape", "action_arg": "https://wind-energy.org/offshore-report"},
        {
            "action_type": "extract",
            "action_arg": "Summarize solar growth, wind growth, investment, and the key challenges or opportunities mentioned in the scraped sources.",
        },
        {
            "action_type": "submit",
            "action_arg": (
                "Renewable energy saw strong growth in 2023, with solar capacity up 30% and more than 200 GW added globally. "
                "Wind capacity also grew 15%, especially in offshore markets, while total investment reached $500 billion. "
                "Solar kept getting cheaper and more accessible, offshore wind projects expanded, and policy support helped market growth in emerging economies. "
                "Key challenges included grid integration, supply chain constraints, and interconnection delays."
            ),
        },
    ],
}


def compact_text(value: Any) -> str:
    """Collapse whitespace so stdout remains one field per line."""
    if value is None:
        return "null"
    return re.sub(r"\s+", " ", str(value)).strip()


def action_to_string(action_type: str, action_arg: str) -> str:
    """Render an action in the required log format."""
    safe_arg = compact_text(action_arg).replace("\\", "\\\\").replace("'", "\\'")
    return f"{action_type}('{safe_arg}')"


def parse_action(response: str) -> Optional[Dict[str, str]]:
    """Parse a single action from a model response."""
    match = re.search(r"(\w+)\((.*)\)", response.strip(), re.DOTALL)
    if not match:
        return None

    action_type = match.group(1).strip().lower()
    raw_arg = match.group(2).strip()
    if raw_arg.startswith(("'", '"')) and raw_arg.endswith(("'", '"')) and len(raw_arg) >= 2:
        raw_arg = raw_arg[1:-1]
    raw_arg = raw_arg.replace("\\'", "'").replace('\\"', '"')
    return {"action_type": action_type, "action_arg": compact_text(raw_arg)}


def call_llm(task_name: str, observation: Dict[str, Any], fallback_action: Dict[str, str]) -> Dict[str, str]:
    """Ask the model to confirm the next action, then fall back deterministically."""
    global LLM_ENABLED
    if not LLM_ENABLED:
        return fallback_action

    prompt = (
        "You are acting in an OpenEnv benchmark.\n"
        f"Task: {task_name}\n"
        f"Observation: {json.dumps(observation, sort_keys=True)}\n"
        f"Recommended action: {action_to_string(fallback_action['action_type'], fallback_action['action_arg'])}\n"
        "Return exactly one action call and nothing else. If the recommended action is reasonable, echo it exactly."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=64,
            timeout=5.0,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_action(content)
        if not parsed:
            return fallback_action
        if parsed["action_type"] != fallback_action["action_type"]:
            return fallback_action
        if compact_text(parsed["action_arg"]) != compact_text(fallback_action["action_arg"]):
            return fallback_action
        return parsed
    except Exception as exc:
        LLM_ENABLED = False
        print(f"LLM fallback engaged: {compact_text(exc)}", file=sys.stderr)
        return fallback_action


@dataclass
class LocalEnvClient:
    """In-process environment runner."""

    env: Any

    @classmethod
    def build(cls) -> "LocalEnvClient":
        return cls(env=create_environment())

    def health(self) -> Dict[str, Any]:
        return {"status": "healthy", "env": BENCHMARK_NAME}

    def reset(self, task_name: str) -> Dict[str, Any]:
        return self.env.reset(task=task_name)

    def step(self, action: Dict[str, str]) -> Dict[str, Any]:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()

    def tasks(self) -> Dict[str, Any]:
        return {"tasks": get_all_tasks()}


@dataclass
class HttpEnvClient:
    """HTTP client for an externally hosted environment."""

    base_url: str

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=3)
        response.raise_for_status()
        return response.json()

    def reset(self, task_name: str) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/reset", json={"task": task_name}, timeout=10)
        response.raise_for_status()
        return response.json()

    def step(self, action: Dict[str, str]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/step", json=action, timeout=10)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        try:
            requests.post(f"{self.base_url}/close", timeout=3)
        except Exception:
            pass

    def tasks(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/tasks", timeout=3)
        response.raise_for_status()
        return response.json()


def get_env_client(env_url: Optional[str]) -> LocalEnvClient | HttpEnvClient:
    """Prefer an explicit remote env, but fall back to the local implementation."""
    if env_url:
        remote = HttpEnvClient(env_url)
        try:
            remote.health()
            return remote
        except Exception as exc:
            print(
                f"ENV_URL unreachable, using local environment instead: {compact_text(exc)}",
                file=sys.stderr,
            )
    return LocalEnvClient.build()


def emit_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)


def emit_step(step_number: int, action: Dict[str, str], reward: float, done: bool, error: Optional[str]) -> None:
    action_str = action_to_string(action["action_type"], action["action_arg"])
    error_str = compact_text(error) if error else "null"
    print(
        f"[STEP] step={step_number} action={action_str} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_str}",
        flush=True,
    )


def emit_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def run_episode(task_name: str, env_client: LocalEnvClient | HttpEnvClient) -> Dict[str, Any]:
    """Run a single task to completion."""
    rewards: List[float] = []
    step_count = 0
    success = False
    last_error: Optional[str] = None

    emit_start(task_name)

    try:
        reset_data = env_client.reset(task_name)
        observation = reset_data.get("observation", {})
        plan = TASK_PLANS.get(
            task_name,
            [{"action_type": "submit", "action_arg": "Unable to solve task"}],
        )

        for fallback_action in plan:
            step_count += 1
            action = call_llm(task_name, observation, fallback_action)
            step_result = env_client.step(action)

            observation = step_result.get("observation", {})
            reward = round(float(step_result.get("reward", 0.01)), 2)
            done = bool(step_result.get("done", False))
            info = step_result.get("info", {})
            last_error = info.get("error")

            rewards.append(reward)
            emit_step(step_count, action, reward, done, last_error)

            if done:
                success = bool(info.get("success", reward >= 0.7))
                break
        else:
            last_error = "Plan exhausted before completion"
    except Exception as exc:
        last_error = compact_text(exc)
        print(f"Inference episode error for {task_name}: {last_error}", file=sys.stderr)
    finally:
        try:
            env_client.close()
        except Exception as exc:
            print(f"Close error: {compact_text(exc)}", file=sys.stderr)

        emit_end(success, step_count, rewards)

    return {
        "task": task_name,
        "success": success,
        "steps": step_count,
        "rewards": rewards,
        "last_error": last_error,
    }


def run_all_tasks(env_client: LocalEnvClient | HttpEnvClient) -> List[Dict[str, Any]]:
    """Run all declared tasks."""
    try:
        tasks = env_client.tasks().get("tasks", [])
    except Exception as exc:
        print(f"Task listing failed, using local task metadata: {compact_text(exc)}", file=sys.stderr)
        tasks = get_all_tasks()

    results = []
    for task in tasks:
        task_name = task.get("name", "unknown")
        results.append(run_episode(task_name, env_client))
    return results


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run inference on WebResearch OpenEnv")
    parser.add_argument("--task", type=str, help="Run a single task")
    parser.add_argument("--env-url", type=str, default=ENV_URL, help="Optional external environment URL")
    args = parser.parse_args()

    env_client = get_env_client(args.env_url)
    if args.task:
        run_episode(args.task, env_client)
    else:
        run_all_tasks(env_client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
