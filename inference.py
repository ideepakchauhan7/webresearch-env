#!/usr/bin/env python3
"""
Inference script for WebResearch OpenEnv.
Uses OpenAI API client to evaluate LLM on web research tasks.
"""

import os
import sys
import json
import requests
from typing import Optional
from openai import OpenAI

# Environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# Validate required environment variables
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


def format_system_prompt(task_name: str, task_description: str) -> str:
    """Format the system prompt for the LLM."""
    return f"""You are an AI agent tasked with completing web research tasks.

Task: {task_name}
Description: {task_description}

You have access to the following actions:
- scrape(url): Scrape content from a URL and return markdown/html content
- search(query): Search the web for information
- extract(question): Extract specific information from previously scraped content
- submit(answer): Submit your final answer to complete the task

Rules:
1. Use scrape() to gather content from specific URLs
2. Use search() to find relevant web pages
3. Use extract() to get specific information from scraped content
4. Use submit() when you have the final answer
5. You have a maximum of 20 steps per task
6. Be precise and accurate in your answers

Think step by step and take actions to complete the task."""


def call_llm(messages: list, max_retries: int = 3) -> str:
    """Call the LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"LLM call failed (attempt {attempt + 1}): {e}", file=sys.stderr)
    return ""


def parse_action(response: str) -> tuple[str, Optional[str]]:
    """Parse the action from LLM response."""
    response = response.strip()

    # Look for action patterns
    import re

    # Match action_name('arg') or action_name("arg") or action_name()
    patterns = [
        r'(\w+)\(["\']([^"\']*)["\']\)',  # action('arg') or action("arg")
        r'(\w+)\(\s*\)',  # action()
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            action_name = match.group(1)
            action_arg = match.group(2) if len(match.groups()) > 1 else ""
            return action_name, action_arg

    # Default to extract if no action found
    return "extract", response


def run_episode(task_name: str, task_config: dict) -> dict:
    """Run a single episode for a task."""
    # Reset environment
    reset_resp = requests.post(f"{ENV_URL}/reset", json={"task": task_name})
    reset_data = reset_resp.json()

    observation = reset_data.get("observation", {})
    task_description = observation.get("task_description", task_config.get("description", ""))

    # Print START marker
    print(f"[START] task={task_name} env=webresearch_env model={MODEL_NAME}")
    sys.stdout.flush()

    # Initialize conversation
    messages = [
        {"role": "system", "content": format_system_prompt(task_name, task_description)},
        {"role": "user", "content": f"Task: {task_description}\n\nObservation: {json.dumps(observation)}\n\nWhat action do you take?"}
    ]

    step_count = 0
    max_steps = 20
    done = False
    rewards = []
    last_error = None
    success = False

    try:
        while not done and step_count < max_steps:
            step_count += 1

            # Get action from LLM
            llm_response = call_llm(messages)

            # Parse action
            action_name, action_arg = parse_action(llm_response)

            # Prepare action payload
            action_payload = {
                "action_type": action_name,
                "action_arg": action_arg if action_arg else ""
            }

            # Execute action
            try:
                step_resp = requests.post(f"{ENV_URL}/step", json=action_payload)
                step_data = step_resp.json()

                observation = step_data.get("observation", {})
                reward = step_data.get("reward", 0.0)
                done = step_data.get("done", False)
                info = step_data.get("info", {})
                last_error = info.get("error")

                rewards.append(round(reward, 2))

                # Print STEP marker
                error_str = str(last_error) if last_error else "null"
                print(f"[STEP] step={step_count} action={action_name}('{action_arg}') reward={reward:.2f} done={str(done).lower()} error={error_str}")
                sys.stdout.flush()

                # Update messages
                messages.append({"role": "assistant", "content": llm_response})

                if done:
                    success = info.get("success", reward > 0.5)
                    break

                # Add observation to messages
                obs_content = json.dumps(observation)
                messages.append({"role": "user", "content": f"Observation: {obs_content}\n\nWhat action do you take next?"})

            except Exception as e:
                error_msg = str(e)
                print(f"[STEP] step={step_count} action={action_name}('{action_arg}') reward=0.00 done=false error={error_msg}")
                sys.stdout.flush()
                rewards.append(0.0)
                last_error = error_msg

                if step_count >= max_steps:
                    break

                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content": f"Error executing action: {error_msg}. Try a different approach."})

    except Exception as e:
        last_error = str(e)
        print(f"Episode error: {e}", file=sys.stderr)

    finally:
        # Close environment
        try:
            requests.post(f"{ENV_URL}/close")
        except:
            pass

        # Print END marker
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_str}")
        sys.stdout.flush()

    return {
        "task": task_name,
        "success": success,
        "steps": step_count,
        "rewards": rewards,
        "total_reward": sum(rewards)
    }


def run_all_tasks():
    """Run inference on all tasks."""
    # Get available tasks
    try:
        tasks_resp = requests.get(f"{ENV_URL}/tasks")
        tasks_data = tasks_resp.json()
        tasks = tasks_data.get("tasks", [])
    except Exception as e:
        print(f"Failed to get tasks: {e}", file=sys.stderr)
        # Default tasks
        tasks = [
            {"name": "company_info_lookup", "description": "Find company information"},
            {"name": "product_price_comparison", "description": "Compare product prices"},
            {"name": "research_synthesis", "description": "Synthesize research from multiple sources"}
        ]

    results = []
    print(f"\n{'='*60}")
    print(f"Running inference on {len(tasks)} tasks")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_URL}")
    print(f"{'='*60}\n")

    for task_config in tasks:
        task_name = task_config.get("name", "unknown")
        print(f"\n--- Running task: {task_name} ---")
        result = run_episode(task_name, task_config)
        results.append(result)
        print(f"--- Task complete: success={result['success']}, reward={result['total_reward']:.2f} ---\n")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_success = sum(1 for r in results if r["success"])
    total_reward = sum(r["total_reward"] for r in results)
    print(f"Tasks completed: {len(results)}")
    print(f"Successful: {total_success}/{len(results)}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward: {total_reward/len(results):.2f}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on WebResearch OpenEnv")
    parser.add_argument("--task", type=str, help="Run specific task only")
    parser.add_argument("--env-url", type=str, default=ENV_URL, help="Environment URL")
    args = parser.parse_args()

    ENV_URL = args.env_url

    if args.task:
        # Run single task
        task_config = {"name": args.task, "description": f"Task: {args.task}"}
        run_episode(args.task, task_config)
    else:
        # Run all tasks
        run_all_tasks()
