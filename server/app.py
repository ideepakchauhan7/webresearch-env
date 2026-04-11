"""Canonical FastAPI app for the WebResearch OpenEnv environment."""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, HTTPException

from environment import create_environment
from models import (
    CloseResponse,
    EnvironmentState,
    GradeRequest,
    GradeResponse,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    StepResponse,
    TaskListResponse,
    TaskSummary,
    WebAction,
)
from tasks import get_all_tasks, get_task

PORT = int(os.getenv("PORT", "7860"))
ENV_NAME = "webresearch_env"

app = FastAPI(title="WebResearch OpenEnv", version="1.0.0")
environment = create_environment()


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Expose a lightweight health payload at the root for HF/validator pings."""
    return await health()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    task_names = [task["name"] for task in get_all_tasks()]
    return HealthResponse(status="healthy", env=ENV_NAME, tasks=task_names, port=PORT)


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest | None = None) -> ResetResponse:
    """Reset the active environment episode."""
    task_name = (request.task if request else None) or "company_info_lookup"
    try:
        return ResetResponse(**environment.reset(task=task_name))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
async def step(request: WebAction) -> StepResponse:
    """Execute a single environment action."""
    response = environment.step(request.model_dump())
    return StepResponse(**response)


@app.get("/state", response_model=EnvironmentState)
async def state() -> EnvironmentState:
    """Return the current episode state."""
    return EnvironmentState(**environment.state())


@app.post("/close", response_model=CloseResponse)
async def close() -> CloseResponse:
    """Close the current episode."""
    environment.close()
    return CloseResponse(status="closed")


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks() -> TaskListResponse:
    """List all supported tasks."""
    tasks = [TaskSummary(**task) for task in get_all_tasks()]
    return TaskListResponse(tasks=tasks)


@app.post("/grade", response_model=GradeResponse)
async def grade(request: GradeRequest) -> GradeResponse:
    """Run a task grader directly for validator compatibility."""
    try:
        task = get_task(request.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    score, reason = task.grade(request.answer)
    return GradeResponse(task=task.name, grader=task.grader_name, score=score, reason=reason)


@app.post("/grader", response_model=GradeResponse)
async def grader(request: GradeRequest) -> GradeResponse:
    """Backward-compatible alias for direct grader invocation."""
    return await grade(request)


def main() -> None:
    """Run the canonical app."""
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
