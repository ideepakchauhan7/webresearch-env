"""
FastAPI server for WebResearch OpenEnv.
Exposes reset, step, state, and close endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os

from environment import create_environment, WebResearchEnvironment
from tasks import get_all_tasks, get_task


# Pydantic models for API
class ResetRequest(BaseModel):
    task: str = Field(default="company_info_lookup", description="Task name to run")


class ActionRequest(BaseModel):
    action_type: str = Field(..., description="Type of action (scrape, search, extract, submit)")
    action_arg: str = Field(default="", description="Argument for the action")


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StateResponse(BaseModel):
    current_task: Optional[str]
    step_count: int
    scraped_urls: list
    done: bool


class TasksResponse(BaseModel):
    tasks: list


# Create FastAPI app
app = FastAPI(
    title="WebResearch OpenEnv",
    description="OpenEnv environment for web research tasks using FireCrawl",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
_env: Optional[WebResearchEnvironment] = None


def get_env() -> WebResearchEnvironment:
    """Get or create environment instance."""
    global _env
    if _env is None:
        _env = create_environment(max_steps=20)
    return _env


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Optional[ResetRequest] = None):
    """Reset the environment for a new episode."""
    try:
        env = get_env()
        task_name = request.task if request else "company_info_lookup"
        result = env.reset(task=task_name)
        return ResetResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/step", response_model=StepResponse)
async def step(request: ActionRequest):
    """Execute one step in the environment."""
    try:
        env = get_env()
        action = {
            "action_type": request.action_type,
            "action_arg": request.action_arg
        }
        result = env.step(action)
        return StepResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/state", response_model=StateResponse)
async def state():
    """Get the current state of the environment."""
    try:
        env = get_env()
        state = env.state()
        return StateResponse(**state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/close")
async def close():
    """Close the environment."""
    global _env
    try:
        if _env is not None:
            _env.close()
            _env = None
        return {"status": "closed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/tasks", response_model=TasksResponse)
async def list_tasks():
    """List all available tasks."""
    return TasksResponse(tasks=get_all_tasks())


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "env": "webresearch",
        "tasks": [t["name"] for t in get_all_tasks()]
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "WebResearch OpenEnv",
        "version": "1.0.0",
        "endpoints": [
            "/reset",
            "/step",
            "/state",
            "/close",
            "/tasks",
            "/health"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
