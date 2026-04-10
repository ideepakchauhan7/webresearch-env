from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI(title="WebResearch OpenEnv", version="1.0.0")

class ResetRequest(BaseModel):
    task: Optional[str] = "company_info_lookup"

class ActionRequest(BaseModel):
    action_type: str
    action_arg: Optional[str] = ""

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "env": "webresearch",
        "tasks": ["company_info_lookup", "product_price_comparison", "research_synthesis"]
    }

@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    task_name = request.task if request else "company_info_lookup"
    return {
        "observation": {
            "content": "",
            "status": "ready",
            "task_description": f"Task: {task_name}",
            "current_url": None,
            "scraped_urls": [],
            "step_count": 0,
            "max_steps": 20
        },
        "info": {"task": task_name}
    }

@app.post("/step")
async def step(request: ActionRequest):
    return {
        "observation": {"content": f"Executed: {request.action_type}"},
        "reward": 0.1,
        "done": False,
        "info": {}
    }

@app.get("/state")
async def state():
    return {
        "current_task": None,
        "step_count": 0,
        "scraped_urls": [],
        "done": False
    }

@app.post("/close")
async def close():
    return {"status": "closed"}

@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"name": "company_info_lookup", "difficulty": "easy"},
            {"name": "product_price_comparison", "difficulty": "medium"},
            {"name": "research_synthesis", "difficulty": "hard"}
        ]
    }
