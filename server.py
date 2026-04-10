from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

SIMULATED_CONTENT = {
    "https://anthropic.com/about": "Anthropic was founded in 2021.",
    "https://tech-retailer.com/products": "Product A: $99, Product B: $129, Product C: $89",
}

TASKS = {
    "company_info_lookup": {"description": "Find founding year", "difficulty": "easy"},
    "product_price_comparison": {"description": "Compare prices", "difficulty": "medium"},
    "research_synthesis": {"description": "Synthesize research", "difficulty": "hard"}
}

class WebEnv:
    def __init__(self):
        self.current_task = None
        self.step_count = 0
        self.done = False

    def reset(self, task="company_info_lookup"):
        self.current_task = task
        self.step_count = 0
        self.done = False
        return {
            "observation": {
                "content": "",
                "status": "ready",
                "task_description": TASKS[task]["description"],
                "current_url": None,
                "scraped_urls": [],
                "step_count": 0,
                "max_steps": 20
            },
            "info": {"task": task}
        }

    def step(self, action_type, action_arg=""):
        if action_type == "scrape":
            content = SIMULATED_CONTENT.get(action_arg, "Not found")
            return {"observation": {"content": content, "status": "ok"}, "reward": 0.1, "done": False, "info": {}}
        elif action_type == "submit":
            return {"observation": {"content": "Submitted", "status": "done"}, "reward": 1.0, "done": True, "info": {"success": True}}
        else:
            return {"observation": {"content": "Action executed", "status": "ok"}, "reward": 0.0, "done": False, "info": {}}

    def state(self):
        return {"current_task": self.current_task, "step_count": self.step_count, "scraped_urls": [], "done": self.done}

_env = None
def get_env():
    global _env
    if _env is None:
        _env = WebEnv()
    return _env

class ResetRequest(BaseModel):
    task: Optional[str] = "company_info_lookup"

class ActionRequest(BaseModel):
    action_type: str
    action_arg: Optional[str] = ""

@app.get("/health")
async def health():
    return {"status": "healthy", "env": "webresearch", "tasks": list(TASKS.keys())}

@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    task = request.task if request else "company_info_lookup"
    return get_env().reset(task)

@app.post("/step")
async def step(request: ActionRequest):
    return get_env().step(request.action_type, request.action_arg)

@app.get("/state")
async def state():
    return get_env().state()

@app.post("/close")
async def close():
    global _env
    _env = None
    return {"status": "closed"}

@app.get("/tasks")
async def list_tasks():
    return {"tasks": [{"name": k, "difficulty": v["difficulty"]} for k, v in TASKS.items()]}
