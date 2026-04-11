from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class ResetRequest(BaseModel):
    task: Optional[str] = "company_info_lookup"

class ActionRequest(BaseModel):
    action_type: str
    action_arg: Optional[str] = ""

TASKS = {
    "company_info_lookup": {"difficulty": "easy"},
    "product_price_comparison": {"difficulty": "medium"},
    "research_synthesis": {"difficulty": "hard"}
}
@app.get("/")
async def root():
    return {"message": "Welcome to the Web Research Agent API!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "env": "webresearch", "tasks": list(TASKS.keys())}

@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    return {
        "observation": {
            "content": "",
            "status": "ready",
            "task_description": "Find the founding year of Anthropic",
            "current_url": None,
            "scraped_urls": [],
            "step_count": 0,
            "max_steps": 20
        },
        "info": {"task": "company_info_lookup"}
    }

@app.post("/step")
async def step(request: ActionRequest):
    return {
        "observation": {"content": "Action executed", "status": "ok"},
        "reward": 0.1,
        "done": False,
        "info": {}
    }

@app.get("/state")
async def state():
    return {"current_task": None, "step_count": 0, "scraped_urls": [], "done": False}

@app.post("/close")
async def close():
    return {"status": "closed"}

@app.get("/tasks")
async def list_tasks():
    return {"tasks": [{"name": k, "difficulty": v["difficulty"]} for k, v in TASKS.items()]}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
