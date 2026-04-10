from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy", "env": "webresearch", "tasks": ["company_info_lookup", "product_price_comparison", "research_synthesis"]}

@app.post("/reset")
async def reset():
    return {
        "observation": {
            "content": "",
            "status": "ready",
            "task_description": "Test task",
            "current_url": None,
            "scraped_urls": [],
            "step_count": 0,
            "max_steps": 20
        },
        "info": {"task": "test"}
    }

@app.post("/step")
async def step():
    return {
        "observation": {"content": "step"},
        "reward": 0.0,
        "done": False,
        "info": {}
    }
