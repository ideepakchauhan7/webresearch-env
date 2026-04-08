# WebResearch OpenEnv

A simulated web research environment for OpenEnv that tasks AI agents with web scraping, searching, and information extraction challenges.

## Overview

WebResearch OpenEnv provides realistic web research tasks where agents must:
- Search for information using simulated search queries
- Scrape content from web pages
- Extract specific information
- Synthesize findings from multiple sources

This environment is designed for the Meta PyTorch OpenEnv Hackathon and implements the full OpenEnv interface with typed Pydantic models.

## Tasks

The environment includes three tasks of increasing difficulty:

### 1. Company Info Lookup (Easy)
**Task**: Find the founding year of Anthropic.

**Skills Required**:
- Basic web navigation
- Scraping specific pages
- Extracting simple facts

**Expected Answer**: "2021" (the year Anthropic was founded)

**Grading**: Exact match on the founding year

---

### 2. Product Price Comparison (Medium)
**Task**: Compare prices of three headphone products (A, B, C) from tech-retailer.com.

**Skills Required**:
- Scraping structured data
- Comparing multiple items
- Identifying price relationships

**Expected Answer**: Price list for all three products (e.g., "Product A: $99, Product B: $129, Product C: $89")

**Grading**: Keyword-based with price extraction validation

---

### 3. Research Synthesis (Hard)
**Task**: Synthesize renewable energy growth research from multiple sources.

**Skills Required**:
- Multi-source research
- Information synthesis
- Complex extraction and summarization

**Expected Answer**: A comprehensive summary covering:
- Solar energy growth (30% capacity increase)
- Wind energy growth (15% capacity increase)
- Investment trends ($500 billion)
- Key findings from multiple sources

**Grading**: Keyword-based requiring synthesis of multiple key elements

## Environment Interface

### Observation Space

```python
{
    "content": "Scraped or extracted content",
    "status": "ok|error|cached|complete",
    "task_description": "Description of current task",
    "current_url": "Current URL being viewed",
    "scraped_urls": ["List of scraped URLs"],
    "step_count": 0,
    "max_steps": 20
}
```

### Action Space

| Action | Description | Example |
|--------|-------------|---------|
| `scrape(url)` | Scrape content from a URL | `scrape("https://anthropic.com/about")` |
| `search(query)` | Search for information | `search("Anthropic founding year")` |
| `extract(question)` | Extract info from scraped content | `extract("When was the company founded?")` |
| `submit(answer)` | Submit final answer | `submit("2021")` |

### Reward Function

- **+0.1**: Successful scrape of new page
- **+0.15**: Bonus for scraping relevant URL
- **+0.1**: Successful search with results
- **+0.05**: Successful extraction
- **-0.01**: Redundant scrape (cached content)
- **-0.02**: Failed search/extraction
- **-0.05**: Invalid action
- **Final**: Task completion score (0.0-1.0) + efficiency bonus

## Setup

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the server**:
```bash
python server.py
# Or with uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8000
```

3. **Test the environment**:
```bash
curl http://localhost:8000/health
```

### Docker

1. **Build the image**:
```bash
docker build -t webresearch-env .
```

2. **Run the container**:
```bash
docker run -p 8000:8000 webresearch-env
```

### Hugging Face Spaces

1. Create a new Hugging Face Space
2. Select "Docker" as the SDK
3. Upload the repository files
4. The space will automatically build and deploy

## Inference Script

Run the inference script to evaluate an LLM on all tasks:

```bash
# Set required environment variables
export HF_TOKEN="your-huggingface-token"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
export ENV_URL="http://localhost:8000"

# Run all tasks
python inference.py

# Run specific task
python inference.py --task company_info_lookup
```

### Expected Output Format

```
[START] task=company_info_lookup env=webresearch_env model=gpt-4.1-mini
[STEP] step=1 action=search('Anthropic founding year') reward=0.10 done=false error=null
[STEP] step=2 action=scrape('https://anthropic.com/about') reward=0.25 done=false error=null
[STEP] step=3 action=submit('2021') reward=1.05 done=true error=null
[END] success=true steps=3 rewards=0.10,0.25,1.05
```

## Validation

Run the validation script to check compliance:

```bash
python validate.py
```

This checks:
- Required files are present
- Inference script format
- OpenEnv YAML specification
- Pydantic models
- Dockerfile structure
- Task definitions
- Grader functions
- Environment implementation

## Baseline Performance

Run with `gpt-4.1-mini` (or similar model):

| Task | Expected Success | Avg Steps | Max Reward |
|------|------------------|-----------|------------|
| company_info_lookup | 90%+ | 3-5 | 1.0 |
| product_price_comparison | 80%+ | 5-8 | 1.0 |
| research_synthesis | 60%+ | 10-15 | 1.0 |

*Note: Performance will vary based on model capability and prompt engineering.*

## Project Structure

```
.
├── inference.py          # Required: Inference script
├── openenv.yaml          # Required: OpenEnv specification
├── Dockerfile            # Required: Docker configuration
├── requirements.txt      # Python dependencies
├── server.py             # FastAPI server
├── environment.py        # Environment implementation
├── models.py             # Pydantic models
├── tasks.py              # Task definitions
├── graders.py            # Grading functions
├── validate.py           # Pre-submission validation
└── README.md             # Documentation
```

## API Endpoints

- `POST /reset` - Reset environment for new episode
- `POST /step` - Execute one step
- `GET /state` - Get current state
- `POST /close` - Close environment
- `GET /tasks` - List available tasks
- `GET /health` - Health check

## Grading System

Each task has a programmatic grader that:
- Returns a score between 0.0 and 1.0
- Provides a reason for the score
- Uses exact match, partial match, keyword, or composite grading

Example graders:
- `grade_company_founding_year`: Numeric extraction and validation
- `grade_product_price_comparison`: Keyword-based with price extraction
- `grade_research_synthesis`: Complex keyword-based requiring multiple elements

## License

MIT License - Created for the Meta PyTorch OpenEnv Hackathon
