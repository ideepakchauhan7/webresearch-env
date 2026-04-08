"""
Task definitions for WebResearch OpenEnv.
Defines three tasks of increasing difficulty: easy, medium, hard.
"""

from typing import Dict, Any, Callable, Tuple
from graders import (
    grade_company_founding_year,
    grade_product_price_comparison,
    grade_research_synthesis
)


# Simulated web content database
SIMULATED_CONTENT = {
    "https://anthropic.com/about": """
    Anthropic is an AI safety company based in San Francisco.
    Founded in 2021 by Dario and Daniela Amodei and others.
    The company focuses on developing safe and beneficial AI systems.
    Anthropic created Claude, an AI assistant designed to be helpful, harmless, and honest.
    The company has raised significant funding to pursue AI safety research.
    """,

    "https://anthropic.com": """
    Welcome to Anthropic.
    We are building reliable, interpretable, and steerable AI systems.
    Our mission is to ensure that artificial general intelligence benefits all of humanity.
    Founded by former OpenAI researchers in 2021.
    """,

    "https://tech-retailer.com/products": """
    Product Catalog - Tech Retailer

    Product A - Wireless Headphones
    Price: $99.99
    Features: Noise cancelling, 20hr battery, Bluetooth 5.0

    Product B - Premium Headphones
    Price: $129.99
    Features: Active noise cancelling, 30hr battery, Hi-res audio

    Product C - Budget Headphones
    Price: $89.99
    Features: Basic noise cancelling, 15hr battery, Bluetooth 4.2
    """,

    "https://market-analysis.com/energy-2023": """
    Global Renewable Energy Report 2023

    The renewable energy sector experienced unprecedented growth in 2023.
    Solar capacity increased by 30% globally, reaching new milestones.
    Wind energy saw a 15% increase in installed capacity.
    Investment in renewable energy reached $500 billion worldwide.

    Key findings:
    - Solar power is now the cheapest source of electricity in many regions
    - Wind energy adoption accelerated in offshore markets
    - Policy support drove significant growth in emerging economies
    - Grid integration challenges remain a concern for rapid scaling

    The market is expected to continue strong growth through 2025.
    """,

    "https://renewable-insights.com/solar-trends": """
    Solar Energy Trends 2023

    Solar power has emerged as the dominant renewable energy source.
    Key statistics:
    - Global solar capacity grew by over 200 GW in 2023
    - Module prices continued to decline, making solar more accessible
    - Residential adoption increased by 25% year-over-year
    - Corporate procurement of solar energy hit record levels

    Challenges include supply chain constraints and interconnection delays.
    """,

    "https://wind-energy.org/offshore-report": """
    Offshore Wind Energy Report

    Offshore wind saw remarkable expansion in 2023:
    - 15% capacity growth globally
    - Major projects commissioned in Europe and Asia
    - Floating wind technology advancing rapidly
    - Average turbine size increased to 8+ MW

    The sector attracted $50 billion in new investments.
    Technology improvements are driving down the levelized cost of energy.
    """,
}


class Task:
    """Represents a single task in the environment."""

    def __init__(
        self,
        name: str,
        description: str,
        difficulty: str,
        target_answer: str,
        grader: Callable[[str, str], Tuple[float, str]],
        hints: list = None,
        relevant_urls: list = None
    ):
        self.name = name
        self.description = description
        self.difficulty = difficulty
        self.target_answer = target_answer
        self.grader = grader
        self.hints = hints or []
        self.relevant_urls = relevant_urls or []

    def grade(self, submitted_answer: str) -> Tuple[float, str]:
        """Grade a submitted answer."""
        return self.grader(submitted_answer, self.target_answer)

    def get_initial_observation(self) -> Dict[str, Any]:
        """Get the initial observation for this task."""
        return {
            "task_description": self.description,
            "content": "",
            "status": "ready",
            "current_url": None,
            "scraped_urls": [],
            "step_count": 0,
            "max_steps": 20
        }


# Define tasks

TASKS = {
    "company_info_lookup": Task(
        name="company_info_lookup",
        description="""
Find the founding year of Anthropic.

Search for information about Anthropic and identify when the company was founded.
You may need to:
1. Search for "Anthropic founding year"
2. Scrape relevant pages from anthropic.com
3. Extract the founding year
4. Submit the year as your answer

Target: Provide the 4-digit year (e.g., "2021").
""",
        difficulty="easy",
        target_answer="2021",
        grader=grade_company_founding_year,
        hints=[
            "Check the About page on anthropic.com",
            "Look for mentions of when the company was founded"
        ],
        relevant_urls=["https://anthropic.com/about", "https://anthropic.com"]
    ),

    "product_price_comparison": Task(
        name="product_price_comparison",
        description="""
Compare the prices of three headphone products (A, B, and C) from tech-retailer.com.

Task:
1. Find and scrape the product catalog page
2. Extract the prices for Product A, Product B, and Product C
3. Identify which product is cheapest and which is most expensive
4. Submit a summary of the prices

Target answer format: List the prices for all three products.
Example: "Product A: $99, Product B: $129, Product C: $89. Cheapest: Product C, Most expensive: Product B"
""",
        difficulty="medium",
        target_answer="Product A: $99, Product B: $129, Product C: $89",
        grader=grade_product_price_comparison,
        hints=[
            "Navigate to tech-retailer.com/products",
            "Look for Wireless Headphones (Product A), Premium Headphones (Product B), Budget Headphones (Product C)"
        ],
        relevant_urls=["https://tech-retailer.com/products"]
    ),

    "research_synthesis": Task(
        name="research_synthesis",
        description="""
Synthesize research about renewable energy growth in 2023 from multiple sources.

Task:
1. Search for information about renewable energy trends in 2023
2. Scrape at least 3 different sources about renewable energy growth
3. Synthesize the information into a coherent summary covering:
   - Solar energy growth
   - Wind energy growth
   - Investment trends
   - Key challenges or opportunities
4. Submit a comprehensive summary (at least 3-4 sentences)

Note: You must use information from multiple sources to get full credit.
""",
        difficulty="hard",
        target_answer="""
Renewable energy experienced significant growth in 2023.
Solar capacity grew by 30% with over 200 GW added globally.
Wind energy saw 15% capacity growth, particularly in offshore markets.
Investment reached $500 billion, showing strong market confidence.
""",
        grader=grade_research_synthesis,
        hints=[
            "Look for market analysis reports",
            "Check renewable energy trend reports",
            "Synthesize information from multiple sources"
        ],
        relevant_urls=[
            "https://market-analysis.com/energy-2023",
            "https://renewable-insights.com/solar-trends",
            "https://wind-energy.org/offshore-report"
        ]
    )
}


def get_task(task_name: str) -> Task:
    """Get a task by name."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")
    return TASKS[task_name]


def get_all_tasks() -> list:
    """Get all task definitions."""
    return [
        {
            "name": task.name,
            "description": task.description,
            "difficulty": task.difficulty
        }
        for task in TASKS.values()
    ]


def get_simulated_content(url: str) -> str:
    """Get simulated content for a URL."""
    # Normalize URL
    url = url.rstrip("/")
    if url in SIMULATED_CONTENT:
        return SIMULATED_CONTENT[url]

    # Try matching without https
    if url.startswith("https://"):
        http_url = "http://" + url[8:]
        if http_url in SIMULATED_CONTENT:
            return SIMULATED_CONTENT[http_url]

    # Try adding https
    if not url.startswith("http"):
        https_url = "https://" + url
        if https_url in SIMULATED_CONTENT:
            return SIMULATED_CONTENT[https_url]

    return f"""
    Page not found: {url}

    This is a simulated web environment. Available URLs:
    - https://anthropic.com/about
    - https://anthropic.com
    - https://tech-retailer.com/products
    - https://market-analysis.com/energy-2023
    - https://renewable-insights.com/solar-trends
    - https://wind-energy.org/offshore-report
    """
