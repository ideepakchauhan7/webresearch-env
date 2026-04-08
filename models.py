"""
Pydantic models for WebResearch OpenEnv.
Defines Observation, Action, and Reward types.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ActionType(str, Enum):
    """Available action types."""
    SCRAPE = "scrape"
    SEARCH = "search"
    EXTRACT = "extract"
    SUBMIT = "submit"


class WebAction(BaseModel):
    """Action model for web research tasks."""
    action_type: ActionType = Field(..., description="Type of action to execute")
    action_arg: str = Field(default="", description="Argument for the action (URL, query, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "scrape",
                "action_arg": "https://example.com"
            }
        }


class WebObservation(BaseModel):
    """Observation returned from the environment."""
    content: str = Field(default="", description="Scraped or extracted content")
    status: str = Field(default="ok", description="Status of the observation")
    task_description: str = Field(default="", description="Description of current task")
    current_url: Optional[str] = Field(default=None, description="Current URL being viewed")
    scraped_urls: List[str] = Field(default=[], description="List of URLs scraped so far")
    step_count: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=20, description="Maximum steps allowed")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Company XYZ was founded in 2010...",
                "status": "ok",
                "task_description": "Find the founding year of Company XYZ",
                "current_url": "https://companyxyz.com/about",
                "scraped_urls": ["https://companyxyz.com/about"],
                "step_count": 1,
                "max_steps": 20
            }
        }


class WebReward(BaseModel):
    """Reward model for web research tasks."""
    value: float = Field(..., description="Reward value between 0.0 and 1.0")
    reason: str = Field(default="", description="Explanation for the reward")

    class Config:
        json_schema_extra = {
            "example": {
                "value": 0.8,
                "reason": "Correct answer with good efficiency"
            }
        }


class TaskConfig(BaseModel):
    """Configuration for a task."""
    name: str = Field(..., description="Task identifier")
    description: str = Field(..., description="Task description")
    difficulty: str = Field(..., description="Task difficulty: easy/medium/hard")
    target_answer: str = Field(..., description="Expected answer")
    allowed_domains: Optional[List[str]] = Field(default=None, description="Allowed domains for scraping")
    hints: Optional[List[str]] = Field(default=[], description="Hints for the task")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "company_info_lookup",
                "description": "Find the founding year of Anthropic",
                "difficulty": "easy",
                "target_answer": "2021",
                "allowed_domains": ["anthropic.com", "wikipedia.org"],
                "hints": ["Check the About page"]
            }
        }


class TaskResult(BaseModel):
    """Result of a task execution."""
    task_name: str = Field(..., description="Name of the task")
    success: bool = Field(..., description="Whether task was completed successfully")
    score: float = Field(..., description="Score between 0.0 and 1.0")
    steps_taken: int = Field(..., description="Number of steps taken")
    answer: str = Field(default="", description="Agent's submitted answer")
    target_answer: str = Field(default="", description="Correct answer")


class EnvironmentState(BaseModel):
    """Current state of the environment."""
    current_task: Optional[str] = Field(default=None, description="Currently active task")
    step_count: int = Field(default=0, description="Current step number")
    scraped_content: Dict[str, str] = Field(default={}, description="Content scraped from URLs")
    search_results: List[Dict[str, Any]] = Field(default=[], description="Results from searches")
    submitted_answer: Optional[str] = Field(default=None, description="Answer submitted by agent")
    done: bool = Field(default=False, description="Whether episode is complete")


class StepResponse(BaseModel):
    """Response from a step action."""
    observation: WebObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    """Response from reset action."""
    observation: WebObservation
    info: Dict[str, Any]
