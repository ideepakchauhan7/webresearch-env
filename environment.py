"""
WebResearch OpenEnv Environment Implementation.
Implements the OpenEnv interface with reset, step, state, and close methods.
"""

from typing import Any, Dict, List, Optional

from tasks import SIMULATED_CONTENT, get_simulated_content, get_task

MIN_REPORTED_REWARD = 0.01
MAX_REPORTED_REWARD = 0.99


class WebResearchEnvironment:
    """
    OpenEnv environment for web research tasks.

    The agent must:
    1. Search for information on the web
    2. Scrape content from relevant pages
    3. Extract specific information
    4. Submit answers to complete tasks
    """

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self.current_task: Optional[str] = None
        self.step_count: int = 0
        self.scraped_content: Dict[str, str] = {}
        self.search_results: List[Dict[str, Any]] = []
        self.submitted_answer: Optional[str] = None
        self.done: bool = False
        self.current_url: Optional[str] = None
        self.scraped_urls: List[str] = []

        # Progress tracking for reward shaping
        self.urls_discovered = set()
        self.content_extracted = []
        self.reward_accumulated = 0.0

    @staticmethod
    def _bounded_reward(value: float) -> float:
        """Clamp rewards into the validator-safe range."""
        return round(min(max(value, MIN_REPORTED_REWARD), MAX_REPORTED_REWARD), 2)

    def reset(self, task: str = None) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.

        Args:
            task: Name of the task to run

        Returns:
            Initial observation
        """
        self.current_task = task or "company_info_lookup"
        self.step_count = 0
        self.scraped_content = {}
        self.search_results = []
        self.submitted_answer = None
        self.done = False
        self.current_url = None
        self.scraped_urls = []
        self.urls_discovered = set()
        self.content_extracted = []
        self.reward_accumulated = 0.0

        task_obj = get_task(self.current_task)
        observation = task_obj.get_initial_observation()

        return {
            "observation": observation,
            "info": {
                "task": self.current_task,
                "difficulty": task_obj.difficulty
            }
        }

    def state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.

        Returns:
            Current state
        """
        return {
            "current_task": self.current_task,
            "step_count": self.step_count,
            "scraped_content": self.scraped_content,
            "search_results": self.search_results,
            "scraped_urls": self.scraped_urls,
            "done": self.done,
            "submitted_answer": self.submitted_answer
        }

    def close(self) -> None:
        """Close the environment."""
        self.done = True

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step in the environment.

        Args:
            action: Dictionary with 'action_type' and 'action_arg'

        Returns:
            Step response with observation, reward, done, info
        """
        if self.done:
            return {
                "observation": self._create_observation(
                    content="Environment is done. Call reset() to start a new episode.",
                    status="done"
                ),
                "reward": self._bounded_reward(0.0),
                "done": True,
                "info": {"error": "Episode already complete"}
            }

        self.step_count += 1

        # Check max steps
        if self.step_count >= self.max_steps:
            self.done = True
            return {
                "observation": self._create_observation(
                    content="Maximum steps reached. Episode terminated.",
                    status="max_steps"
                ),
                "reward": self._bounded_reward(0.0),
                "done": True,
                "info": {
                    "error": "Maximum steps reached",
                    "success": False
                }
            }

        action_type = action.get("action_type", "").lower()
        action_arg = action.get("action_arg", "")

        # Validate action
        if action_type not in ["scrape", "search", "extract", "submit"]:
            return {
                "observation": self._create_observation(
                    content=f"Invalid action type: {action_type}. Valid actions: scrape, search, extract, submit",
                    status="error"
                ),
                "reward": self._bounded_reward(0.01),
                "done": False,
                "info": {"error": f"Invalid action type: {action_type}"}
            }

        # Execute action
        if action_type == "scrape":
            return self._execute_scrape(action_arg)
        elif action_type == "search":
            return self._execute_search(action_arg)
        elif action_type == "extract":
            return self._execute_extract(action_arg)
        elif action_type == "submit":
            return self._execute_submit(action_arg)

        # Should not reach here
        return {
            "observation": self._create_observation(content="Unknown error", status="error"),
            "reward": self._bounded_reward(0.0),
            "done": True,
            "info": {"error": "Unexpected state"}
        }

    def _create_observation(self, content: str, status: str) -> Dict[str, Any]:
        """Create an observation dictionary."""
        task_obj = get_task(self.current_task)
        return {
            "content": content,
            "status": status,
            "task_description": task_obj.description,
            "current_url": self.current_url,
            "scraped_urls": self.scraped_urls,
            "step_count": self.step_count,
            "max_steps": self.max_steps
        }

    def _execute_scrape(self, url: str) -> Dict[str, Any]:
        """Execute scrape action."""
        if not url:
            return {
                "observation": self._create_observation(
                    content="Error: URL required for scrape action",
                    status="error"
                ),
                "reward": self._bounded_reward(0.01),
                "done": False,
                "info": {"error": "URL required"}
            }

        # Normalize URL
        url = url.strip()
        if not url.startswith("http"):
            url = "https://" + url

        self.current_url = url

        # Check if already scraped
        if url in self.scraped_content:
            return {
                "observation": self._create_observation(
                    content=f"(Already scraped) {self.scraped_content[url][:500]}...",
                    status="cached"
                ),
                "reward": self._bounded_reward(0.02),
                "done": False,
                "info": {"cached": True}
            }

        # Get content from simulated web
        content = get_simulated_content(url)
        self.scraped_content[url] = content
        self.scraped_urls.append(url)

        # Reward for discovering new content
        reward = 0.1
        task_obj = get_task(self.current_task)
        if url in task_obj.relevant_urls:
            reward += 0.15  # Bonus for finding relevant URL

        return {
            "observation": self._create_observation(
                content=content[:1000] + ("..." if len(content) > 1000 else ""),
                status="success"
            ),
            "reward": self._bounded_reward(reward),
            "done": False,
            "info": {"url": url, "content_length": len(content)}
        }

    def _execute_search(self, query: str) -> Dict[str, Any]:
        """Execute search action."""
        if not query:
            return {
                "observation": self._create_observation(
                    content="Error: Query required for search action",
                    status="error"
                ),
                "reward": self._bounded_reward(0.01),
                "done": False,
                "info": {"error": "Query required"}
            }

        # Simulate search results based on query
        results = []
        query_lower = query.lower()
        query_terms = []
        for raw_term in query_lower.split():
            term = raw_term.strip(".,!?;:()")
            if len(term) <= 2:
                continue
            query_terms.append(term)
            for suffix in ("ing", "ed", "es", "s"):
                if term.endswith(suffix) and len(term) - len(suffix) >= 3:
                    query_terms.append(term[:-len(suffix)])
                    break

        query_terms = list(dict.fromkeys(query_terms))
        minimum_score = 1 if len(query_terms) <= 1 else 2

        # Match against available URLs
        for url, content in SIMULATED_CONTENT.items():
            url_lower = url.lower()
            content_lower = content.lower()
            score = sum(
                1
                for term in query_terms
                if term in content_lower or term in url_lower
            )
            if score >= minimum_score:
                results.append({
                    "url": url,
                    "title": url.split("/")[-1].replace("-", " ").title(),
                    "snippet": content[:200] + "...",
                    "score": score,
                })

        results.sort(key=lambda item: (-item["score"], item["url"]))
        for result in results:
            result.pop("score", None)

        self.search_results.extend(results)

        # Add discovered URLs
        for result in results:
            self.urls_discovered.add(result["url"])

        if results:
            content = f"Search results for '{query}':\n\n" + "\n".join([
                f"- {r['title']}: {r['url']}\n  {r['snippet'][:100]}..."
                for r in results[:5]
            ])
            reward = 0.1
        else:
            content = f"No results found for '{query}'. Try different keywords."
            reward = 0.01

        return {
            "observation": self._create_observation(
                content=content,
                status="success"
            ),
            "reward": self._bounded_reward(reward),
            "done": False,
            "info": {"query": query, "results_count": len(results)}
        }

    def _execute_extract(self, question: str) -> Dict[str, Any]:
        """Execute extract action."""
        if not self.scraped_content:
            return {
                "observation": self._create_observation(
                    content="No content scraped yet. Use scrape() or search() first.",
                    status="error"
                ),
                "reward": self._bounded_reward(0.01),
                "done": False,
                "info": {"error": "No content available"}
            }

        # Simple extraction based on keywords
        question_lower = question.lower()
        relevant_content = []

        for url, content in self.scraped_content.items():
            if any(term in content.lower() for term in question_lower.split()):
                relevant_content.append(f"From {url}:\n{content[:500]}")

        if relevant_content:
            content = "\n\n".join(relevant_content)
            self.content_extracted.append(question)
            reward = 0.05
        else:
            content = f"No relevant information found for: {question}\n\nScraped content:\n" + "\n".join([
                f"- {url}" for url in self.scraped_urls
            ])
            reward = 0.01

        return {
            "observation": self._create_observation(
                content=content,
                status="success"
            ),
            "reward": self._bounded_reward(reward),
            "done": False,
            "info": {"question": question, "sources": len(relevant_content)}
        }

    def _execute_submit(self, answer: str) -> Dict[str, Any]:
        """Execute submit action and grade the answer."""
        self.submitted_answer = answer
        self.done = True

        # Grade the answer
        task_obj = get_task(self.current_task)
        score, reason = task_obj.grade(answer)

        # Additional scoring based on efficiency
        efficiency_bonus = 0.0
        if self.step_count < 10:
            efficiency_bonus = 0.1
        elif self.step_count < 15:
            efficiency_bonus = 0.05

        final_reward = self._bounded_reward(score + efficiency_bonus)

        success = score >= 0.7

        content = f"Answer submitted: {answer}\n\nScore: {score:.2f}/1.0\nReason: {reason}"
        if efficiency_bonus > 0:
            content += f"\nEfficiency bonus: +{efficiency_bonus:.2f}"

        return {
            "observation": self._create_observation(
                content=content,
                status="complete"
            ),
            "reward": final_reward,
            "done": True,
            "info": {
                "success": success,
                "score": score,
                "final_reward": final_reward,
                "reason": reason,
                "steps": self.step_count,
                "answer": answer,
                "target": task_obj.target_answer
            }
        }


def create_environment(max_steps: int = 20) -> WebResearchEnvironment:
    """Factory function to create a new environment instance."""
    return WebResearchEnvironment(max_steps=max_steps)
