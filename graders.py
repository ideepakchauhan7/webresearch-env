"""
Grader functions for WebResearch tasks.
Each grader evaluates task completion and returns a score between 0.0 and 1.0.
"""

import re
from difflib import SequenceMatcher

MIN_SUBMISSION_SCORE = 0.01
MAX_SUBMISSION_SCORE = 0.99


def clamp_submission_score(score: float) -> float:
    """Keep reported grader scores inside the validator-safe range."""
    return round(min(max(score, MIN_SUBMISSION_SCORE), MAX_SUBMISSION_SCORE), 2)


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove common punctuation at ends
    text = text.strip(".,!?;:")
    return text


def exact_match_grader(submitted: str, target: str, case_sensitive: bool = False) -> float:
    """
    Exact match grader.
    Returns 1.0 if answers match exactly, 0.0 otherwise.
    """
    if not submitted:
        return 0.0

    if case_sensitive:
        return 1.0 if submitted.strip() == target.strip() else 0.0
    else:
        return 1.0 if normalize_answer(submitted) == normalize_answer(target) else 0.0


def partial_match_grader(submitted: str, target: str) -> float:
    """
    Partial match grader using sequence similarity.
    Returns score between 0.0 and 1.0 based on similarity ratio.
    """
    if not submitted:
        return 0.0

    norm_submitted = normalize_answer(submitted)
    norm_target = normalize_answer(target)

    # Use difflib's sequence matcher for similarity ratio
    similarity = SequenceMatcher(None, norm_submitted, norm_target).ratio()

    # Also check if target is contained within submission
    if norm_target in norm_submitted or norm_submitted in norm_target:
        similarity = max(similarity, 0.7)

    return round(similarity, 2)


def numeric_grader(submitted: str, target: str, tolerance: float = 0.0) -> float:
    """
    Numeric value grader.
    Extracts numbers from answers and compares with optional tolerance.
    """
    if not submitted:
        return 0.0

    try:
        # Extract numbers from submission
        submitted_nums = re.findall(r'-?\d+\.?\d*', submitted)
        target_num = float(target)

        if not submitted_nums:
            return 0.0

        for num_str in submitted_nums:
            try:
                submitted_num = float(num_str)
                if tolerance > 0:
                    if abs(submitted_num - target_num) <= tolerance:
                        return 1.0
                else:
                    if abs(submitted_num - target_num) < 0.01:
                        return 1.0
            except ValueError:
                continue

        # No matching number found
        return 0.0

    except (ValueError, TypeError):
        return 0.0


def list_match_grader(submitted: str, target_items: list, min_matches: int = 1) -> float:
    """
    List match grader.
    Checks if submitted answer contains required items from a list.
    """
    if not submitted or not target_items:
        return 0.0

    norm_submitted = normalize_answer(submitted)
    matches = 0

    for item in target_items:
        norm_item = normalize_answer(item)
        if norm_item in norm_submitted or norm_submitted in norm_item:
            matches += 1

    if matches >= len(target_items):
        return 1.0
    elif matches >= min_matches:
        return matches / len(target_items)
    else:
        return 0.0


def keyword_grader(submitted: str, required_keywords: list, optional_keywords: list = None) -> float:
    """
    Keyword-based grader.
    Checks for required and optional keywords in the answer.
    """
    if not submitted:
        return 0.0

    norm_submitted = normalize_answer(submitted)

    # Check required keywords
    required_matches = 0
    for keyword in required_keywords:
        if normalize_answer(keyword) in norm_submitted:
            required_matches += 1

    if required_matches < len(required_keywords):
        # Missing some required keywords
        return (required_matches / len(required_keywords)) * 0.5

    # All required keywords found, check optional
    score = 0.5

    if optional_keywords:
        optional_matches = 0
        for keyword in optional_keywords:
            if normalize_answer(keyword) in norm_submitted:
                optional_matches += 1
        score += (optional_matches / len(optional_keywords)) * 0.5
    else:
        score = 1.0

    return round(score, 2)


def composite_grader(submitted: str, target: str, graders: list, weights: list = None) -> tuple[float, str]:
    """
    Composite grader that combines multiple grading strategies.
    Returns weighted average score.
    """
    if weights is None:
        weights = [1.0 / len(graders)] * len(graders)

    total_score = 0.0
    total_weight = 0.0
    reasons = []

    for grader, weight in zip(graders, weights):
        score = grader(submitted, target)
        total_score += score * weight
        total_weight += weight
        reasons.append(f"{grader.__name__}: {score:.2f}")

    final_score = total_score / total_weight if total_weight > 0 else 0.0
    reason = "; ".join(reasons)

    return round(min(final_score, 1.0), 2), reason


# Task-specific graders

def grade_company_founding_year(submitted: str, target: str = "2021") -> tuple[float, str]:
    """
    Grade task: Find the founding year of a company.
    """
    score = numeric_grader(submitted, target)

    if score == 1.0:
        return clamp_submission_score(1.0), "Correct founding year"
    elif score > 0:
        return clamp_submission_score(score), "Partial credit - wrong year"
    else:
        # Check if submission contains the year 2021 as text
        if "2021" in submitted:
            return clamp_submission_score(1.0), "Correct founding year found in text"
        return clamp_submission_score(0.0), "No valid founding year found"


def grade_product_price_comparison(submitted: str, target: str = "Product A: $99, Product B: $129, Product C: $89") -> tuple[float, str]:
    """
    Grade task: Compare prices across multiple products.
    """
    normalized = normalize_answer(submitted)
    checks = []

    labeled_prices = {
        "Product A": "99",
        "Product B": "129",
        "Product C": "89",
    }
    for product, price in labeled_prices.items():
        pattern = rf"{product.lower()}[^\d$]*(?:\$)?{price}"
        checks.append(1.0 if re.search(pattern, normalized) else 0.0)

    checks.append(1.0 if "cheapest" in normalized and "product c" in normalized else 0.0)
    checks.append(1.0 if ("most expensive" in normalized or "expensive" in normalized) and "product b" in normalized else 0.0)

    score = sum(checks) / len(checks)
    if score >= 0.99:
        return clamp_submission_score(score), "All prices and comparisons correctly identified"
    if score >= 0.6:
        return clamp_submission_score(score), "Most prices correctly identified"

    keyword_score = keyword_grader(
        submitted,
        required_keywords=["Product A", "Product B", "Product C"],
        optional_keywords=["$99", "$129", "$89", "cheapest", "expensive"],
    )
    return clamp_submission_score(max(score, keyword_score)), "Partial product price comparison"


def grade_research_synthesis(submitted: str, target: str = "") -> tuple[float, str]:
    """
    Grade task: Synthesize research from multiple sources.
    Complex task requiring synthesis of information from 3+ sources.
    """
    required_elements = [
        "renewable energy",
        "2023",
        "growth",
        "solar",
        "wind"
    ]

    optional_elements = [
        "percentage",
        "capacity",
        "investment",
        "policy",
        "market"
    ]

    score = keyword_grader(submitted, required_elements, optional_elements)

    # Additional check for coherence (length indicates some synthesis)
    if len(submitted.split()) < 20:
        score = max(score - 0.2, 0.0)
        return clamp_submission_score(round(score, 2)), "Answer too short for proper synthesis"

    if score == 1.0:
        return clamp_submission_score(1.0), "Comprehensive research synthesis with all key elements"
    elif score >= 0.7:
        return clamp_submission_score(score), "Good synthesis with most key elements"
    elif score >= 0.4:
        return clamp_submission_score(score), "Partial synthesis, missing key elements"
    else:
        return clamp_submission_score(score), "Insufficient synthesis"
