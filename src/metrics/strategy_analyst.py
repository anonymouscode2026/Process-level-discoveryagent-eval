"""
Strategy labeling for trajectory steps using LLM.

Post-hoc analysis: annotates each step's strategy_label by asking an LLM
to assign a single snake_case label to the code change.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.trajectory.schemas import Trajectory
    from src.llm_client import LLMClient


STRATEGY_LABEL_PROMPT = """\
You are analyzing a code optimization trajectory. Given a previous code version and a new code version for a task, assign exactly ONE snake_case strategy label that best describes the type of change made.

Task description: {task_description}

Previous code:
```python
{prev_code}
```

New code:
```python
{new_code}
```

Respond with ONLY a single snake_case label (no explanation, no punctuation, no spaces). Examples of valid labels: parameter_tuning, algorithm_switch, data_structure_change, loop_optimization, vectorization, heuristic_refinement, initialization_change, pruning_strategy.

Label:"""

# In-memory cache keyed on (sha1(prev_code), sha1(new_code))
_STRATEGY_CACHE: dict[tuple[str, str], str] = {}


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _parse_label(response: str) -> Optional[str]:
    """Extract a valid snake_case label from LLM response."""
    text = response.strip()
    # Take first line
    first_line = text.split("\n")[0].strip().lower()
    # Remove any leading/trailing punctuation
    first_line = first_line.strip("`:\"' \t")
    # Match snake_case pattern
    match = re.match(r'^([a-z][a-z0-9_]*)$', first_line)
    if match:
        return match.group(1)
    # Try to find a snake_case token anywhere in the first line
    tokens = re.findall(r'[a-z][a-z0-9_]*', first_line)
    if tokens:
        # Prefer multi-word tokens (likely a label)
        for tok in tokens:
            if '_' in tok:
                return tok
        return tokens[0]
    return None


def label_step_strategy(
    prev_code: str,
    new_code: str,
    task_description: str,
    llm_client: "LLMClient",
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """
    Ask the LLM to assign a single snake_case strategy label for the code change.

    Uses an in-memory cache. Retries once on parse failure.

    Returns a snake_case label string.
    """
    cache_key = (_sha1(prev_code), _sha1(new_code))
    if cache_key in _STRATEGY_CACHE:
        return _STRATEGY_CACHE[cache_key]

    prompt = STRATEGY_LABEL_PROMPT.format(
        task_description=task_description,
        prev_code=prev_code or "(empty)",
        new_code=new_code or "(empty)",
    )

    label: Optional[str] = None
    for attempt in range(2):
        try:
            result = llm_client.complete(prompt, max_tokens=32, temperature=0.0)
            label = _parse_label(result["text"])
            if label:
                break
        except Exception:
            if attempt == 1:
                label = "unknown_strategy"

    if not label:
        label = "unknown_strategy"

    _STRATEGY_CACHE[cache_key] = label
    return label


def annotate_trajectory_strategies(
    trajectory: "Trajectory",
    llm_client: "LLMClient",
    model: str = "claude-haiku-4-5-20251001",
    task_description: str = "",
) -> "Trajectory":
    """
    Annotate each TrajectoryStep with a strategy_label.

    For step i, prev_code is the generation.code from step i-1 (or "" for step 1).
    Fills step.generation's strategy_label attribute if present, else stores in
    step's eval_result under "strategy_label" key for compatibility.

    Returns the modified trajectory (in-place modification).
    """
    from src.trajectory.schemas import Trajectory as _Trajectory  # noqa: F401

    desc = task_description or getattr(trajectory, "task_id", "") or ""

    for i, step in enumerate(trajectory.steps):
        prev_code = ""
        if i > 0:
            prev_step = trajectory.steps[i - 1]
            prev_code = prev_step.generation.code if prev_step.generation else ""

        new_code = step.generation.code if step.generation else ""

        label = label_step_strategy(
            prev_code=prev_code,
            new_code=new_code,
            task_description=desc,
            llm_client=llm_client,
            model=model,
        )

        # Store on step — use eval_result dict as a side-channel if no dedicated field
        if hasattr(step, "strategy_label"):
            step.strategy_label = label
        else:
            if step.eval_result is None:
                step.eval_result = {}
            step.eval_result["strategy_label"] = label

    return trajectory


def get_step_strategy_label(step) -> Optional[str]:
    """Helper to retrieve strategy_label from a step regardless of where it's stored."""
    if hasattr(step, "strategy_label"):
        return step.strategy_label
    if step.eval_result and isinstance(step.eval_result, dict):
        return step.eval_result.get("strategy_label")
    return None
