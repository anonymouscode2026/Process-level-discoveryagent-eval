"""
Runner — loads task configs and dispatches experiment runs to the appropriate agent.
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.trajectory.schemas import Trajectory
from src.evaluation.task_adapter import TaskAdapter
from src.llm_client import LLMClient


def load_task_config(task_name: str, path: str = "configs/tasks.yaml") -> dict[str, Any]:
    """
    Load a single task config dict by name from a tasks YAML file.

    Args:
        task_name: The "name" field of the task entry.
        path:      Path to the tasks YAML file.

    Returns:
        The task config dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If the task_name is not found.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Tasks config not found: {path}")

    with config_path.open() as fh:
        data = yaml.safe_load(fh)

    for entry in data.get("tasks", []):
        if entry["name"] == task_name:
            return entry

    available = [t["name"] for t in data.get("tasks", [])]
    raise KeyError(f"Task {task_name!r} not found in {path}. Available: {available}")


def run_experiment(
    task_name: str,
    model_name: str,
    agency_level: str,
    step_budget: int = 150,
    run_id: str = "",
    output_dir: str = "data/trajectories",
    l0_rewrite_every: int = 5,
) -> Trajectory:
    """
    Run one experiment cell: one agent on one task.

    Args:
        task_name:         Task name key (from configs/tasks.yaml).
        model_name:        Model name key (from configs/models.yaml).
        agency_level:      One of L0, L0f, L1, , L2b, .
        step_budget:       Maximum number of steps per run (default 150).
        run_id:            Unique identifier for this run.
        output_dir:        Directory to save trajectory JSON.
        l0_rewrite_every:  REWRITE cadence for Level0Agent fixed schedule
                           (default 5 → 4:1 REFINE:REWRITE ratio). Ignored
                           for L1//L2b/ which let the model choose the mode.
    Returns:
        The completed (or partial) Trajectory.
    """
    # Lazy imports to avoid circular deps at module level
    from src.agent.level0 import Level0Agent
    from src.agent.level0f import Level0fAgent
    from src.agent.level1 import Level1Agent
    from src.agent.level2b import Level2bAgent

    task_config = load_task_config(task_name)
    task_adapter = TaskAdapter(task_config)
    llm_client = LLMClient(model_name)

    agent_kwargs: dict[str, Any] = dict(
        model_name=model_name,
        task_adapter=task_adapter,
        step_budget=step_budget,
        run_id=run_id,
        agency_level=agency_level,
        llm_client=llm_client,
        output_dir=output_dir,
    )

    level_map = {
        "L0": Level0Agent,
        "L0f": Level0fAgent,
        "L1": Level1Agent,
        "L2b": Level2bAgent,
    }

    if agency_level not in level_map:
        raise ValueError(
            f"Unknown agency_level {agency_level!r}. Valid: {list(level_map)}"
        )

    # Level-specific kwargs
    if agency_level in ("L0", "L0f"):
        agent_kwargs["rewrite_every"] = l0_rewrite_every

    AgentClass = level_map[agency_level]

    try:
        agent = AgentClass(**agent_kwargs)
        trajectory = agent.run()
    except Exception as exc:
        # Build a partial trajectory with error metadata if possible
        trajectory = Trajectory(
            run_id=run_id,
            task_id=task_adapter.task_id,
            model=model_name,
            level=agency_level,
            started_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "step_budget": step_budget,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"[runner] ERROR in {task_name}/{model_name}/{agency_level}: {exc}")

    # Save trajectory
    out_path = Path(output_dir) / f"{run_id}.json"
    trajectory.save(out_path)
    print(f"[runner] Saved trajectory → {out_path}")

    return trajectory
