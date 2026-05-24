"""
Edit type classifier for trajectory steps.

Classifies code changes into one of four categories using AST analysis:
- parameter_tuning: only constant values changed
- structural_change: same structure, different internals
- algorithm_switch: fundamentally different algorithm
- representation_change: same intent, different data structures/abstractions
"""

from __future__ import annotations

import ast
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from src.trajectory.schemas import Trajectory

EditType = Literal[
    "parameter_tuning",
    "structural_change",
    "algorithm_switch",
    "representation_change",
]


def _get_top_level_names(tree: ast.AST) -> set[str]:
    """Get top-level function and class names from an AST."""
    names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _get_all_constants(tree: ast.AST) -> list:
    """Extract all constant values from an AST."""
    constants = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            constants.append(node.value)
        elif isinstance(node, ast.Num):  # Python < 3.8 compat
            constants.append(node.n)
    return constants


def _get_structure_fingerprint(tree: ast.AST) -> list[str]:
    """Get a structural fingerprint: list of AST node type names (non-constant, non-trivial)."""
    fingerprint = []
    for node in ast.walk(tree):
        name = type(node).__name__
        if name not in ("Constant", "Num", "Str", "Load", "Store", "Del"):
            fingerprint.append(name)
    return fingerprint


def _get_function_signatures(tree: ast.AST) -> dict[str, list[str]]:
    """Get function name -> argument names mapping."""
    sigs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            sigs[node.name] = args
    return sigs


def _get_call_graph(tree: ast.AST) -> dict[str, list[str]]:
    """Get a simple call graph: function name -> list of called function names."""
    graph: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.append(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.append(child.func.attr)
            graph[node.name] = calls
    return graph


def classify_edit(
    prev_code: str,
    new_code: str,
    mode: str,
) -> EditType:
    """
    Classify the type of edit between prev_code and new_code.

    Args:
        prev_code: Previous code string.
        new_code:  New code string.
        mode:      Agent mode: "REWRITE" or "REFINE".

    Returns:
        One of: "parameter_tuning", "structural_change",
                "algorithm_switch", "representation_change".
    """
    # Try to parse both
    try:
        prev_tree = ast.parse(prev_code) if prev_code.strip() else None
    except SyntaxError:
        prev_tree = None

    try:
        new_tree = ast.parse(new_code) if new_code.strip() else None
    except SyntaxError:
        new_tree = None

    # Fall back to mode-based defaults if parsing fails
    if prev_tree is None or new_tree is None:
        if mode == "REWRITE":
            return "algorithm_switch"
        else:
            return "structural_change"

    mode_upper = mode.upper()

    if mode_upper == "REWRITE":
        # Default for REWRITE: algorithm_switch
        # Exception: if top-level names overlap significantly -> representation_change
        prev_names = _get_top_level_names(prev_tree)
        new_names = _get_top_level_names(new_tree)
        if prev_names and new_names:
            overlap = len(prev_names & new_names)
            total = len(prev_names | new_names)
            if total > 0 and overlap / total >= 0.5:
                return "representation_change"
        return "algorithm_switch"

    else:
        # REFINE mode: detailed AST diff
        prev_fp = _get_structure_fingerprint(prev_tree)
        new_fp = _get_structure_fingerprint(new_tree)

        # Check if only constants changed (structure identical)
        if prev_fp == new_fp:
            return "parameter_tuning"

        # Check if same function signatures and same call graph
        prev_sigs = _get_function_signatures(prev_tree)
        new_sigs = _get_function_signatures(new_tree)
        prev_calls = _get_call_graph(prev_tree)
        new_calls = _get_call_graph(new_tree)

        if prev_sigs == new_sigs and prev_calls == new_calls:
            return "structural_change"

        return "algorithm_switch"


def annotate_trajectory_edits(trajectory: "Trajectory") -> "Trajectory":
    """
    Annotate each TrajectoryStep with an edit_type.

    Fills step.eval_result["edit_type"] (or step.edit_type if the field exists).
    prev_code for step i is step i-1's generation.code (or "" for step 1).
    mode is taken from step.decision.action_type if it contains REWRITE/REFINE,
    else from trajectory metadata or defaults to REFINE.

    Returns the modified trajectory (in-place modification).
    """
    for i, step in enumerate(trajectory.steps):
        prev_code = ""
        if i > 0:
            prev_step = trajectory.steps[i - 1]
            prev_code = prev_step.generation.code if prev_step.generation else ""

        new_code = step.generation.code if step.generation else ""

        # Determine mode
        mode = "REFINE"
        if step.decision:
            action = step.decision.action_type.upper() if step.decision.action_type else ""
            if "REWRITE" in action:
                mode = "REWRITE"
            elif "REFINE" in action:
                mode = "REFINE"
        # Also check eval_result or metadata for explicit mode_choice
        if step.eval_result and isinstance(step.eval_result, dict):
            mc = step.eval_result.get("mode_choice", "")
            if mc:
                mode = mc.upper()

        edit_type = classify_edit(prev_code, new_code, mode)

        if hasattr(step, "edit_type"):
            step.edit_type = edit_type
        else:
            if step.eval_result is None:
                step.eval_result = {}
            step.eval_result["edit_type"] = edit_type

    return trajectory


def get_step_edit_type(step) -> str | None:
    """Helper to retrieve edit_type from a step regardless of where it's stored."""
    if hasattr(step, "edit_type"):
        return step.edit_type
    if step.eval_result and isinstance(step.eval_result, dict):
        return step.eval_result.get("edit_type")
    return None
