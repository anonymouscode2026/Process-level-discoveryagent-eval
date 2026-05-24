"""
DiscoveryAgent — abstract base class for all agency levels.

Subclasses implement run() and override _build_context() to inject
level-specific information into the LLM prompt.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from src.trajectory.schemas import (
    DecisionStep,
    GenerationStep,
    Trajectory,
    TrajectoryStep,
)
from src.trajectory.recorder import TrajectoryRecorder
from src.llm_client import LLMClient
from src.evaluation.task_adapter import TaskAdapter


# ---------------------------------------------------------------------------
# Prompt templates — neutral wording, no "explore" / "exploit"
# ---------------------------------------------------------------------------

# Templates are split into a STABLE prefix (system intro + task description)
# and a DYNAMIC suffix (per-step state). The prefix is identical across every
# LLM call in a run, so we hand it to LLMClient as a `cache_prefix` and rely
# on Anthropic prompt caching to avoid re-billing the README/seed every step.
# The suffix carries the per-step varying parts (best_score, best_code,
# context, reasoning) and is sent as the regular prompt body.

_DECISION_PREFIX_TEMPLATE = """\
You are an optimization agent working on a programming task.

Task description:
{task_prompt}"""

_DECISION_SUFFIX_TEMPLATE = """\

Current best score: {best_score}
Current best solution available: {has_best_code}
Steps completed: {steps_completed}
Steps remaining: {budget_remaining}

Additional context:
{context_text}

Based on the above, decide how to proceed:
- REFINE: make targeted edits to the current best solution (requires a current best solution to be available)
- REWRITE: start fresh with a completely new approach

Note: REFINE is only valid when a current best solution is available. If none is available, choose REWRITE.

Respond EXACTLY in this format (no other text):
[MODE: REFINE|REWRITE]
[REASONING: your reasoning here]
"""

# Backward-compatible monolithic template (some external scripts/tests may
# still import this). Composed from prefix+suffix.
DECISION_PROMPT_TEMPLATE = _DECISION_PREFIX_TEMPLATE + "\n" + _DECISION_SUFFIX_TEMPLATE

_GENERATION_REFINE_PREFIX_TEMPLATE = """\
You are an expert programmer optimizing a solution.

Task description:
{task_prompt}"""

_GENERATION_REFINE_SUFFIX_TEMPLATE = """\

Current best score: {best_score}
Current best solution:
```{code_language}
{best_code}
```

Reasoning for this attempt: {reasoning}

Additional context:
{context_text}

Make targeted improvements to the solution above by emitting one or more
SEARCH/REPLACE blocks. Do NOT rewrite the whole file — only emit the small
fragments that need to change.

Format each change exactly like this (no surrounding code fence):

<<<<<<< SEARCH
<exact text from the current best solution, including indentation>
=======
<new text to replace it with>
>>>>>>> REPLACE

HARD RULES (violations cause the change to be rejected):
- The SEARCH text must match the current best solution VERBATIM (whitespace,
  indentation, blank lines, comments — all exact).
- **Each SEARCH block must contain AT MOST 30 lines.** A SEARCH block that
  spans the whole file (or more than ~30 lines) will be rejected. If you find
  yourself wanting a larger SEARCH block, split it into several smaller ones.
- Each SEARCH text must be unique within the file. If a snippet appears
  multiple times, expand the SEARCH text with one or two surrounding lines
  until unique (still under the 30-line cap).
- Emit multiple small SEARCH/REPLACE blocks rather than one large one. A good
  refinement is typically 1–5 small blocks.
- To delete code, use an empty REPLACE section.
- To insert new code, put the existing anchor line(s) in SEARCH and put those
  same anchor line(s) plus your new lines in REPLACE.

WHEN NOT TO USE REFINE: if the right move is to scrap the current approach
and start over with a different algorithm (e.g. switch from SA to ILP), do
NOT try to express that as a giant SEARCH/REPLACE — your refinement will be
rejected. Instead, in the next decision step, choose REWRITE.

Do NOT output anything other than the SEARCH/REPLACE blocks: no commentary,
no explanation, no fenced full-file code block.
"""

GENERATION_PROMPT_REFINE_TEMPLATE = (
    _GENERATION_REFINE_PREFIX_TEMPLATE + "\n" + _GENERATION_REFINE_SUFFIX_TEMPLATE
)

_GENERATION_REWRITE_PREFIX_TEMPLATE = """\
You are an expert programmer solving an optimization task.

Task description:
{task_prompt}"""

_GENERATION_REWRITE_SUFFIX_TEMPLATE = """\

Current best score: {best_score}

Reasoning for this attempt: {reasoning}

Additional context:
{context_text}

Write a completely new solution from scratch. Do not reference or copy previous attempts.
Output ONLY valid {code_language} code in a single fenced code block like ```{code_language} ... ```.
Your code will be submitted to the task's evaluator exactly as-is, with no additional processing.
"""

GENERATION_PROMPT_REWRITE_TEMPLATE = (
    _GENERATION_REWRITE_PREFIX_TEMPLATE + "\n" + _GENERATION_REWRITE_SUFFIX_TEMPLATE
)


class DiscoveryAgent(ABC):
    """
    Abstract base for all discovery agents.

    Args:
        model_name:    Model name key (from configs/models.yaml).
        task_adapter:  TaskAdapter for the target task.
        step_budget:   Maximum number of steps for this run (default 150).
        run_id:        Unique identifier for this run.
        agency_level:  One of L0, L1, , L2b.
        llm_client:    Optional pre-constructed LLMClient; constructed if None.
        output_dir:    Directory where trajectory JSON will be saved.
    """

    def __init__(
        self,
        model_name: str,
        task_adapter: TaskAdapter,
        step_budget: int = 150,
        run_id: str = "",
        agency_level: str = "L1",
        llm_client: Optional[LLMClient] = None,
        output_dir: str = "data/trajectories",
    ) -> None:
        self.model_name = model_name
        self.task = task_adapter
        self.step_budget = step_budget
        self.run_id = run_id
        self.agency_level = agency_level
        self.output_dir = output_dir

        self.llm_client = llm_client or LLMClient(model_name)
        # Lazy analyst client — constructed on first use in _record_step
        self._analyst_client: Optional[LLMClient] = None

        # best_score initialized to the lower bound of the task's score_range
        # so score_delta is finite on the first step. The first real evaluation
        # will still become the new best (as long as it equals or exceeds the bound).
        score_range = getattr(task_adapter, "score_range", [0.0, 1.0]) or [0.0, 1.0]
        try:
            initial_best = float(score_range[0])
        except (TypeError, ValueError, IndexError):
            initial_best = 0.0

        # Seed best_code with the task's initial_program if available so that
        # (a) step-1 decisions can legitimately choose REFINE, and
        # (b) the LLM can see the expected interface before writing anything.
        try:
            initial_code = task_adapter.get_initial_code() or ""
        except Exception:
            initial_code = ""

        self.trajectory = Trajectory(
            run_id=run_id,
            task_id=task_adapter.task_id,
            model=model_name,
            level=agency_level,
            started_at=datetime.now(timezone.utc).isoformat(),
            step_budget=step_budget,
            best_score=initial_best,
            best_code=initial_code,
            metadata={
                "step_budget": step_budget,
                "seeded_with_initial_code": bool(initial_code),
            },
        )
        self._recorder = TrajectoryRecorder(self.trajectory)
        self._tokens_used: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> Trajectory:
        """Execute the full agent loop and return the completed trajectory."""
        ...

    # ------------------------------------------------------------------
    # Core query methods
    # ------------------------------------------------------------------

    def _decision_query(self, context_text: str) -> DecisionStep:
        """
        Send a decision prompt to the LLM and parse the response.

        Parses [MODE: REFINE|REWRITE] and [REASONING: ...] from the response.
        Updates tokens_used. Returns a DecisionStep.
        """
        task_prompt = self.task.get_task_prompt()
        best_score = self.trajectory.best_score
        if best_score == float("-inf"):
            best_score_str = "none yet"
        else:
            best_score_str = f"{best_score:.6f}"

        has_best_code = "yes" if self.trajectory.best_code else "no"
        cache_prefix = _DECISION_PREFIX_TEMPLATE.format(task_prompt=task_prompt)
        prompt = _DECISION_SUFFIX_TEMPLATE.format(
            best_score=best_score_str,
            has_best_code=has_best_code,
            steps_completed=len(self.trajectory.steps),
            budget_remaining=self._budget_remaining(),
            context_text=context_text,
        )

        response = self.llm_client.complete(prompt, cache_prefix=cache_prefix)
        text = response["text"]
        prompt_tokens = response["prompt_tokens"]
        completion_tokens = response["completion_tokens"]
        self._tokens_used += prompt_tokens + completion_tokens
        self.trajectory.tokens_used = self._tokens_used

        # Parse mode
        mode_match = re.search(r"\[MODE:\s*(REFINE|REWRITE)\]", text, re.IGNORECASE)
        mode = mode_match.group(1).upper() if mode_match else "REFINE"

        # Parse reasoning
        reasoning_match = re.search(r"\[REASONING:\s*(.*?)\]", text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()

        return DecisionStep(
            level=self.agency_level,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning=reasoning,
            action_type=mode,
            raw_response=text,
        )

    def _generation_query(self, mode: str, context_text: str) -> GenerationStep:
        """
        Send a generation prompt and extract code from the response.

        Uses REFINE template if mode == "REFINE", else REWRITE template.
        Extracts code from ```python ... ``` fenced block.
        Updates tokens_used. Returns a GenerationStep.
        """
        task_prompt = self.task.get_task_prompt()
        best_score = self.trajectory.best_score
        if best_score == float("-inf"):
            best_score_str = "none yet"
        else:
            best_score_str = f"{best_score:.6f}"

        reasoning = ""
        if self.trajectory.steps:
            last_step = self.trajectory.steps[-1]
            reasoning = last_step.decision.reasoning

        # Task-appropriate language (python for most, cpp for
        # frontier_cs_algorithmic). Falls back to "python" if the adapter
        # didn't set the attribute.
        code_language = getattr(self.task, "code_language", "python") or "python"

        # SEARCH/REPLACE diff mode is only used for Python tasks. C++ tasks
        # (frontier_cs_algorithmic) use full-file REFINE because: (a) the
        # initial code is only 3 lines so diff savings are negligible, and
        # (b) the <<<<<<< SEARCH markers get compiled as C++ and cause
        # "version control conflict marker" errors if any fallback path
        # leaks them through.
        use_diff_refine = (mode == "REFINE" and code_language == "python")

        if mode == "REFINE":
            best_code = self.trajectory.best_code or ""
            cache_prefix = _GENERATION_REFINE_PREFIX_TEMPLATE.format(task_prompt=task_prompt)
            if use_diff_refine:
                prompt = _GENERATION_REFINE_SUFFIX_TEMPLATE.format(
                    best_score=best_score_str,
                    best_code=best_code,
                    reasoning=reasoning,
                    context_text=context_text,
                    code_language=code_language,
                )
            else:
                # C++ / non-Python: full-file REFINE (old behavior)
                prompt = (
                    f"\nCurrent best score: {best_score_str}\n"
                    f"Current best solution:\n```{code_language}\n{best_code}\n```\n\n"
                    f"Reasoning for this attempt: {reasoning}\n\n"
                    f"Additional context:\n{context_text}\n\n"
                    f"Make targeted improvements to the solution above. "
                    f"Return the complete improved solution.\n"
                    f"Output ONLY valid {code_language} code in a single "
                    f"fenced code block like ```{code_language} ... ```.\n"
                    f"Your code will be submitted to the task's evaluator "
                    f"exactly as-is, with no additional processing.\n"
                )
        else:
            cache_prefix = _GENERATION_REWRITE_PREFIX_TEMPLATE.format(task_prompt=task_prompt)
            prompt = _GENERATION_REWRITE_SUFFIX_TEMPLATE.format(
                best_score=best_score_str,
                reasoning=reasoning,
                context_text=context_text,
                code_language=code_language,
            )

        response = self.llm_client.complete(prompt, cache_prefix=cache_prefix)
        text = response["text"]
        prompt_tokens = response["prompt_tokens"]
        completion_tokens = response["completion_tokens"]
        self._tokens_used += prompt_tokens + completion_tokens
        self.trajectory.tokens_used = self._tokens_used

        if use_diff_refine:
            # Try SEARCH/REPLACE diff parsing first; this is the cheap path
            # (LLM emits ~100-500 tokens of diffs instead of the full file).
            blocks = _parse_search_replace_blocks(text)
            if blocks:
                code, apply_errors = _apply_search_replace(
                    self.trajectory.best_code or "", blocks
                )
                if apply_errors and not any(
                    s in (self.trajectory.best_code or "") for s, _ in blocks
                ):
                    fallback = _extract_code(text, preferred_language=code_language)
                    if fallback and fallback != text.strip():
                        code = fallback
            else:
                code = _extract_code(text, preferred_language=code_language)
            # Safety: if SEARCH/REPLACE markers leaked into the final code
            # (regex edge cases, partial captures, nested fences), the code
            # will fail to run. Fall back to full-file extraction.
            if "<<<<<<< SEARCH" in code or ">>>>>>> REPLACE" in code:
                fallback = _extract_code(text, preferred_language=code_language)
                if fallback and "<<<<<<< SEARCH" not in fallback:
                    code = fallback
                else:
                    code = (self.trajectory.best_code or "")
        else:
            code = _extract_code(text, preferred_language=code_language)

        return GenerationStep(
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            code=code,
            stop_reason="end_turn",
            raw_response=text,
        )

    def _record_step(
        self,
        decision: DecisionStep,
        generation: GenerationStep,
        eval_result: dict[str, Any],
        prompt_summary: str,
        belief_state: Optional[dict[str, Any]] = None,
    ) -> TrajectoryStep:
        """
        Delegate step recording to TrajectoryRecorder, then checkpoint the
        trajectory to disk so a crash mid-run does not lose prior work.
        """
        step = self._recorder.record_step(
            decision=decision,
            generation=generation,
            eval_result=eval_result,
            prompt_summary=prompt_summary,
            belief_state=belief_state,
        )

        # Annotate strategy_label using the analyst LLM (Sonnet, cheap).
        # Failure here must never abort the run — wrap in try/except.
        try:
            from src.metrics.strategy_analyst import label_step_strategy as _label_step
            # Lazily construct the analyst client once. Override the default
            # (claude-sonnet-4-6) via env var OMC_ANALYST_MODEL so runs using
            # a non-Anthropic main model don't implicitly require Anthropic
            # credentials just for labeling.
            if self._analyst_client is None:
                import os as _os
                analyst_model = _os.environ.get("OMC_ANALYST_MODEL", "claude-sonnet-4-6")
                self._analyst_client = LLMClient(analyst_model)
            # prev_code: generation from the step before this one (or "" for step 1)
            prev_code = ""
            if len(self.trajectory.steps) >= 2:
                prev_step = self.trajectory.steps[-2]
                prev_code = prev_step.generation.code if prev_step.generation else ""
            new_code = generation.code or ""
            task_desc = (
                getattr(self.task, "description", None)
                or self.task.get_task_prompt()[:500]
            )
            label = _label_step(
                prev_code=prev_code,
                new_code=new_code,
                task_description=task_desc,
                llm_client=self._analyst_client,
            )
            if step.eval_result is None:
                step.eval_result = {}
            step.eval_result["strategy_label"] = label
        except Exception as exc:  # pragma: no cover — defensive
            print(f"[agent] strategy labeling failed at step {step.step_number}: {exc}")

        # Incremental checkpoint — overwrites the same file after every step.
        # Any exception during save should not abort the run; log and continue.
        try:
            from pathlib import Path as _Path
            out_path = _Path(self.output_dir) / f"{self.run_id}.json"
            self.trajectory.save(out_path)
        except Exception as exc:  # pragma: no cover — defensive
            print(f"[agent] incremental save failed at step {step.step_number}: {exc}")
        return step

    def _budget_remaining(self) -> int:
        """Return remaining step budget."""
        return max(0, self.step_budget - len(self.trajectory.steps))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def truncate_error_hybrid(text: str, head: int = 800, tail: int = 400) -> str:
    """
    Head+tail truncation preserving both ends of an error message.

    C++ compile errors have the key info in the first error lines; Python
    tracebacks put it at the end. Keeping both ends covers either case.
    Short messages are returned unchanged.
    """
    if not text:
        return ""
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n…[{len(text) - head - tail} chars truncated]…\n{text[-tail:]}"


def _eval_feedback_suffix(step) -> str:
    """
    Build a compact suffix string with evaluator feedback for a trajectory step.

    Appended to history lines in L1//L2b context so the agent knows:
      - WHY a step failed (error message)
      - WHY a successful step scored zero/negative (details excerpt)
    Returns empty string when there's nothing noteworthy to report.
    """
    import json as _json
    er = step.eval_result or {}
    parts = []

    if not er.get("success", True):
        err_msg = truncate_error_hybrid(er.get("error") or "")
        if err_msg:
            parts.append(f"error: {err_msg}")

    # Successful but degenerate (score <= 0): show details excerpt
    # so the agent understands what went wrong despite no crash
    if er.get("success", False) and step.score is not None and step.score <= 0:
        details = er.get("details") or {}
        if isinstance(details, dict) and details:
            try:
                det_str = _json.dumps(details, default=str)
            except Exception:
                det_str = repr(details)
            if len(det_str) > 1500:
                det_str = det_str[:1500] + "…"
            parts.append(f"details: {det_str}")

    if not parts:
        return ""
    return " | " + " | ".join(parts)


_SR_BLOCK_RE = re.compile(
    r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE",
    re.DOTALL,
)
_MAX_SEARCH_LINES = 30


def _parse_search_replace_blocks(text: str) -> list[tuple[str, str]]:
    """
    Extract (search, replace) tuples from an LLM response.

    Tolerates the standard aider/OpenEvolve format:
        <<<<<<< SEARCH
        ...
        =======
        ...
        >>>>>>> REPLACE
    The marker counts are slightly flexible (5+ angle brackets / equals signs)
    in case the LLM emits 6 vs 7 vs 8.
    """
    return _SR_BLOCK_RE.findall(text)


def _apply_search_replace(code: str, blocks: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """
    Apply each (search, replace) block in order. Returns (new_code, errors).

    Errors are recorded but do not abort: subsequent blocks are still tried
    against the partially-modified code, so a malformed block doesn't poison
    the whole step.
    """
    errors: list[str] = []
    for i, (search, replace) in enumerate(blocks, 1):
        if not search:
            errors.append(f"block {i}: empty SEARCH text")
            continue
        # Reject single-block whole-file rewrites disguised as diffs.
        # The prompt template promises a 30-line cap; enforce it here so a
        # non-compliant LLM cannot smuggle a full rewrite through REFINE.
        if search.count("\n") + 1 > _MAX_SEARCH_LINES:
            errors.append(
                f"block {i}: SEARCH spans {search.count(chr(10))+1} lines, "
                f"exceeds {_MAX_SEARCH_LINES}-line cap (use smaller targeted blocks)"
            )
            continue
        count = code.count(search)
        if count == 0:
            errors.append(f"block {i}: SEARCH text not found in current best code")
            continue
        if count > 1:
            errors.append(
                f"block {i}: SEARCH text appears {count} times — ambiguous, expand context"
            )
            continue
        code = code.replace(search, replace, 1)
    return code, errors


def _extract_code(text: str, preferred_language: str = "python") -> str:
    """
    Extract code from a fenced code block in the LLM response.

    Strategy:
      1. Closed fence with preferred language:   ```{lang}\\n...\\n```
      2. Closed fence with fallback languages:   ```python / ```cpp / ```c++ / ```c
      3. Closed generic fence:                   ```\\n...\\n```
      4. UNCLOSED opening fence with any of the above languages (handles
         responses truncated by max_tokens mid-code — take everything after
         the opening fence, strip any leftover backticks).
      5. Unclosed generic opening fence.
      6. Whole text (absolute last resort).

    Truncation-aware extraction is critical: if an LLM response hits the
    max_tokens cap mid-code, the closing ``` is missing. Without this
    fallback, the fenced-block regex fails and ``text.strip()`` would
    return the full prose+code blob, which is then written to a temp .py
    file and fails with "invalid syntax at line 1".
    """
    candidates: list[str] = [preferred_language]
    if preferred_language != "python":
        candidates.append("python")
    for lang in ("cpp", "c\\+\\+", "c"):
        if lang not in candidates:
            candidates.append(lang)

    # Step 1-2: closed fence with language tag
    for lang in candidates:
        pattern = rf"```{lang}\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Step 3: closed generic fence
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Step 4: unclosed fence with language tag (truncation recovery)
    for lang in candidates:
        pattern = rf"```{lang}\s*\n(.*)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            body = match.group(1)
            # Strip any stray backticks at end (if partial closing fence)
            return body.rstrip("` \n\t").strip()

    # Step 5: unclosed generic fence
    match = re.search(r"```\s*\n(.*)", text, re.DOTALL)
    if match:
        body = match.group(1)
        return body.rstrip("` \n\t").strip()

    return text.strip()
