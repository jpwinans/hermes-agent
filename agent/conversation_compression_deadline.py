"""Compression time limits: the configured idle and total limits, and the fence deadline shared live
with the summary stream (sibling of ``conversation_compression``).

Compression waits a total time limit for its summary. That limit stops a summary that trickles
out a token at a time; on a laptop GPU, reading a 150K-token conversation alone can take longer
than the whole limit. Time llama-server spends reading the prompt therefore pushes the fence
deadline back (``CompressionCommitFence.extend_deadline``), and hosts read the deadline from the
fence instead of adding the configured limit to the time they started waiting.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent.conversation_compression import CompressionCommitFence

# Defaults for the in-agent progress-aware wrap; mirror hermes_cli.config.DEFAULT_CONFIG["compression"] keys.
DEFAULT_CONTEXT_TIMEOUT_SECONDS = 120.0
DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS = 600.0


def resolve_context_compression_timeouts(compression_cfg: Optional[dict] = None) -> tuple[float, float]:
    """Return ``(idle_timeout_seconds, total_ceiling_seconds)``.
    ``idle_timeout_seconds <= 0`` disables the progress-aware wrapper. The ceiling is clamped to at least one
    idle window when the idle budget is positive."""
    idle = DEFAULT_CONTEXT_TIMEOUT_SECONDS
    ceiling = DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS
    cfg = compression_cfg
    if cfg is None:
        cfg = {}
        with contextlib.suppress(Exception):
            from hermes_cli.config import load_config
            raw = load_config()
            maybe = raw.get("compression", {}) if isinstance(raw, dict) else {}
            cfg = maybe if isinstance(maybe, dict) else {}
    if isinstance(cfg, dict):
        # Explicit 0/negative idle disables; a non-positive ceiling is ignored.
        with contextlib.suppress(TypeError, ValueError):
            if cfg.get("context_timeout_seconds") is not None:
                idle = float(cfg["context_timeout_seconds"])
        with contextlib.suppress(TypeError, ValueError):
            if cfg.get("context_total_ceiling_seconds") is not None and float(cfg["context_total_ceiling_seconds"]) > 0:
                ceiling = float(cfg["context_total_ceiling_seconds"])
    if idle > 0:
        ceiling = max(ceiling, idle)
        # #114594: the aux summary attempt legitimately needs up to its own budget (floor 300s), while the
        # host idle watchdog defaults to 120s. Clamp the idle window up to at least the effective aux
        # compression timeout so prep/chunking work with no streamed tokens yet is not cut off. Only raises:
        # an explicit cfg value above the aux budget is kept, and idle never exceeds the ceiling.
        from agent.auxiliary_client import _effective_aux_timeout
        _aux_budget = float(_effective_aux_timeout("compression", None))
        if _aux_budget > 0:
            ceiling = max(ceiling, _aux_budget)
            idle = max(idle, min(_aux_budget, ceiling))
    return idle, ceiling


def effective_ceiling(fence: CompressionCommitFence, ceiling: float, wait_started: float) -> float:
    """The total limit for a host that started waiting at *wait_started*: the configured
    *ceiling*, or longer once prompt reading pushed the fence deadline back."""
    deadline = fence.deadline_monotonic
    return ceiling if deadline is None else max(ceiling, deadline - wait_started)


def quota_budget_clock(fence: CompressionCommitFence, now: float) -> float:
    active = 0.0 if fence._quota_wait_started is None else now - fence._quota_wait_started
    return now - fence._quota_paused - active


def quota_deadline(fence: CompressionCommitFence, clock) -> float | None:
    with fence._quota_clock_lock:
        now = clock()
        active = 0.0 if fence._quota_wait_started is None else now - fence._quota_wait_started
        return None if fence._deadline is None else fence._deadline + active


def quota_deadline_exceeded(fence: CompressionCommitFence, clock) -> bool:
    with fence._quota_clock_lock:
        now = clock()
        deadline = quota_deadline(fence, lambda: now)
        return deadline is not None and now >= deadline


def account_quota_wait(fence: CompressionCommitFence, decision, clock) -> bool:
    """Exclude verified waiting from idle AND absolute precommit consumption.

    Entry must precede both expiries; cancellation/ownership revocation still win.
    Exit freezes the extension for the fallback's remaining shared budget.
    """
    from agent.quota_pacing import Decision
    if decision is not None and (not isinstance(decision, Decision) or decision.action not in ("hold", "blocked")):
        return False
    if not fence._lock.acquire(blocking=False):
        return False
    try:
        with fence._quota_clock_lock:
            now = clock()
            if decision is None:
                if fence._quota_wait_started is not None:
                    paused = now - fence._quota_wait_started
                    fence._quota_paused += paused
                    if fence._deadline is not None:
                        fence._deadline += paused
                    fence._quota_wait_started = None
                return not fence.is_cancelled
            if fence.is_cancelled or fence._commit_started:
                return False
            if fence._idle_timeout_seconds is not None and fence.seconds_since_progress() >= fence._idle_timeout_seconds:
                return False
            if fence._quota_wait_started is None:
                fence._quota_wait_started = now
            return True
    finally:
        fence._lock.release()


@contextlib.contextmanager
def compression_deadline_scope(fence: Optional[CompressionCommitFence]):
    """Share *fence*'s deadline with the summary stream, live, and push it back while the
    server reads the prompt. A stream checks the deadline as it reads, so a later summary
    call in the same attempt sees the extended deadline too."""
    from agent.auxiliary_client import aux_stream_deadline
    from agent.local_prompt_progress import prompt_reading_hook

    if fence is None:
        yield
        return
    with aux_stream_deadline(lambda: fence.deadline_monotonic, quota_wait=fence.quota_wait), prompt_reading_hook(fence.extend_deadline):
        yield
