"""Shared session activity observation contract: timestamp + bounded description/provenance, observation
only (notification, timeout, kill and retry policy live elsewhere). Provenance is a small closed enum of
*noun* sources; the default agent clock stamps ``unknown`` unless a writer passes ``provenance=``."""

from __future__ import annotations

import sys
import time
from contextlib import suppress
from enum import Enum
from typing import Any, Mapping, Optional

from agent.i18n import t

ACTIVITY_DESCRIPTION_MAX = 120

# Durable SessionDB heartbeat cadence. Contract: MUST stay >= 30s — the SessionDB write path is contended and
# this observation-only projection never justifies extra write pressure. A code constant on purpose (no config
# can make it a high-frequency writer); matches the kanban auto-heartbeat. force_persist and terminal
# compression stamps (see TERMINAL_COMPRESSION_PROVENANCES) are the only bypasses.
SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0


class ActivityProvenance(str, Enum):
    """Where a durable/in-memory activity stamp came from."""

    UNKNOWN = "unknown"
    # Compression writers: heartbeat, host timeout, cooldown, turn hold.
    # See #72424.
    AGENT_COMPRESSION = "agent.compression"
    AGENT_COMPRESSION_TIMEOUT = "agent.compression_timeout"
    AGENT_COMPRESSION_COOLDOWN = "agent.compression_cooldown"
    AGENT_COMPRESSION_TURNHOLD = "agent.compression_turnhold"


# Provenances that END the user-visible "compressing" phase: the host's progress timeout, a cooldown/
# backoff block, and the gateway's hygiene turn-hold. Each is written once, at a terminal edge, and must
# reach the durable projection IMMEDIATELY — the heartbeat above wrote "context compression in progress"
# moments earlier, so a rate-limited terminal stamp leaves `sessions.last_activity_description` advertising
# a compression that has already stopped, with no later writer to correct it (the agent may be idle, the
# process gone). That is the permanently "stuck / compressing" chat users see.
#
# Distinct from ``conversation_compression._TERMINAL_COMPRESSION_PROVENANCES``, which answers a different
# question — "may a detached heartbeat still overwrite this stamp?" — and deliberately excludes TURNHOLD,
# because after a turn-hold the worker may still be alive and adoptable.
TERMINAL_COMPRESSION_PROVENANCES = frozenset(
    {
        ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
        ActivityProvenance.AGENT_COMPRESSION_TURNHOLD,
    }
)


def is_terminal_compression_provenance(provenance: Optional[ActivityProvenance | str]) -> bool:
    """True when this stamp ends a compression phase and must bypass the persist rate limit."""
    return normalize_activity_provenance(provenance) in TERMINAL_COMPRESSION_PROVENANCES


def bound_activity_description(description: Optional[str]) -> str:
    """Clamp free-form activity text to the shared description budget."""
    text = (description or "").strip()
    return text if len(text) <= ACTIVITY_DESCRIPTION_MAX else text[: ACTIVITY_DESCRIPTION_MAX - 1] + "…"


def normalize_activity_provenance(provenance: Optional[ActivityProvenance | str]) -> ActivityProvenance:
    """Return a known provenance, or ``UNKNOWN`` when unset/unrecognized."""
    if isinstance(provenance, ActivityProvenance):
        return provenance
    try:
        return ActivityProvenance((provenance or "").strip())
    except ValueError:
        return ActivityProvenance.UNKNOWN


def format_iteration_progress(api_call_count: Any, max_iterations: Any) -> str:
    """``iteration N/M`` for user-facing status lines, or ``iteration N`` when the cap is unbounded.

    ``AIAgent.max_iterations`` defaults to ``sys.maxsize`` (unlimited), so printing the pair verbatim
    shows ``iteration 3/9223372036854775807`` in busy acks, heartbeats and timeout diagnostics (#102806).
    """
    try:
        cap = int(max_iterations)
    except (TypeError, ValueError):
        cap = sys.maxsize
    if cap >= sys.maxsize:
        return t("display.iteration_progress.unbounded", n=api_call_count)
    return t("display.iteration_progress.bounded", n=api_call_count, max=cap)


def reset_session_activity_persist_window(agent: Any) -> None:
    """Clear the persist rate-limit so the next stamp writes through (terminal compression labels must not stick on mid-compress text)."""
    with suppress(Exception):
        agent._session_activity_last_persist_mono = 0.0


def build_activity_snapshot(
    *,
    last_activity_at: Optional[float],
    last_activity_description: Optional[str],
    last_activity_provenance: Optional[ActivityProvenance | str] = None,
    now: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the shared activity snapshot (plus optional caller extras)."""
    when = float(last_activity_at) if last_activity_at is not None else None
    clock = float(now if now is not None else time.time())
    desc = bound_activity_description(last_activity_description)
    prov = normalize_activity_provenance(last_activity_provenance).value
    return {
        "last_activity_at": when,
        "last_activity_description": desc,
        "last_activity_provenance": prov,
        "seconds_since_activity": round(clock - when, 1) if when is not None else None,
        # Short aliases used by existing gateway/delegate readers.
        "last_activity_ts": when, "last_activity_desc": desc, "description": desc, "provenance": prov,
        **(extra or {}),
    }


class AwakeIdleMeter:
    """Removes host sleep from a polling watchdog's idle readings.

    Idle time is measured on the wall clock, which keeps running while the host sleeps, so a laptop
    that sleeps 15 minutes mid-turn wakes up "idle" for 15 minutes and the first poll after resume
    kills a turn that never had a chance to run. ``time.monotonic()`` pauses during sleep on macOS
    (``mach_absolute_time``) and Linux (``CLOCK_MONOTONIC``), so the two clocks drift apart by the
    time spent asleep. Call :meth:`measure` once per poll with the wall-clock idle seconds.
    """

    def __init__(self) -> None:
        self._wall, self._mono = time.time(), time.monotonic()
        self._asleep_s = 0.0

    def measure(self, idle_s: float) -> float:
        """Return *idle_s* minus the sleep observed since the agent's last activity."""
        wall, mono = time.time(), time.monotonic()
        slept_s = max(0.0, (wall - self._wall) - (mono - self._mono))
        # Activity since the previous sample (it is younger than the sampling interval) makes all
        # earlier credit predate it. A rarely sampled meter (stale-agent eviction samples only
        # when a message arrives) would otherwise keep charging old sleep against new idle time.
        if idle_s < wall - self._wall:
            self._asleep_s = 0.0
        self._asleep_s += slept_s
        self._wall, self._mono = wall, mono
        # The idle sample may predate sleep observed by these clock reads. Keep that new
        # credit until the next poll; only older credit can be trimmed by renewed activity.
        self._asleep_s = max(0.0, min(self._asleep_s, idle_s + slept_s))
        return max(0.0, idle_s - self._asleep_s)


def set_quota_wait(agent: Any, decision) -> None:
    """Local live wait, not a progress heartbeat; invalidate pre-wait abort claims.

    The immutable clock tuple keeps lock-free gateway observations consistent.
    Credit belongs only to the last real activity stamp; a new stamp discards it.
    """
    from agent.activity_tracking import _activity_lock
    notice = None
    with _activity_lock(agent):
        now = time.time()
        activity = getattr(agent, "_last_activity_ts", None)
        started, credit, stamp = quota_wait_clock(agent) or (None, 0.0, activity)
        if stamp != activity:
            credit = 0.0
        if decision is not None:
            if started is None:
                started = now
            status = (decision.action, decision.reason)
            message = quota_wait_message(decision)
            if (status != getattr(agent, "_quota_wait_status", None) and decision.action == "hold"
                    and decision.reason != "checking_quota"):
                notice = message  # a real pause only; a routine check or a meter fault says nothing
            agent._quota_wait_status = status
            agent._quota_wait_message = message
        else:
            if started is not None:
                credit += max(0.0, now - max(started, activity or started))
            started = None
            agent._quota_wait_status = None
            agent._quota_wait_message = None
        agent._quota_wait_clock = (started, credit, activity)
        agent._turn_liveness_activity_generation = getattr(agent, "_turn_liveness_activity_generation", 0) + 1
        agent._turn_liveness_abort_claim = None
    if notice is not None:
        import logging
        logging.getLogger(__name__).warning("%s", notice)
        # Drivers may re-enter the activity accessor. Publish first, then present
        # outside the lock; a failed presentation must never become a send retry.
        with suppress(Exception):
            agent._emit_warning(notice)


def quota_wait_clock(agent: Any) -> Optional[tuple]:
    """The ``(started, credit, stamp)`` clock ``set_quota_wait`` wrote, or None before any wait.

    Only that tuple counts: an agent-like stand-in that answers every attribute is not waiting.
    """
    clock = getattr(agent, "_quota_wait_clock", None)
    return clock if type(clock) is tuple else None


def quota_waiting(agent: Any) -> bool:
    """True while the agent is inside a live quota wait."""
    clock = quota_wait_clock(agent)
    return clock is not None and clock[0] is not None


def quota_idle_seconds(agent: Any, idle_seconds: float) -> float:
    """Exclude only live quota waits, retaining raw activity timestamps/labels."""
    started, credit, stamp = quota_wait_clock(agent) or (None, 0.0, None)
    if started is not None:
        return 0.0
    if stamp != getattr(agent, "_last_activity_ts", None):
        credit = 0.0
    return max(0.0, float(idle_seconds or 0.0) - credit)


def quota_wait_message(decision) -> str:
    """The one line shown for a real pace pause; work continues by itself when it lifts."""
    note = getattr(decision, "note", "")
    if decision.action == "hold" and note:
        return note
    return "Pacing pause: waiting for provider quota, then continuing automatically."
