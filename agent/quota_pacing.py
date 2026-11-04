"""Cooperative quota waiting for one live, unsent logical inference attempt.

No routing, credentials, persistence, provider calls, or background workers live
here. The caller retains its original inputs and owns the final send boundary.
Native adapters own thresholds, shared account latches and fresh quota polling.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Awaitable, Callable


@dataclass(frozen=True)
class Route:
    """Actual resolved destination; account is stable native identity, not a token."""

    provider: str
    model: str
    account: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.provider, self.model, self.account)):
            raise ValueError("explicit provider, model and account required")


@dataclass(frozen=True)
class Decision:
    """Reason is a nonsecret code, NEVER raw provider text or credential data.

    active_hold is the native policy's opaque hysteresis state, not a gate latch.
    Blocked means the meter could not read usage. It never stops work: the wait
    proceeds as if allowed, quietly. Only a hold, a real pace pause, waits.
    note is the human message for a hold, built from native numbers, never secrets.
    """

    action: str
    reason: str
    retry_after: float = 1.0
    active_hold: object = None
    note: str = ""


class PacingStopped(BaseException):
    """Control flow, deliberately outside broad provider Exception recovery."""


class Lifetime:
    """One-way cancellation for a pending attempt; never clear or reuse it."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def check(self, route: Route, current_route: Callable[[], Route]) -> None:
        if self._cancelled.is_set():
            raise PacingStopped("cancelled")
        try:
            actual = current_route()
        except Exception:
            raise PacingStopped("route_unavailable") from None
        if actual != route:
            raise PacingStopped("route_changed")
        if self._cancelled.is_set():
            raise PacingStopped("cancelled")

    def wait(self, seconds: float) -> None:
        self._cancelled.wait(seconds)


# Adapters must return within budget, bound ALL their native I/O by that budget,
# and apply native stale/initial-unknown quota policy within that budget.
# Sync Python callbacks cannot be forcibly timed out here without orphan workers.
Policy = Callable[[Route, object, float], Decision]
AsyncPolicy = Callable[[Route, object, float], Awaitable[Decision]]
WaitingHook = Callable[[Decision | None], None]


class PendingAttempt:
    """Single-owner wait state. Keep this object through repair, not per poll.

    All inputs are explicit, including cancellation and the waiting-state hook;
    no parent-context admission is inherited. The hook receives hold/blocked on
    entry/change and None on exit. It must update actual owner waiting state,
    not a fake progress pulse. Each logical attempt and outer retry calls wait
    afresh; check_before_send only rechecks that admitted frame's route/cancel.
    Native bounded transport retries belong to the already-admitted attempt.
    current_route and on_waiting must be bounded local callbacks,
    never credential discovery, network calls or routing side effects.
    """

    def __init__(
        self, *, route: Route, lifetime: Lifetime,
        current_route: Callable[[], Route], on_waiting: WaitingHook,
        active_hold: object = None, poll_min: float = 1.0,
        poll_max: float = 60.0, cancel_quantum: float = 0.25,
        adapter_budget: float = 5.0,
    ) -> None:
        values = (poll_min, poll_max, cancel_quantum, adapter_budget)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("positive finite pacing bounds required")
        if poll_min > poll_max:
            raise ValueError("poll_min exceeds poll_max")
        self.route = route
        self.lifetime = lifetime
        self.current_route = current_route
        self.on_waiting = on_waiting
        self.active_hold = active_hold
        self.poll_min = poll_min
        self.poll_max = poll_max
        self.cancel_quantum = cancel_quantum
        self.adapter_budget = adapter_budget
        self.decision: Decision | None = None
        self._waiting: Decision | None = None

    def _check_current(self) -> None:
        self.lifetime.check(self.route, self.current_route)

    def check_before_send(self) -> None:
        self._check_current()
        if self.decision is None or self.decision.action != "allow":
            raise PacingStopped("not_admitted")

    def _accept(self, value: object) -> Decision:
        valid = (
            isinstance(value, Decision)
            and value.action in ("allow", "hold", "blocked")
            and isinstance(value.reason, str)
            and 0 < len(value.reason) <= 80
            and value.reason.isascii()
            and all(c.isalnum() or c in "_-" for c in value.reason)
            and isinstance(value.retry_after, (int, float))
            and not isinstance(value.retry_after, bool)
            and 0 <= value.retry_after < math.inf
        )
        decision = value if valid and isinstance(value, Decision) else Decision(
            "blocked", "invalid_policy_decision"
        )
        if not (isinstance(decision.note, str) and len(decision.note) <= 240 and decision.note.isprintable()):
            decision = Decision(decision.action, decision.reason, decision.retry_after, decision.active_hold)
        if decision.action == "blocked" and decision.reason != "cancelled":
            # A meter that cannot check usage is not over pace. Never stop or warn for it.
            logging.getLogger(__name__).info("quota pacing unavailable (%s); proceeding", decision.reason)
            decision = Decision("allow", "meter_unavailable", active_hold=decision.active_hold)
        if decision.active_hold is not None:
            self.active_hold = decision.active_hold
        self.decision = decision
        return decision

    def _notify(self, value: Decision | None) -> None:
        # Compare only public state, not opaque native hold objects.
        old = self._waiting
        if old is None and value is None:
            return
        if old is not None and value is not None:
            if (old.action, old.reason) == (value.action, value.reason):
                return
        self._waiting = value
        try:
            self.on_waiting(value)
        except Exception:
            raise PacingStopped("waiting_hook_failed") from None

    def _delay(self, decision: Decision) -> float:
        return min(self.poll_max, max(self.poll_min, decision.retry_after))

    def wait(
        self, policy: Policy, *, clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], None] | None = None,
    ) -> None:
        """Return only on allow; held/blocked work stays in the live caller.

        Exceptions from the adapter fail closed with a fixed nonsecret reason.
        A wait callback must actually wait (or advance an injected test clock).
        """
        self.decision = None
        wait = self.lifetime.wait if wait is None else wait
        try:
            while True:
                self._check_current()
                started = clock()
                try:
                    value = policy(self.route, self.active_hold, self.adapter_budget)
                except Exception:
                    value = Decision("blocked", "policy_unavailable")
                if clock() - started > self.adapter_budget:
                    value = Decision("blocked", "policy_deadline")
                self._check_current()
                decision = self._accept(value)
                if decision.action == "allow":
                    break
                self._notify(decision)
                deadline = clock() + self._delay(decision)
                while True:
                    self._check_current()
                    remaining = deadline - clock()
                    if remaining <= 0:
                        break
                    wait(min(remaining, self.cancel_quantum))
        finally:
            self._notify(None)
        # Exit hooks can cancel or change route, including after allow.
        self.check_before_send()

    async def wait_async(
        self, policy: AsyncPolicy, *, clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Async twin; native task cancellation is never turned into a decision.

        Async adapters must cooperate with task cancellation and obey the budget;
        no unowned task is spawned to race a hung adapter.
        """
        self.decision = None
        try:
            while True:
                self._check_current()
                started = clock()
                try:
                    value = await policy(self.route, self.active_hold, self.adapter_budget)
                except Exception:
                    value = Decision("blocked", "policy_unavailable")
                if clock() - started > self.adapter_budget:
                    value = Decision("blocked", "policy_deadline")
                self._check_current()
                decision = self._accept(value)
                if decision.action == "allow":
                    break
                self._notify(decision)
                deadline = clock() + self._delay(decision)
                while True:
                    self._check_current()
                    remaining = deadline - clock()
                    if remaining <= 0:
                        break
                    await wait(min(remaining, self.cancel_quantum))
        except asyncio.CancelledError:
            self.lifetime.cancel()
            raise
        finally:
            self._notify(None)
        self.check_before_send()
