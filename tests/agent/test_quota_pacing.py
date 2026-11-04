"""Live caller preservation and cancellation contracts for the pure pacing gate.

No provider/client/credential imports, network, account probes or real sleeps.
These are gate tests, not proof that the unmodified native seams are integrated.
"""

import asyncio

import pytest

from agent.quota_pacing import Decision, Lifetime, PacingStopped, PendingAttempt, Route


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []
        self.on_wait = lambda: None

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds
        self.on_wait()

    async def await_wait(self, seconds):
        self.wait(seconds)


def make_pending(**kwargs):
    route = kwargs.pop("route", Route("provider", "model", "account"))
    events = kwargs.pop("on_waiting", lambda state: None)
    return PendingAttempt(
        route=route, lifetime=kwargs.pop("lifetime", Lifetime()),
        current_route=kwargs.pop("current_route", lambda: route),
        on_waiting=events, **kwargs,
    )


def run_wait(pending, policy, clock, asynchronous):
    if asynchronous:
        async def async_policy(*args):
            return policy(*args)
        asyncio.run(pending.wait_async(async_policy, clock=clock, wait=clock.await_wait))
    else:
        pending.wait(policy, clock=clock, wait=clock.wait)


def send_original(pending, policy, clock, asynchronous, original, sends, fallback):
    # Models the live caller. A provider Exception would enter recovery; a held
    # attempt and PacingStopped must not. No serializer, replay, or message rebuild.
    try:
        run_wait(pending, policy, clock, asynchronous)
        pending.check_before_send()
        sends.append(original)
    except Exception:
        fallback.append("provider_failure")
        raise


@pytest.mark.parametrize("asynchronous", [False, True])
def test_release_sends_original_inputs_once_without_replaying_effects(asynchronous):
    clock, states, sends, fallback = Clock(), [], [], []
    completed_effects = ["already_applied"]
    messages = [{"role": "user", "content": "original goal"},
                {"role": "assistant", "tool_calls": [{"id": "done"}]},
                {"role": "tool", "tool_call_id": "done", "content": "committed result"}]
    original = {"messages": messages, "tools": [{"name": "existing_tool"}]}
    hold = object()
    seen = []
    pending = make_pending(on_waiting=states.append)

    def policy(route, active_hold, budget):
        seen.append((route, active_hold, budget))
        if len(seen) < 3:
            return Decision("hold", "quota_window", retry_after=2, active_hold=hold)
        return Decision("allow", "native_released")

    send_original(pending, policy, clock, asynchronous, original, sends, fallback)
    assert sends == [original]
    assert sends[0] is original and sends[0]["messages"] is messages
    assert messages[-1]["content"] == "committed result"
    assert completed_effects == ["already_applied"]
    assert fallback == []
    assert [row[1] for row in seen] == [None, hold, hold]
    assert [state.action if state else None for state in states] == ["hold", None]
    assert clock.now == 4
    assert all(0 < seconds <= pending.cancel_quantum for seconds in clock.waits)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_cancelled_wait_never_sends_or_enters_provider_fallback(asynchronous):
    clock, states, sends, fallback = Clock(), [], [], []
    pending = make_pending(on_waiting=states.append)
    original = {"messages": [{"role": "user", "content": "keep me"}]}
    clock.on_wait = pending.lifetime.cancel
    calls = []

    def policy(*args):
        calls.append(args)
        return Decision("hold", "quota_window", 30, active_hold="native_hold")

    with pytest.raises(PacingStopped, match="cancelled"):
        send_original(pending, policy, clock, asynchronous, original, sends, fallback)
    assert sends == fallback == []
    assert len(calls) == 1
    assert original["messages"][0]["content"] == "keep me"
    assert pending.active_hold == "native_hold"
    assert states[-1] is None
    # Irreversible retirement also rejects a later release or late worker send.
    with pytest.raises(PacingStopped, match="cancelled"):
        pending.check_before_send()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("bad", [None, {}, Decision("unknown", "bad"),
                                Decision("allow", "raw provider text"),
                                Decision("allow", "bad", float("nan")),
                                Decision("allow", "bad", -1)])
def test_malformed_decision_proceeds_unmetered_keeping_native_hold(asynchronous, bad):
    clock, states = Clock(), []
    hold = object()
    pending = make_pending(active_hold=hold, on_waiting=states.append)
    observed = []

    def policy(route, active_hold, budget):
        observed.append(active_hold)
        if len(observed) == 1:
            return bad
        assert clock.now >= pending.poll_min
        return Decision("allow", "repaired")

    run_wait(pending, policy, clock, asynchronous)
    assert observed == [hold]  # a meter fault never waits
    assert states == []  # nor says anything
    assert pending.decision.reason == "meter_unavailable"
    assert pending.active_hold is hold


@pytest.mark.parametrize("asynchronous", [False, True])
def test_meter_fault_proceeds_without_waiting_or_notice(asynchronous):
    clock, states = Clock(), []
    hold = object()
    pending = make_pending(active_hold=hold, on_waiting=states.append)
    repaired = []
    clock.on_wait = lambda: repaired.append(True)
    calls = []

    def policy(route, active_hold, budget):
        calls.append(active_hold)
        if not repaired:
            return Decision("blocked", "reauth_required", 0)
        return Decision("allow", "native_repaired")

    run_wait(pending, policy, clock, asynchronous)
    assert calls == [hold]
    assert states == []
    assert clock.now == 0


@pytest.mark.parametrize("asynchronous", [False, True])
def test_held_work_can_cancel_without_policy_release(asynchronous):
    clock, states = Clock(), []
    pending = make_pending(on_waiting=states.append, active_hold="retained")
    clock.on_wait = pending.lifetime.cancel
    with pytest.raises(PacingStopped, match="cancelled"):
        run_wait(pending, lambda *args: Decision("hold", "native_hold"), clock, asynchronous)
    assert pending.active_hold == "retained"
    assert states[0].action == "hold" and states[-1] is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_model_account_and_provider_have_independent_decisions(asynchronous):
    held = Route("provider", "large", "account_a")
    routes = [held, Route("provider", "small", "account_a"),
              Route("provider", "large", "account_b"),
              Route("other_provider", "large", "account_a")]
    seen = []
    for route in routes:
        clock = Clock()
        pending = make_pending(route=route)
        clock.on_wait = pending.lifetime.cancel

        def policy(actual, active_hold, budget):
            seen.append(actual)
            return Decision("hold" if actual == held else "allow", "native_scope")

        if route == held:
            with pytest.raises(PacingStopped):
                run_wait(pending, policy, clock, asynchronous)
        else:
            run_wait(pending, policy, clock, asynchronous)
            assert clock.waits == []
    assert seen == routes


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("change", ["cancel", "route"])
def test_final_waiting_exit_hook_cannot_authorize_late_send(asynchronous, change):
    clock, sends, fallback = Clock(), [], []
    original_route = Route("provider", "model", "account")
    current = [original_route]
    lifetime = Lifetime()

    def state_hook(state):
        if state is None:
            if change == "cancel":
                lifetime.cancel()
            else:
                current[0] = Route("provider", "model", "different_account")

    pending = make_pending(route=original_route, lifetime=lifetime,
                           current_route=lambda: current[0], on_waiting=state_hook)
    decisions = iter([Decision("hold", "quota"), Decision("allow", "released")])
    with pytest.raises(PacingStopped):
        send_original(pending, lambda *args: next(decisions), clock, asynchronous,
                      {"messages": []}, sends, fallback)
    assert sends == fallback == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_final_boundary_rechecks_route_after_allow(asynchronous):
    clock = Clock()
    current = [Route("provider", "model", "account")]
    pending = make_pending(route=current[0], current_route=lambda: current[0])
    run_wait(pending, lambda *args: Decision("allow", "available"), clock, asynchronous)
    current[0] = Route("provider", "changed_model", "account")
    with pytest.raises(PacingStopped, match="route_changed"):
        pending.check_before_send()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["raises", "over_budget"])
def test_adapter_failure_or_late_answer_proceeds_unmetered(asynchronous, failure):
    clock, states = Clock(), []
    pending = make_pending(on_waiting=states.append, active_hold="native_hold")
    calls = []

    def policy(route, active_hold, budget):
        calls.append((active_hold, budget))
        if len(calls) > 1:
            return Decision("allow", "repaired")
        if failure == "raises":
            raise RuntimeError("must never expose raw credential-bearing error")
        clock.now += budget + 1
        return Decision("allow", "too_late")

    run_wait(pending, policy, clock, asynchronous)
    assert calls == [("native_hold", pending.adapter_budget)]
    assert states == []
    assert pending.decision.reason == "meter_unavailable"


def test_async_task_cancellation_unwinds_waiting_state_without_send():
    async def scenario():
        entered, blocked = asyncio.Event(), asyncio.Event()
        states, sends = [], []
        pending = make_pending(on_waiting=states.append)

        async def policy(*args):
            return Decision("hold", "quota")

        async def wait(seconds):
            entered.set()
            await blocked.wait()

        async def caller():
            await pending.wait_async(policy, wait=wait)
            pending.check_before_send()
            sends.append("must_not_send")

        task = asyncio.create_task(caller())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sends == [] and states[-1] is None
        with pytest.raises(PacingStopped, match="cancelled"):
            pending.check_before_send()

    asyncio.run(scenario())


def test_final_check_rejects_attempt_that_never_waited():
    with pytest.raises(PacingStopped, match="not_admitted"):
        make_pending().check_before_send()


def test_cancel_before_entry_does_not_even_poll():
    lifetime = Lifetime()
    lifetime.cancel()
    pending = make_pending(lifetime=lifetime)
    with pytest.raises(PacingStopped, match="cancelled"):
        pending.wait(lambda *args: pytest.fail("retired work polled"))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_wait_polling_is_clamped_and_does_not_spam_state(asynchronous):
    clock, states = Clock(), []
    pending = make_pending(on_waiting=states.append, poll_min=2, poll_max=3)
    times = []

    def policy(*args):
        times.append(clock.now)
        if len(times) <= 2:
            return Decision("hold", "quota", 0 if len(times) == 1 else 10000)
        return Decision("allow", "released")

    run_wait(pending, policy, clock, asynchronous)
    assert times == [0, 2, 5]
    assert [state.action if state else None for state in states] == ["hold", None]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_cancellation_during_adapter_call_rejects_its_late_allow(asynchronous):
    clock, sends, fallback = Clock(), [], []
    pending = make_pending()

    def policy(*args):
        pending.lifetime.cancel()
        return Decision("allow", "late_release")

    with pytest.raises(PacingStopped, match="cancelled"):
        send_original(pending, policy, clock, asynchronous, {}, sends, fallback)
    assert sends == fallback == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_waiting_hook_failure_cannot_become_provider_fallback(asynchronous):
    clock, sends, fallback = Clock(), [], []

    def hook(state):
        raise RuntimeError("owner state could not be marked")

    pending = make_pending(on_waiting=hook)
    with pytest.raises(PacingStopped, match="waiting_hook_failed"):
        send_original(pending, lambda *args: Decision("hold", "quota"), clock,
                      asynchronous, {}, sends, fallback)
    assert sends == fallback == []


@pytest.mark.parametrize("field", ["provider", "model", "account"])
def test_route_requires_explicit_actual_identity(field):
    values = dict(provider="provider", model="model", account="account")
    values[field] = ""
    with pytest.raises(ValueError):
        Route(**values)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_new_logical_wait_clears_previous_allow_before_policy_runs(asynchronous):
    clock, sends = Clock(), []
    pending = make_pending()
    run_wait(pending, lambda *args: Decision("allow", "available"), clock, asynchronous)
    pending.check_before_send()
    sends.append("first_logical_attempt")

    def next_policy(*args):
        # A prior attempt's allow cannot authorize work during this fresh wait.
        with pytest.raises(PacingStopped, match="not_admitted"):
            pending.check_before_send()
        return Decision("hold", "quota")

    clock.on_wait = pending.lifetime.cancel
    with pytest.raises(PacingStopped, match="cancelled"):
        run_wait(pending, next_policy, clock, asynchronous)
    assert sends == ["first_logical_attempt"]
