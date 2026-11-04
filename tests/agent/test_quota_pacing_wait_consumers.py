"""F1/C3 and F2/C1/C3: native wait presentation and detached idle consumers.

Prepared for the parent's contained lane only. No constructors, providers, usage
children, credentials, real transports, timers or external status delivery.
"""
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from agent import quota_pacing_host as host
from agent import quota_pacing_native as native
from agent import session_activity as activity
from agent.quota_pacing import Decision, PendingAttempt, Route
from agent.turn_facade_lease import DurableTurnLease
from run_agent import AIAgent
from tui_gateway import compute_host, server


@pytest.fixture
def owner(monkeypatch):
    agent: Any = AIAgent.__new__(AIAgent)
    agent._last_activity_ts = 100.0
    agent._last_activity_desc = "completed tool result"
    agent._current_tool = None
    agent._api_call_count = 1
    agent.max_iterations = 4
    agent.iteration_budget = SimpleNamespace(used=1, max_total=4)
    agent._turn_liveness_activity_lock = threading.Lock()
    agent._pending_redirect_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_steer = None
    agent._interrupt_requested = False
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = []
    agent._model_request_active = threading.Event()
    agent._tool_interrupt_reason = None
    agent.session_id = "local-owned-session"
    agent.platform = "cli"
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent.suppress_status_output = False
    agent._notification_config = {}
    agent._print_fn = lambda *args, **kwargs: None
    agent.status_callback = None
    clock = SimpleNamespace(wall=110.0, ns=2_000_000_000_000)
    monkeypatch.setattr(activity.time, "time", lambda: clock.wall)
    monkeypatch.setattr(compute_host, "now_ns", lambda: clock.ns)
    monkeypatch.setattr(server.time, "perf_counter_ns", lambda: clock.ns)
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 600.0)
    return agent, clock


def test_native_warning_sink_observes_published_unlocked_state_once(owner):
    agent, clock = owner
    warnings, prints = [], []
    agent._print_fn = lambda text, **kwargs: prints.append(text)

    def sink(kind, message):
        lock = agent._liveness_activity_lock()
        assert lock.acquire(blocking=False), "status delivery holds the activity lock"
        try:
            assert agent._quota_wait_message == message
            assert agent._quota_wait_clock[0] is not None
            assert agent.get_activity_summary()["last_activity_at"] == 100.0
            warnings.append((kind, message))
        finally:
            lock.release()

    agent.status_callback = sink
    activity.set_quota_wait(agent, Decision("hold", "checking_quota"))
    assert warnings == [] and prints == []
    note = "Pacing for OpenAI Codex is +9.0 hours over the pace. Pausing for about 30 minutes, then continuing automatically."
    for decision in (Decision("hold", "native_hold", note=note), Decision("blocked", "credential_repair")):
        activity.set_quota_wait(agent, decision)
        clock.wall += 1000
        activity.set_quota_wait(agent, decision)
    assert warnings == [("warn", note)]  # the pause speaks once; a meter fault never does
    assert prints == [message for _, message in warnings]
    activity.set_quota_wait(agent, None)
    assert len(warnings) == 1
    assert agent._last_activity_ts == 100.0
    assert agent._last_activity_desc == "completed tool result"


def _mirror_sample(monkeypatch, agent, clock, *, started_at=90.0):
    frames = []
    # Exercise the real serializer and bridge without constructing a host/pool.
    compute = SimpleNamespace(_transport=SimpleNamespace(write=frames.append))
    compute_host.ComputeHost._emit_turn_activity(compute, "local", {"agent": agent}, "turn", started_at)
    mirror = dict(agent=agent, running=True, history_lock=threading.Lock(),
                  _compute_host_turn_id="turn")
    monkeypatch.setattr(server, "_sessions", {"local": mirror})
    monkeypatch.setattr(server, "write_json", lambda frame: pytest.fail("internal activity leaked to client"))
    assert len(frames) == 1
    server._relay_compute_host_rpc(frames[0])
    return mirror, frames[0]["params"]["activity_ns"]


@pytest.mark.parametrize("decision", [Decision("hold", "native_hold"), Decision("blocked", "credential_repair")])
def test_long_wait_is_fresh_locally_and_through_compute_mirror(owner, monkeypatch, decision):
    agent, clock = owner
    activity.set_quota_wait(agent, decision)
    clock.wall = 1110.0
    assert agent.get_activity_summary()["seconds_since_activity"] == 1010.0
    assert server._ws_orphan_turn_activity_is_fresh({"agent": agent})
    mirror, stamp = _mirror_sample(monkeypatch, agent, clock)
    assert stamp == clock.ns
    assert server._ws_orphan_turn_activity_is_fresh(mirror)
    # Even with a live quota marker on the parent double, the serialized sample
    # must age normally if the child stops reporting (never double-credit).
    clock.ns += 601_000_000_000
    assert not server._ws_orphan_turn_activity_is_fresh(mirror)
    assert agent._last_activity_ts == 100.0
    assert agent._last_activity_desc == "completed tool result"


def test_exit_wait_credit_applied_once_and_new_real_activity_resets_it(owner, monkeypatch):
    agent, clock = owner
    activity.set_quota_wait(agent, Decision("hold", "native_hold"))
    clock.wall = 1110.0
    activity.set_quota_wait(agent, None)
    clock.wall = 1711.0
    # Raw=1611; wait credit=1000; effective=611, not zero or negative.
    assert agent.get_activity_summary()["seconds_since_activity"] == 1611.0
    assert not server._ws_orphan_turn_activity_is_fresh({"agent": agent})
    mirror, stamp = _mirror_sample(monkeypatch, agent, clock)
    assert stamp == clock.ns - 611_000_000_000
    assert not server._ws_orphan_turn_activity_is_fresh(mirror)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    agent._touch_activity("actual tool progress")
    assert agent.get_activity_summary()["seconds_since_activity"] == 0.0
    clock.wall += 601.0
    assert not server._ws_orphan_turn_activity_is_fresh({"agent": agent})
    assert agent._last_activity_ts == 1711.0
    assert agent._last_activity_desc == "actual tool progress"


@pytest.mark.parametrize("stamp", [None, 80.0, 100.0])
def test_stock_idle_and_prior_turn_stamp_not_laundered(owner, monkeypatch, stamp):
    agent, clock = owner
    agent._last_activity_ts = stamp
    clock.wall = 800.0
    assert not server._ws_orphan_turn_activity_is_fresh({"agent": agent})
    mirror, serialized = _mirror_sample(monkeypatch, agent, clock)
    assert serialized == (clock.ns - 700_000_000_000 if stamp == 100.0 else None)
    assert not server._ws_orphan_turn_activity_is_fresh(mirror)


def _fake_binding(monkeypatch, on_poll):
    route = Route("local-fake", "selected", "opaque-account")
    selected = host._Selected("local-fake", "selected", None, "", "fake", "")
    monkeypatch.setattr(host, "_selected", lambda agent, request: selected)
    monkeypatch.setattr(host, "_section", lambda: {"enabled": True})
    monkeypatch.setattr(native, "bind_request", lambda *args, **kwargs:
                        SimpleNamespace(route=route, policy=lambda *args: Decision("hold", "native_hold")))
    original = PendingAttempt.wait
    mono = [0.0]

    def advance(seconds):
        mono[0] += seconds
        on_poll()

    monkeypatch.setattr(PendingAttempt, "wait", lambda attempt, policy:
                        original(attempt, policy, clock=lambda: mono[0], wait=advance))


@pytest.mark.parametrize("stop", ["user", "lease"])
def test_detached_fresh_wait_still_honors_native_stop_without_send(owner, monkeypatch, stop):
    agent, clock = owner
    sends = []
    request = {"messages": [{"role": "tool", "tool_call_id": "done", "content": "committed effect"}]}
    lease = DurableTurnLease(agent, SimpleNamespace(), agent.session_id, "fake-holder")
    lease.turn_active = True

    def poll():
        clock.wall = 1110.0
        assert server._ws_orphan_turn_activity_is_fresh({"agent": agent})
        mirror, _ = _mirror_sample(monkeypatch, agent, clock)
        assert server._ws_orphan_turn_activity_is_fresh(mirror)
        if stop == "lease":
            lease._interrupt_turn("local fake lease lost")
            assert agent._tool_interrupt_reason == "session turn lease lost"
        else:
            agent.interrupt(hard_cancel=True)
        agent.clear_interrupt()

    _fake_binding(monkeypatch, poll)
    with pytest.raises(InterruptedError):
        host.paced_call(agent, request, sends.append)
    assert sends == []
    assert request["messages"][0]["content"] == "committed effect"
    assert agent._quota_wait_status is None
    assert not agent._quota_pacing_lifetimes
    assert agent._last_activity_ts == 100.0


@pytest.mark.parametrize("failure", ["callback", "emitter"])
def test_presentation_failure_does_not_escape_into_provider_recovery(owner, monkeypatch, failure):
    agent, _clock = owner
    calls, sends = [], []

    def broken(*args):
        calls.append(args)
        raise RuntimeError("local presentation failure")

    if failure == "callback":
        agent.status_callback = broken  # Real native warning path swallows sink errors.
    else:
        agent._emit_warning = broken  # Even an unavailable presentation layer stays local.

    def poll():
        assert calls and sends == []
        assert agent._quota_wait_status == ("hold", "native_hold")
        agent.interrupt(hard_cancel=True)

    _fake_binding(monkeypatch, poll)
    with pytest.raises(InterruptedError):
        host.paced_call(agent, {"messages": []}, sends.append)
    assert len(calls) == 1
    assert sends == [] and agent._quota_wait_status is None
