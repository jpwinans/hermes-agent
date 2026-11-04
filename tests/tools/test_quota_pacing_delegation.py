"""Source-only native delegation wait contracts; run only in released containment.

Owned in-process AIAgents, inert SDK objects and a fake binding. No child CLI,
bootstrap, native usage reader, credential discovery or provider sends. The real
heartbeat, configured wait, synchronous join and detached stale sweep are used.
"""
from types import SimpleNamespace
from typing import Any
import threading
import time

import pytest

from agent import quota_pacing_host as host, quota_pacing_native as native
from agent import session_activity as activity
from agent.activity_tracking import _activity_lock
from agent.quota_pacing import Decision, PendingAttempt, Route
from agent.turn_liveness import TurnLivenessWatchdog
from run_agent import AIAgent
from tools import async_delegation as detached
from tools import delegate_tool as facade
from tools import delegate_tool_child_run as child_run
from tools import delegate_tool_dispatch as dispatch


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=100.0)
    monkeypatch.setattr(time, "time", lambda: state.now)
    monkeypatch.setattr(time, "monotonic", lambda: state.now)
    monkeypatch.setattr(facade, "_HEARTBEAT_INTERVAL", 1.0)
    monkeypatch.setattr(facade, "_HEARTBEAT_STALE_CYCLES_IDLE", 3)
    monkeypatch.setattr(facade, "_HEARTBEAT_STALE_CYCLES_IN_TOOL", 5)
    monkeypatch.setattr(detached, "_STALE_IDLE_SECONDS", 3.0)
    monkeypatch.setattr(detached, "_STALE_IN_TOOL_SECONDS", 5.0)
    return state


def owner(name):
    """No constructor: native control/activity methods, no service/client startup."""
    from openai import OpenAI
    import httpx
    agent: Any = AIAgent.__new__(AIAgent)
    agent.provider = "openai-codex"
    agent.model = name + "-model"
    agent.api_mode = "chat_completions"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent._client_kwargs = {"api_key": "owned-fake-token", "base_url": agent.base_url}
    agent.client = object.__new__(OpenAI)
    agent.client.api_key = "owned-fake-token"
    agent.client._api_key_provider = None
    agent.client._base_url = httpx.URL(agent.base_url)
    agent.session_id = name
    agent.platform = "subagent"
    agent.quiet_mode = True
    agent.thinking_callback = None
    agent._pending_redirect_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_steer = None
    agent._model_request_active = threading.Event()
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_interrupt_reason = None
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = []
    agent._last_activity_ts = 100.0
    agent._last_activity_desc = "committed tool result"
    agent._turn_liveness_activity_generation = 0
    agent._turn_liveness_abort_claim = None
    agent._current_tool = None
    agent._api_call_count = 1
    agent.max_iterations = 4
    agent.iteration_budget = SimpleNamespace(used=1, max_total=4)
    agent._persist_session_activity_if_due = lambda: None
    return agent


def watchdog(agent):
    return TurnLivenessWatchdog(
        agent, session_id=agent.session_id, timeout_s=3.0, poll_s=1.0,
        stop_event=threading.Event(), activity_lock=_activity_lock(agent),
        is_turn_active=lambda: True, commit_abort=lambda *_: False,
        deactivate_turn=lambda: None,
    )


def idle(watch):
    sample = watch._sample()
    assert sample is not None
    return sample.idle_seconds


@pytest.fixture
def gate(monkeypatch, clock):
    state = SimpleNamespace(bindings=[], observe=None, held_model="held-model", held=False)
    real_wait = PendingAttempt.wait
    monkeypatch.setattr(host, "_section", lambda: {"enabled": True})

    def bind(config, **kwargs):
        state.bindings.append(kwargs)
        route = Route(kwargs["provider"], kwargs["model"], "owned-account")

        def policy(actual, hold, budget):
            assert actual == route
            if actual.model == state.held_model and not state.held:
                return Decision("hold", "owned_native_hold", retry_after=1.0)
            return Decision("allow", "owned_release")

        return SimpleNamespace(route=route, policy=policy)

    def wait(attempt, policy):
        def advance(seconds):
            assert state.observe is not None
            state.observe()
            state.held = True
        return real_wait(attempt, policy, clock=lambda: clock.now, wait=advance)

    monkeypatch.setattr(native, "bind_request", bind)
    monkeypatch.setattr(PendingAttempt, "wait", wait)
    return state


@pytest.mark.parametrize("decision", [Decision("hold", "owned_hold"), Decision("blocked", "credential_repair")])
def test_heartbeat_excludes_real_wait_without_progress_and_keeps_stale_teeth(clock, decision):
    child, parent = owner("held"), owner("parent")
    touches = []
    parent._touch_activity = touches.append
    hb = child_run._Heartbeat(child, parent, 0)
    activity.set_quota_wait(child, decision)
    before = child_run._child_activity_fingerprint(child)
    for _ in range(10):
        clock.now += 10
        assert hb.tick() is None
    assert not hb.settled.is_set()
    assert child_run._child_activity_fingerprint(child) == before
    assert touches == []
    activity.set_quota_wait(child, None)
    assert child_run._quota_stale_seconds(child, 100) == 0
    # No quota credit is semantic progress; ordinary idle eventually expires.
    clock.now += 10
    for _ in range(4):
        hb.tick()
    assert hb.settled.is_set()
    assert hb.stale_threshold_seconds == 3.0


@pytest.mark.parametrize("cap", [None, 3.0])
def test_actual_configured_wait_survives_hold_then_accounts_post_wait_idle(clock, cap):
    child, parent = owner("held"), owner("parent")
    run = child_run._ChildRun(child, parent, 0, "original goal", None, None)
    activity.set_quota_wait(child, Decision("blocked", "credential_repair"))
    steps = []

    class Settled:
        def is_set(self):
            return cap is None and len(steps) == 4

        def wait(self, timeout):
            steps.append(timeout)
            if len(steps) <= 3:
                clock.now += 100
                assert child_run._quota_stale_seconds(child, 300) == 0
            else:
                activity.set_quota_wait(child, None)
                clock.now += 4
            assert len(steps) <= 5, "wait must not retain quota credit as permanent exemption"

    run.wait_liveness_aware(Settled(), SimpleNamespace(), cap)
    assert len(steps) >= 4
    assert child._api_call_count == 1
    assert child._last_activity_ts == 100.0


def make_batch(parent, children):
    return dispatch._Batch(
        task_list=[{"goal": "original " + c.session_id} for c in children],
        children=[(i, {"goal": "original " + c.session_id}, c) for i, c in enumerate(children)],
        parent_agent=parent, creds={"model": None}, context="original context",
        top_role="leaf", max_children=2, live_deleg_id=None, live_writers=[],
        live_paths=[], origin_wake_sid="", origin_ui_session_id="",
        origin_owner_transport=None, origin_owner_session_record=None,
        origin_session_history_delivery=False, overall_start=time.monotonic(),
    )


@pytest.mark.parametrize("background", [False, True])
def test_real_batch_retains_finished_sibling_during_own_route_wait(clock, gate, monkeypatch, background):
    parent, sibling, held = owner("parent"), owner("sibling"), owner("held")
    batch = make_batch(parent, [sibling, held])
    parent._active_children = [] if background else [sibling, held]
    effects, sends, reports, recorded, sibling_sends = [], [], [], [], []
    sibling_reported = threading.Event()
    request = {"model": held.model, "messages": [{"role": "tool", "content": "held prior effect"}]}
    hb = child_run._Heartbeat(held, parent, 1)
    parent_watch = watchdog(parent)
    finished = {"task_index": 0, "status": "completed", "summary": "saved sibling data", "tool_trace": [{"result": "once"}]}
    held_done = {"task_index": 1, "status": "completed", "summary": "original held result"}
    monkeypatch.setattr(dispatch, "_finalize_child_results", lambda *args: None)
    monkeypatch.setattr("tools.delegation_live_log.update_manifest_statuses", lambda *args, **kwargs: None)
    monkeypatch.setattr(dispatch, "finish_delegation_unit", lambda *args, **kwargs: None)
    monkeypatch.setattr(dispatch, "record_unit_child", lambda unit, entry: recorded.append((unit, entry.copy())))
    batch.unit_id = "owned-unit" if background else None

    def report(parent, spinner, entry, *args):
        reports.append(entry)
        if entry["task_index"] == 0:
            sibling_reported.set()
    monkeypatch.setattr(dispatch, "_report_child_done", report)

    monitor_record = {
        "delegation_id": "owned-unit", "status": "running", "_started": True,
        "progress_fn": lambda: dispatch._batch_progress_token([sibling, held]),
        "_progress_token": None, "_progress_ts": clock.now,
    }
    monkeypatch.setattr(detached, "_records", {"owned-unit": monitor_record})

    def observe():
        assert sibling_reported.wait(5), "real batch must collect finished sibling before release"
        initial_token = dispatch._batch_progress_token([sibling, held])[0]
        assert hb.tick() is None
        with detached._records_lock:
            detached._sweep_stale_locked(clock.now)
        for _ in range(10):
            clock.now += 100
            assert hb.tick() is None
            with detached._records_lock:
                stalled, expired, monitorable = detached._sweep_stale_locked(clock.now)
            assert stalled == expired == [] and monitorable
            assert monitor_record["status"] == "running"
            assert dispatch._batch_progress_token([sibling, held])[0] == initial_token
            assert idle(parent_watch) == (clock.now - 100 if background else 0)
            assert parent._last_activity_ts == held._last_activity_ts == 100.0
            assert sends == []
            assert effects == ["sibling effect"]
            assert reports == [finished]
            assert finished["tool_trace"] == [{"result": "once"}]
        assert not hb.settled.is_set()
        assert monitor_record["_progress_ts"] == 100.0
        if background:
            assert recorded == [("owned-unit", finished)]
            assert getattr(parent, "_quota_wait_status", None) is None

    gate.observe = observe

    def run(i, task, child):
        if i == 0:
            effects.append("sibling effect")
            return host.paced_call(child, {"model": child.model, "messages": []},
                                   lambda original, **private: sibling_sends.append(original) or finished)
        response = host.paced_call(child, request, lambda original, **private: sends.append(original) or held_done)
        return response
    batch.run_child = run
    combined = dispatch._execute_and_aggregate(batch, honor_parent_interrupt=not background)
    assert combined["results"] == [finished, held_done]
    assert sends == [request] and sends[0] is request
    assert effects == ["sibling effect"]
    assert held._last_activity_ts == 100.0
    assert getattr(parent, "_quota_wait_status", None) is None
    assert getattr(parent, "_quota_wait_delegation_children", None) is None
    assert {b["model"] for b in gate.bindings} == {held.model, sibling.model}
    assert {b["session_id"] for b in gate.bindings} == {held.session_id, sibling.session_id}
    assert len(sibling_sends) == 1
    assert all(b["model"] != parent.model for b in gate.bindings)
    # Wait exit alone must not make either parent or detached unit immediately stale.
    assert child_run._quota_stale_seconds(held, 1000) == 0
    if not background:
        assert idle(parent_watch) == 0
    clock.now += 4
    with detached._records_lock:
        stalled, _, _ = detached._sweep_stale_locked(clock.now)
    assert stalled and stalled[0][0] == "owned-unit"
    if not background:
        assert idle(parent_watch) == 4


@pytest.mark.parametrize("reason", ["explicit stop requested", "session turn lease lost", "gateway shutdown"])
@pytest.mark.parametrize("detached_mode", [False, True])
def test_native_parent_hard_cancel_fanout_is_irreversible(clock, gate, reason, detached_mode, monkeypatch):
    parent, held = owner("parent"), owner("held")
    parent._active_children = [] if detached_mode else [held]
    if detached_mode:
        monkeypatch.setattr(detached, "_records", {"owned": {
            "delegation_id": "owned", "parent_session_id": parent.session_id, "status": "running",
            "interrupt_fn": lambda: child_run._signal_child_stop(held, "explicit detached stop"),
        }})
    sends = []

    def cancel():
        parent.interrupt(hard_cancel=True, tool_reason=reason)
        if detached_mode:
            assert not held._interrupt_requested  # ordinary turn stop must NOT reach detached work
            assert detached.interrupt_for_session(parent_session_id=parent.session_id, reason="session_end") == 1
        held.clear_interrupt()
        parent.clear_interrupt()
        clock.now += 1000
        assert not held._interrupt_requested

    gate.observe = cancel
    with pytest.raises(InterruptedError):
        host.paced_call(held, {"model": held.model, "messages": [{"role": "tool", "content": "already done"}]}, sends.append)
    assert sends == []
    assert held._interrupt_requested
    assert not held._quota_pacing_lifetimes
    assert held._quota_wait_status is None


@pytest.mark.parametrize("failure", [False, True])
def test_single_join_scopes_parent_marker_and_unwinds_without_phantom_activity(clock, monkeypatch, failure):
    parent, child = owner("parent"), owner("held")
    batch = make_batch(parent, [child])
    activity.set_quota_wait(child, Decision("blocked", "credential_repair"))
    before = parent._last_activity_ts
    monkeypatch.setattr(dispatch, "_finalize_child_results", lambda *args: None)
    monkeypatch.setattr("tools.delegation_live_log.update_manifest_statuses", lambda *args, **kwargs: None)
    monkeypatch.setattr(dispatch, "finish_delegation_unit", lambda *args, **kwargs: None)

    def run(i, task, child):
        hb = child_run._Heartbeat(child, parent, 0)
        clock.now += 1000
        assert hb.tick() is None
        assert idle(watchdog(parent)) == 0
        assert parent._last_activity_ts == before
        if failure:
            raise ValueError("owned child failure")
        activity.set_quota_wait(child, None)
        return {"task_index": 0, "status": "completed", "summary": "original"}

    batch.run_child = run
    if failure:
        with pytest.raises(ValueError, match="owned child failure"):
            dispatch._execute_and_aggregate(batch)
    else:
        assert dispatch._execute_and_aggregate(batch)["results"][0]["summary"] == "original"
    assert parent._quota_wait_delegation_children is None
    assert parent._quota_wait_status is None
    assert parent._last_activity_ts == before
    clock.now += 4
    assert idle(watchdog(parent)) == 4


def test_detached_stalling_is_not_revived_by_late_quota_marker(clock, monkeypatch):
    held = owner("held")
    record = {"delegation_id": "owned", "status": "stalling", "_interrupted_at": 100.0,
              "progress_fn": lambda: dispatch._batch_progress_token([held])}
    monkeypatch.setattr(detached, "_records", {"owned": record})
    activity.set_quota_wait(held, Decision("hold", "late_hold"))
    clock.now += detached._STALL_GRACE_SECONDS + 1
    with detached._records_lock:
        stalled, expired, _ = detached._sweep_stale_locked(clock.now)
    assert stalled == [] and expired == ["owned"]
    assert record["status"] == "stalling"


def test_legacy_progress_callback_and_unreadable_wait_keep_watchdog_authority(clock, monkeypatch):
    def broken():
        raise ValueError("unreadable")
    records = {name: {"delegation_id": name, "status": "running", "_started": True,
                     "progress_fn": callback, "_progress_token": "frozen", "_progress_ts": 100.0}
               for name, callback in [("legacy", lambda: ("frozen", False)), ("broken", broken)]}
    monkeypatch.setattr(detached, "_records", records)
    clock.now += 10
    with detached._records_lock:
        stalled, expired, _ = detached._sweep_stale_locked(clock.now)
    assert {row[0] for row in stalled} == {"legacy", "broken"}
    assert expired == []
