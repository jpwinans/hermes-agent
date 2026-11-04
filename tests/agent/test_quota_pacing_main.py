"""Prepared native owner-thread pacing contracts; execute only in released containment.

No real providers, credentials, usage children or SDK sends. Real main/summary,
interrupt, core gate and inactivity seams are exercised with fake local bindings.
"""
from collections import deque
from types import SimpleNamespace
from typing import Any
import threading

import pytest

from agent import chat_completion_helpers as summary
from agent import quota_pacing_host as host
from agent import quota_pacing_native as native
from agent import session_activity as activity
from agent import turn_api_call as main
from agent.quota_pacing import Decision, PendingAttempt, Route
from agent.turn_liveness import TurnLivenessWatchdog
from run_agent import AIAgent


@pytest.fixture
def owner():
    agent: Any = AIAgent.__new__(AIAgent)
    agent.provider = "openai-codex"
    agent.model = "display-model-not-selected"
    agent.api_mode = "chat_completions"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent._client_kwargs = {"api_key": "fake-selected-bearer", "base_url": agent.base_url}
    # Inert SDK instance: no constructor, HTTP client, auth discovery or send.
    from openai import OpenAI
    import httpx
    agent.client = object.__new__(OpenAI)
    agent.client.api_key = "fake-selected-bearer"
    agent.client._api_key_provider = None
    agent.client._base_url = httpx.URL(agent.base_url)
    agent.session_id = "owned-fake-session"
    agent.platform = "cli"
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
    agent._last_activity_desc = "completed tool result"
    agent._turn_liveness_activity_generation = 0
    agent._turn_liveness_abort_claim = None
    agent._current_tool = None
    agent._api_call_count = 1
    agent.max_iterations = 4
    agent.iteration_budget = SimpleNamespace(used=1, max_total=4)
    return agent


@pytest.fixture
def pacing(monkeypatch):
    """Keep the real PendingAttempt, advancing its explicit test clock instead of sleeping."""
    state = SimpleNamespace(now=0.0, advance=None, calls=[], decisions=deque(), policy=None)
    original_wait = PendingAttempt.wait

    def advance(seconds):
        state.now += seconds
        if state.advance is not None:
            state.advance()

    def wait(attempt, policy):
        return original_wait(attempt, policy, clock=lambda: state.now, wait=advance)

    def bind(section, **kwargs):
        state.calls.append(kwargs)
        route = Route(kwargs["provider"], kwargs["model"], "owned-account")

        def policy(actual, hold, budget):
            assert actual == route
            if state.policy is not None:
                return state.policy(actual, hold, budget)
            assert state.decisions, "fixture must explicitly release or cancel every held request"
            return state.decisions.popleft()

        return SimpleNamespace(route=route, policy=policy)

    monkeypatch.setattr(PendingAttempt, "wait", wait)
    monkeypatch.setattr(native, "bind_request", bind)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"quota_pacing": {"enabled": True}})
    return state


def request():
    return {"model": "selected-model", "messages": [
        {"role": "assistant", "tool_calls": [{"id": "done", "type": "function", "function": {"name": "effect", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "done", "content": "already committed effect"},
    ]}


def call_main(agent, kwargs):
    return main.perform_api_call(
        agent, api_kwargs=kwargs, _original_api_kwargs=kwargs, _llm_middleware_trace=[],
        _moa_prepared_request=None, _retry=SimpleNamespace(), thinking_spinner=None,
        retry_count=0, api_call_count=1, api_request_id="owned-attempt", effective_task_id="owned",
        turn_id="owned-turn", interrupted=False,
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_main_waits_in_owner_preserves_post_middleware_results(owner, pacing, monkeypatch, streaming):
    from hermes_cli import middleware
    from agent import relay_llm
    original, actual = request(), request()
    actual["model"] = "middleware-selected-model"
    effects = ["already committed effect"]
    sends, held = [], []
    thread_id = threading.get_ident()
    monkeypatch.setattr(main, "_should_stream", lambda _: streaming)
    monkeypatch.setattr(middleware, "run_llm_execution_middleware", lambda req, send, **kw: send(actual))
    monkeypatch.setattr(relay_llm, "execute", lambda req, send, **kw: send(req))

    def send(req, **kwargs):
        assert owner._quota_wait_status is None
        assert threading.get_ident() == thread_id
        sends.append(req)
        return "original response"

    owner._interruptible_api_call = send
    owner._interruptible_streaming_api_call = send
    pacing.decisions.extend([Decision("hold", "native_hold"), Decision("allow", "native_allow")])

    def observe_wait():
        assert sends == []
        assert effects == ["already committed effect"]
        assert actual["messages"][1]["content"] == effects[0]
        assert owner._quota_wait_status == ("hold", "native_hold")
        held.append(threading.get_ident())

    pacing.advance = observe_wait
    result = call_main(owner, original)
    assert result.response == "original response"
    assert len(sends) == 1 and sends[0] is actual
    assert held and all(tid == thread_id for tid in held)
    assert all(row["model"] == "middleware-selected-model" for row in pacing.calls)
    assert effects == ["already committed effect"]
    assert owner._last_activity_ts == 100.0
    assert owner._last_activity_desc == "completed tool result"
    assert owner._quota_pacing_lifetimes == set()


@pytest.mark.parametrize("reason", [None, "session turn lease lost"])
def test_native_interrupt_is_irreversible_after_clear(owner, pacing, reason):
    sends = []
    pacing.decisions.append(Decision("hold", "native_hold"))

    def stop_then_clear():
        pacing.advance = None
        owner.interrupt(hard_cancel=True, tool_reason=reason)
        owner.clear_interrupt()
        assert not owner._interrupt_requested

    pacing.advance = stop_then_clear
    with pytest.raises(InterruptedError):
        host.paced_call(owner, request(), sends.append)
    assert sends == []
    assert owner._interrupt_requested
    assert owner._quota_wait_status is None
    assert not owner._quota_pacing_lifetimes


def test_redirect_latches_without_losing_native_correction(owner, pacing):
    sends = []
    owner._model_request_active.set()
    pacing.decisions.append(Decision("hold", "native_hold"))

    def redirect_then_clear():
        pacing.advance = None
        assert owner.redirect("keep this correction")
        assert owner.clear_interrupt(preserve_redirect=True)

    pacing.advance = redirect_then_clear
    with pytest.raises(InterruptedError):
        host.paced_call(owner, request(), sends.append)
    assert sends == []
    assert owner._pending_redirect == "keep this correction"


@pytest.mark.parametrize("field", ["model", "credential", "endpoint"])
def test_route_change_stops_without_send_or_provider_recovery(owner, pacing, field):
    kwargs, sends = request(), []
    pacing.decisions.append(Decision("hold", "native_hold"))

    def change():
        pacing.advance = None
        if field == "model":
            kwargs["model"] = "different-selected-model"
        else:
            owner._client_kwargs["api_key" if field == "credential" else "base_url"] = "different-selected-value"

    pacing.advance = change
    with pytest.raises(InterruptedError):
        host.paced_call(owner, kwargs, sends.append)
    assert not sends
    assert owner._quota_wait_status is None


def test_managed_summary_gates_before_relay_on_each_retry(owner, pacing, monkeypatch):
    from agent import relay_llm
    kwargs, sends, relay_entries = request(), [], []
    thread_id = threading.get_ident()

    def relay(req, callback, **metadata):
        assert owner._quota_wait_status is None
        relay_entries.append(threading.get_ident())
        return callback(req)

    monkeypatch.setattr(relay_llm, "execute_current", relay)
    for retry in (0, 1):
        pacing.decisions.extend([Decision("hold", "native_hold"), Decision("allow", "native_allow")])
        assert summary._managed_summary_call(owner, "owned-summary", kwargs,
            lambda req, **private: sends.append(req) or "summary", retry_count=retry) == "summary"
    assert len(sends) == 2 and all(req is kwargs for req in sends)
    assert relay_entries == [thread_id, thread_id]


def test_codex_summary_builds_once_per_logical_attempt_not_per_poll(owner, pacing, monkeypatch):
    owner.api_mode = "codex_responses"
    built, sends = [], []

    def build(messages):
        kwargs = {"model": "selected-model", "input": messages, "tools": [], "tool_choice": "auto", "parallel_tool_calls": True}
        built.append(kwargs)
        return kwargs

    owner._build_api_kwargs = build
    owner._interruptible_api_call = lambda req, **private: sends.append(req) or ("" if len(sends) == 1 else "summary")
    monkeypatch.setattr(summary, "_summary_text", lambda agent, response: response)
    attempt = summary._codex_summary_attempt(owner, request()["messages"], "owned-summary")
    for retry in (0, 1):
        pacing.decisions.extend([Decision("hold", "native_hold"), Decision("allow", "native_allow")])
        assert attempt(retry) == ("" if retry == 0 else "summary")
    assert len(built) == len(sends) == 2
    assert all(sent is kwargs for sent, kwargs in zip(sends, built))
    assert all(not any(key in kwargs for key in ("tools", "tool_choice", "parallel_tool_calls")) for kwargs in sends)


@pytest.mark.parametrize("section", [{}, {"enabled": False}])
def test_disabled_stock_call_does_not_bind_or_mark_wait(owner, monkeypatch, section):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"quota_pacing": section})
    monkeypatch.setattr(native, "bind_request", lambda *a, **k: pytest.fail("stock mode must not bind"))
    kwargs, sends = request(), []
    assert host.paced_call(owner, kwargs, lambda req: sends.append(req) or "stock") == "stock"
    assert sends[0] is kwargs
    assert not hasattr(owner, "_quota_wait_clock")
    assert not hasattr(owner, "_quota_pacing_lifetimes")


def test_enabled_invalid_native_binding_proceeds_unmetered(owner, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"quota_pacing": {"enabled": True, "accounts": "invalid"}})
    sends, observed = [], []
    original_wait = PendingAttempt.wait
    now = [0.0]

    def advance(seconds):
        observed.append(owner._quota_wait_status)
        owner.interrupt()
        now[0] += seconds

    monkeypatch.setattr(PendingAttempt, "wait", lambda attempt, policy: original_wait(
        attempt, policy, clock=lambda: now[0], wait=advance))
    host.paced_call(owner, request(), lambda req, **private: sends.append(req))
    assert len(sends) == 1
    assert observed == []  # no wait, no status, no message


def test_claude_selected_native_key_is_admitted_without_refresh(owner, pacing, monkeypatch):
    owner.api_mode = "anthropic_messages"
    owner.provider = "anthropic"
    owner._anthropic_api_key = "fake-selected-bearer"
    owner._anthropic_base_url = "https://api.anthropic.com"
    owner._try_refresh_anthropic_client_credentials = lambda: pytest.fail("selection must not refresh")
    monkeypatch.setattr("agent.client_lifecycle.get_provider_request_timeout", lambda *args: 30.0)
    kwargs, sends = request(), []
    key = owner._request_anthropic_client_key()
    pacing.decisions.append(Decision("allow", "native_allow"))

    def send(req, *, _credential_snapshot):
        sends.append((req, _credential_snapshot))
        return "native response"

    assert host.paced_call(owner, kwargs, send) == "native response"
    assert len(sends) == 1 and sends[0][0] is kwargs and sends[0][1] == key
    assert all(row["credential"] == key[1] and row["base_url"] == key[2] for row in pacing.calls)
    assert "fake-selected-bearer" not in repr(host._selected(owner, kwargs))
    assert "_credential_snapshot" not in kwargs


def test_wait_clock_exemption_then_ordinary_idle_without_fake_progress(owner, monkeypatch):
    clock = [110.0]
    monkeypatch.setattr(activity.time, "time", lambda: clock[0])
    watcher = TurnLivenessWatchdog(owner, session_id="owned", timeout_s=30, poll_s=1,
        stop_event=threading.Event(), activity_lock=owner._liveness_activity_lock(),
        is_turn_active=lambda: True, commit_abort=lambda *args: pytest.fail("sample only"), deactivate_turn=lambda: None)
    stale = watcher._sample()
    assert stale is not None
    activity.set_quota_wait(owner, Decision("hold", "native_hold"))
    clock[0] = 1010.0
    sample = watcher._sample()
    assert sample is not None and sample.idle_seconds == 0.0
    assert owner.interrupt(require_generation=stale.generation) is False
    activity.set_quota_wait(owner, None)
    sample = watcher._sample()
    assert sample is not None and sample.idle_seconds == 10.0
    clock[0] = 1050.0
    sample = watcher._sample()
    assert sample is not None and sample.idle_seconds == 50.0
    assert owner._last_activity_ts == 100.0
    assert owner._last_activity_desc == "completed tool result"


def test_gateway_reaper_rechecks_live_wait_and_post_wait_idle(owner, monkeypatch):
    from gateway import run as gateway
    clock = [110.0]
    monkeypatch.setattr(activity.time, "time", lambda: clock[0])
    monkeypatch.setattr(gateway, "_dump_wedged_turn_stacks", lambda *a: pytest.fail("live wait is not a wedge"))
    monkeypatch.setattr(gateway, "_reap_gateway_turn_processes", lambda *a, **k: pytest.fail("must not reap live wait"))
    fired = threading.Event()
    args = dict(agent_holder=[owner], task_id="owned", process_baseline=set(), worker_done=threading.Event(),
        timeout_fired=fired, cleanup_lock=threading.Lock(), inactivity_timeout=30)
    activity.set_quota_wait(owner, Decision("blocked", "configuration_repair"))
    clock[0] = 1010.0
    assert gateway._abandon_timed_out_gateway_turn(**args) is False
    activity.set_quota_wait(owner, None)
    assert gateway._abandon_timed_out_gateway_turn(**args) is False
    assert not fired.is_set()
    assert not owner._interrupt_requested


def test_profile_scoped_loader_uses_owned_configuration(tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    homes = [tmp_path / "disabled-profile", tmp_path / "enabled-profile"]
    for home, enabled in zip(homes, ("false", "true")):
        home.mkdir()
        (home / "config.yaml").write_text("quota_pacing:\n  enabled: " + enabled + "\n", encoding="utf-8")
    for home, expected in zip(homes, (False, True)):
        token = set_hermes_home_override(home)
        try:
            assert host._section()["enabled"] is expected
        finally:
            reset_hermes_home_override(token)


def test_max_iterations_cancellation_drops_only_unanswered_nudge(owner, pacing, monkeypatch):
    from agent import relay_llm
    owner.api_mode = "codex_responses"
    owner.max_iterations = 2
    owner.suppress_status_output = True
    owner._build_api_kwargs = lambda messages: {"model": "selected-model", "input": messages}
    sends = []
    owner._interruptible_api_call = sends.append
    monkeypatch.setattr(summary, "_iteration_summary_api_messages", lambda agent, messages: list(messages))
    outcomes = []
    monkeypatch.setattr(relay_llm, "complete_logical_call", lambda request_id, **kwargs: outcomes.append(kwargs["outcome"]))
    messages = request()["messages"]
    original_rows = list(messages)
    pacing.decisions.append(Decision("hold", "native_hold"))
    pacing.advance = lambda: owner.interrupt()
    with pytest.raises(InterruptedError):
        summary.handle_max_iterations(owner, messages, api_call_count=2)
    assert sends == []
    assert messages == original_rows
    assert all(row is original for row, original in zip(messages, original_rows))
    assert outcomes == ["cancelled"]


def test_parse_failure_cannot_turn_enabled_pacing_into_default_disabled(owner, monkeypatch):
    from hermes_cli.config_read_errors import FailedConfigRead
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: FailedConfigRead({}, error=ValueError("fake malformed YAML")))
    assert host._section()["enabled"] is True
    assert host._section()["invalid"] is True


def test_configuration_fault_proceeds_keeping_native_hold(owner, pacing, monkeypatch):
    config = {"enabled": True, "repair": "pending"}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"quota_pacing": dict(config)})
    sends, holds = [], []

    def policy(route, hold, budget):
        holds.append(hold)
        if config["repair"] == "pending":
            return Decision("blocked", "configuration_repair", active_hold="native-held-bucket")
        assert hold == "native-held-bucket"
        return Decision("allow", "native_allow", active_hold=hold)

    pacing.policy = policy
    pacing.advance = lambda: config.update(repair="fixed")
    kwargs = request()
    host.paced_call(owner, kwargs, lambda req, **private: sends.append(req))
    assert sends == [kwargs] and sends[0] is kwargs
    assert holds == [None]


@pytest.mark.parametrize("kind", ["opaque", "virtual", "callable", "auth_override"])
def test_unbound_transports_proceed_unmetered(owner, pacing, kind):
    kwargs, sends = request(), []
    if kind == "opaque":
        owner.client = object()
    elif kind == "virtual":
        owner.provider = "moa"
    elif kind == "callable":
        owner._client_kwargs["api_key"] = lambda: pytest.fail("must not consume credential provider")
    else:
        kwargs["extra_headers"] = {"Authorization": "fake-other-bearer"}
    pacing.decisions.append(Decision("allow", "native_allow"))
    observed = []

    def cancel():
        observed.append(owner._quota_wait_status)
        owner.interrupt()

    pacing.advance = cancel
    host.paced_call(owner, kwargs, lambda req, **private: sends.append(req))
    assert sends == [kwargs]
    assert observed == []


def test_body_model_override_binds_the_actual_wire_model(owner, pacing):
    kwargs = request()
    kwargs["extra_body"] = {"model": "body-selected-model"}
    pacing.decisions.append(Decision("allow", "native_allow"))
    host.paced_call(owner, kwargs, lambda req, **private: None)
    assert all(row["model"] == "body-selected-model" for row in pacing.calls)


def test_fresh_side_question_owner_does_not_reuse_parent_admission(owner, pacing):
    kwargs = request()
    pacing.decisions.extend([Decision("allow", "native_allow"), Decision("hold", "native_hold"), Decision("allow", "native_allow")])
    parent_sends, fork_sends = [], []
    host.paced_call(owner, kwargs, lambda req, **private: parent_sends.append(req))
    # Detached fork construction is native and creates a new AIAgent; construct an
    # init-less instance here to avoid any provider/auth/plugin initialization.
    fork: Any = AIAgent.__new__(AIAgent)
    fork.__dict__.update({key: value for key, value in vars(owner).items()
        if not key.startswith(("_quota_", "_turn_liveness_"))})
    fork._pending_redirect_lock = threading.Lock()
    fork._client_kwargs = dict(owner._client_kwargs)
    fork._client_kwargs["base_url"] = owner._client_kwargs["base_url"]
    pacing.advance = lambda: (parent_sends == [kwargs] and fork_sends == []) or pytest.fail("no fork send before own admission")
    host.paced_call(fork, kwargs, lambda req, **private: fork_sends.append(req))
    assert fork_sends == [kwargs]
    assert fork._quota_pacing_lifetimes is not owner._quota_pacing_lifetimes


def test_real_progress_during_wait_discards_old_wait_credit(owner, monkeypatch):
    clock = [110.0]
    monkeypatch.setattr(activity.time, "time", lambda: clock[0])
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    owner._persist_session_activity_if_due = lambda: None
    activity.set_quota_wait(owner, Decision("hold", "native_hold"))
    clock[0] = 200.0
    owner._touch_activity("real completed work")
    clock[0] = 300.0
    activity.set_quota_wait(owner, None)
    assert activity.quota_idle_seconds(owner, 100.0) == 0.0
    clock[0] = 310.0
    owner._touch_activity("another real result")
    assert activity.quota_idle_seconds(owner, 10.0) == 10.0
    assert owner._last_activity_desc == "another real result"


@pytest.mark.parametrize("summary_unavailable", [False, True])
def test_thread_gateway_watchdog_uses_quota_clock_not_raw_idle(owner, monkeypatch, summary_unavailable):
    from gateway import run as gateway
    monkeypatch.setattr(activity.time, "time", lambda: 1000.0)
    activity.set_quota_wait(owner, Decision("hold", "native_hold"))
    monkeypatch.setattr(gateway, "_abandon_timed_out_gateway_turn", lambda **kw: pytest.fail("live quota wait must not time out"))

    class PollDone(threading.Event):
        polls = 0

        def wait(self, timeout=None):
            self.polls += 1
            return self.polls > 1

    done = PollDone()
    assert owner.get_activity_summary()["seconds_since_activity"] == 900.0
    if summary_unavailable:
        def unavailable():
            raise RuntimeError("fake unavailable diagnostic")
        owner.get_activity_summary = unavailable
    gateway._watch_gateway_turn_inactivity(agent_holder=[owner], task_id="owned", process_baseline=set(),
        timeout=30, worker_done=done, timeout_fired=threading.Event(), cleanup_lock=threading.Lock())
    assert done.polls == 2
    assert owner._last_activity_ts == 100.0
    activity.set_quota_wait(owner, None)


def test_async_gateway_consumer_waits_without_warning_or_timeout(owner, monkeypatch):
    import asyncio
    from gateway.run_turn import GatewayTurnMixin
    monkeypatch.setattr(activity.time, "time", lambda: 1000.0)
    activity.set_quota_wait(owner, Decision("blocked", "credential_repair"))

    async def exercise():
        future = asyncio.get_running_loop().create_future()
        polls, backups = [], []

        async def poll(tasks, timeout):
            polls.append(timeout)
            if len(polls) == 1:
                return set(), tasks
            future.set_result({"messages": request()["messages"], "completed": True})
            return {future}, set()

        async def backup(*args):
            backups.append(True)

        monkeypatch.setattr(asyncio, "wait", poll)
        runner: Any = GatewayTurnMixin.__new__(GatewayTurnMixin)
        runner._agent_activity_summary = lambda agent: agent.get_activity_summary()
        runner._run_agent_backup_interrupt_check = backup
        runner._run_agent_inactivity_warning = lambda *args: pytest.fail("quota wait is not inactivity")
        worker = SimpleNamespace(executor_task=future, agent_timeout=30, agent_warning=10,
            timeout_fired=threading.Event())
        ctx = SimpleNamespace(agent_holder=[owner], awake_idle=activity.AwakeIdleMeter())
        result = await runner._run_agent_await_turn_worker(worker, ctx, asyncio.Event(), None)
        assert result["completed"] is True
        assert backups == [True]
        assert not worker.timeout_fired.is_set()

    try:
        asyncio.run(exercise())
    finally:
        activity.set_quota_wait(owner, None)


def test_credential_fault_proceeds_once_without_message(owner, pacing):
    kwargs, sends, observed, holds = request(), [], [], []

    def policy(route, active_hold, budget):
        holds.append(active_hold)
        return Decision("blocked", "credential_repair", active_hold="existing-account-hold")

    def stop():
        observed.append(owner._quota_wait_message)
        assert sends == []
        assert kwargs["messages"][1]["content"] == "already committed effect"
        owner.interrupt()

    pacing.policy, pacing.advance = policy, stop
    host.paced_call(owner, kwargs, lambda req, **private: sends.append(req))
    assert observed == []
    assert sends == [kwargs]
    assert holds == [None]
    assert owner._quota_wait_message is None


def test_codex_main_binds_the_once_preflighted_request(owner, pacing, monkeypatch):
    from hermes_cli import middleware
    from agent import relay_llm
    owner.api_mode = "codex_responses"
    built, sends = [], []

    def preflight(kwargs, **options):
        actual = {**kwargs, "model": "preflight-selected-model"}
        built.append(actual)
        return actual

    owner._get_transport = lambda: SimpleNamespace(preflight_kwargs=preflight)
    owner._is_codex_backend = lambda: True
    owner._interruptible_api_call = lambda kwargs, **private: sends.append(kwargs) or "codex-response"
    monkeypatch.setattr(main, "_should_stream", lambda agent: False)
    monkeypatch.setattr(middleware, "run_llm_execution_middleware", lambda kwargs, send, **metadata: send(kwargs))
    monkeypatch.setattr(relay_llm, "execute", lambda kwargs, send, **metadata: send(kwargs))
    pacing.decisions.extend([Decision("hold", "native_hold"), Decision("allow", "native_allow")])
    pacing.advance = lambda: not sends or pytest.fail("no send while held")
    result = call_main(owner, request())
    assert result.response == "codex-response"
    assert len(built) == len(sends) == 1 and sends[0] is built[0]
    assert all(row["model"] == "preflight-selected-model" for row in pacing.calls)


def test_actual_native_lease_loss_cancels_wait_even_if_flag_is_cleared(owner, pacing):
    from agent.turn_facade_lease import DurableTurnLease
    lease: Any = DurableTurnLease.__new__(DurableTurnLease)
    lease._lock = threading.Lock()
    lease.stop = threading.Event()
    lease.turn_active = True
    lease.agent = owner
    sends = []
    pacing.decisions.append(Decision("hold", "native_hold"))

    def lose_lease():
        pacing.advance = None
        lease._interrupt_turn("Session turn lease lost; stopping to protect the transcript.")
        owner.clear_interrupt()

    pacing.advance = lose_lease
    with pytest.raises(InterruptedError):
        host.paced_call(owner, request(), sends.append)
    assert not sends
    assert lease.interrupt_message == "Session turn lease lost; stopping to protect the transcript."
    assert not owner._quota_pacing_lifetimes


@pytest.fixture
def claude_wire(owner, monkeypatch):
    """Native constructors/slot machinery with only SDK build/send made inert."""
    from agent import anthropic_adapter
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    owner.provider = "anthropic"
    owner.api_mode = "anthropic_messages"
    owner.model = "claude-selected-model"
    owner.base_url = owner._anthropic_base_url = "https://api.anthropic.com"
    owner._anthropic_api_key = "fake-admitted-bearer"
    owner._oauth_1m_beta_disabled = True
    state = SimpleNamespace(builds=[], sends=[], refreshes=[], closes=[], entries=[], requests=[])
    monkeypatch.setattr("agent.client_lifecycle.get_provider_request_timeout", lambda *args: 37.0)

    def refresh():
        state.refreshes.append(threading.get_ident())
        owner._anthropic_api_key = "fake-refreshed-bearer"
        owner._anthropic_base_url = "https://refreshed.invalid"

    class InertClient:
        is_closed = False

        def __init__(self, token, endpoint, options):
            self.token, self.endpoint, self.options = token, endpoint, options

        def close(self):
            self.is_closed = True
            state.closes.append(self)

    def build(token, endpoint, **options):
        client = InertClient(token, endpoint, options)
        state.builds.append(client)
        return client

    def send(req, *, client):
        state.sends.append((req, client))
        return "native-response"

    owner._try_refresh_anthropic_client_credentials = refresh
    owner._anthropic_messages_create = send
    monkeypatch.setattr(anthropic_adapter, "build_anthropic_client", build)
    monkeypatch.setattr(owner, "_force_close_tcp_sockets", lambda client: None)
    monkeypatch.setattr(summary, "_resolve_nonstream_watchdogs", lambda *args: SimpleNamespace(
        stale_timeout=float("inf"), codex=False, idle_requires_progress=False,
        ttfb_enabled=False, idle_enabled=False, progress_timeout=0.0))

    def streaming_run(call):
        # Real entry, __init__, wire dispatch, constructor and registry;
        # stream decoding/polling is outside this counted-wire fixture.
        state.requests.append(call)
        try:
            return call._call_wire(1)
        finally:
            call.clients.close_once("stream_request_complete")

    monkeypatch.setattr(summary._StreamingCall, "run", streaming_run)
    monkeypatch.setattr(summary._StreamingCall, "_call_anthropic", lambda call, client: send(call.api_kwargs, client=client))
    return state


@pytest.mark.parametrize("lane", ["nonstream", "streaming", "summary"])
def test_claude_native_entries_send_only_admitted_snapshot_after_global_change(owner, pacing, claude_wire, monkeypatch, lane):
    from agent import relay_llm
    from hermes_cli import middleware
    state, kwargs = claude_wire, request()
    owner_tid = threading.get_ident()
    admitted_key = owner._request_anthropic_client_key()
    relay_metadata = []

    def relay(req, callback, **metadata):
        relay_metadata.append(metadata)
        assert req is kwargs and "_credential_snapshot" not in req
        return callback(req)

    def entry(req, **private):
        state.entries.append(threading.get_ident())
        # Change live credentials AFTER final owner admission; neither this
        # nor the stock worker refresh may alter this request's actual client.
        owner._anthropic_api_key = "fake-unadmitted-bearer"
        owner._anthropic_base_url = "https://unadmitted.invalid"
        owner._oauth_1m_beta_disabled = False
        if lane == "streaming":
            return summary.interruptible_streaming_api_call(owner, req, **private)
        return summary.interruptible_api_call(owner, req, **private)

    owner._interruptible_api_call = entry
    owner._interruptible_streaming_api_call = entry
    monkeypatch.setattr(relay_llm, "execute", relay)
    monkeypatch.setattr(relay_llm, "execute_current", relay)
    monkeypatch.setattr(middleware, "run_llm_execution_middleware", lambda req, send, **metadata: send(req))
    monkeypatch.setattr(main, "_should_stream", lambda agent: lane == "streaming")
    pacing.decisions.extend([Decision("hold", "native_hold"), Decision("allow", "native_allow")])

    def held():
        assert threading.get_ident() == owner_tid
        assert not state.entries and not state.builds and not state.sends
        assert kwargs["messages"][1]["content"] == "already committed effect"

    pacing.advance = held
    if lane == "summary":
        result = summary._managed_summary_call(owner, "snapshot-summary", kwargs, entry, retry_count=0)
    else:
        result = call_main(owner, kwargs).response
    assert result == "native-response"
    assert state.entries == [owner_tid]
    assert len(state.builds) == len(state.sends) == 1
    client = state.builds[0]
    assert state.sends[0][0] is kwargs and state.sends[0][1] is client
    assert (client.token, client.endpoint) == admitted_key[1:3]
    assert client.options == {"timeout": admitted_key[3], "drop_context_1m_beta": admitted_key[4]}
    assert not state.refreshes and not state.closes
    assert all(row["credential"] == client.token and row["base_url"] == client.endpoint for row in pacing.calls)
    assert all("_credential_snapshot" not in metadata for metadata in relay_metadata)
    assert "fake-admitted-bearer" not in repr(relay_metadata)
    assert "_credential_snapshot" not in kwargs and "_credential_snapshot" not in vars(owner)
    if lane == "streaming":
        assert state.requests[0]._credential_snapshot == admitted_key


@pytest.mark.parametrize("streaming", [False, True])
def test_claude_snapshot_absent_keeps_stock_worker_refresh(owner, claude_wire, monkeypatch, streaming):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"quota_pacing": {"enabled": False}})
    kwargs = request()
    entry = summary.interruptible_streaming_api_call if streaming else summary.interruptible_api_call
    assert host.paced_call(owner, kwargs, lambda req: entry(owner, req)) == "native-response"
    assert len(claude_wire.refreshes) == len(claude_wire.sends) == 1
    client = claude_wire.sends[0][1]
    assert client.token == "fake-refreshed-bearer" and client.endpoint == "https://refreshed.invalid"
    assert not hasattr(owner, "_quota_pacing_lifetimes")


@pytest.mark.parametrize("override", ["Authorization", "x-api-key", "Host"])
def test_claude_auth_override_proceeds_unmetered(owner, pacing, claude_wire, override):
    kwargs = request()
    kwargs["extra_headers"] = {override: "fake-unadmitted-value"}
    pacing.decisions.append(Decision("allow", "native_allow"))
    pacing.advance = lambda: pytest.fail("a meter that cannot verify must not wait")
    sends = []
    host.paced_call(owner, kwargs, lambda req, **private: sends.append(req))
    assert sends == [kwargs]
    assert not claude_wire.refreshes


def test_claude_snapshot_constructor_reuses_native_slot_and_does_not_refresh(owner, claude_wire):
    key = owner._request_anthropic_client_key()
    first = owner._create_request_anthropic_client(reason="owned", _credential_snapshot=key)
    owner._close_request_anthropic_client(first, reason="request_complete")
    owner._anthropic_api_key = "fake-other-live-bearer"
    second = owner._create_request_anthropic_client(reason="owned-retry", _credential_snapshot=key)
    assert second is first
    owner._close_request_anthropic_client(second, reason="request_complete")
    assert len(claude_wire.builds) == 1 and not claude_wire.refreshes and not claude_wire.closes


@pytest.mark.parametrize("field", ["credential", "endpoint"])
def test_claude_change_during_hold_stops_instead_of_adopting(owner, pacing, claude_wire, field):
    kwargs = request()
    pacing.decisions.append(Decision("hold", "native_hold"))

    def change():
        setattr(owner, "_anthropic_api_key" if field == "credential" else "_anthropic_base_url", "fake-other-value")

    pacing.advance = change
    with pytest.raises(InterruptedError):
        summary._managed_summary_call(owner, "held-summary", kwargs, owner._interruptible_api_call, retry_count=0)
    assert not claude_wire.builds and not claude_wire.sends and not claude_wire.refreshes
    assert kwargs["messages"][1]["content"] == "already committed effect"


@pytest.mark.parametrize("streaming", [False, True])
def test_claude_interrupt_before_admission_never_constructs_request(owner, pacing, claude_wire, monkeypatch, streaming):
    from hermes_cli import middleware
    monkeypatch.setattr(main, "_should_stream", lambda agent: streaming)
    monkeypatch.setattr(middleware, "run_llm_execution_middleware", lambda req, send, **metadata: send(req))
    owner._interruptible_api_call = lambda *args, **private: pytest.fail("no entry before admission")
    owner._interruptible_streaming_api_call = owner._interruptible_api_call
    pacing.decisions.append(Decision("hold", "native_hold"))
    pacing.advance = lambda: owner.interrupt()
    with pytest.raises(InterruptedError):
        call_main(owner, request())
    assert not claude_wire.builds and not claude_wire.sends and not claude_wire.refreshes


@pytest.mark.parametrize("header", ["Authorization", "x-api-key", "Host"])
def test_claude_custom_provider_auth_headers_cannot_replace_admitted_identity(owner, pacing, claude_wire, monkeypatch, header):
    from agent import anthropic_adapter
    configuration = {"quota_pacing": {"enabled": True}, "custom_providers": [{
        "name": "owned-official-route", "base_url": "https://api.anthropic.com",
        "extra_headers": {header: "fake-unadmitted-value"}}]}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: configuration)
    # Exercise the real route matcher: the review's hypothesized configuration
    # must actually reach the same helper used by the SDK builder.
    assert anthropic_adapter._custom_provider_extra_headers(owner._anthropic_base_url) == {header: "fake-unadmitted-value"}
    pacing.decisions.append(Decision("allow", "native_allow"))
    pacing.advance = lambda: pytest.fail("a meter that cannot verify must not wait")
    sends = []
    host.paced_call(owner, request(), lambda req, **private: sends.append(req))
    assert len(sends) == 1
    assert not claude_wire.refreshes
