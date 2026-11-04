"""Source-only AUX-F1/F2 proofs; execute only in the parent's released lane.

Real native interrupt signalling, auxiliary entry/dispatch/gate and tool-result
consumer; inert request planning, quota policy and counted provider leaf. No SDK,
auth, native usage or network. These cases have NOT been executed by the author.
"""
import asyncio
import contextvars
import json
import threading
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import conversation_compression as compression
from agent import quota_pacing_aux as pacing
from agent import quota_pacing_native as native
from agent.interrupt_control import InterruptControlMixin
from agent.micro_compaction import MicroCompactionMixin
from agent.quota_pacing import Decision, PendingAttempt, Route
from agent.tool_dispatch_helpers import make_tool_result_message
from tools import interrupt
from tools.registry import ToolRegistry


@pytest.fixture
def lane(monkeypatch):
    state = SimpleNamespace(
        decision=Decision("hold", "weekly_limit"), credential="owned-A",
        sends=[], plans=[], binds=[], steps=0, on_wait=None, on_exit=None,
        threads=[], attempts=[], recovery=[], clock=0.0,
    )
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="owned summary"), finish_reason="stop")])

    def create(**request):
        state.sends.append(request)
        state.threads.append(threading.current_thread().name)
        return response

    async def acreate(**request):
        return create(**request)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        base_url="https://owned.invalid/v1", api_key="owned-A",
        _hermes_aux_effective_provider="openai",
    )
    async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=acreate)),
        base_url=client.base_url, api_key="owned-A", _hermes_aux_effective_provider="openai",
    )
    monkeypatch.setattr(pacing, "_section", lambda: {"enabled": True})
    monkeypatch.setattr(pacing, "_snapshot", lambda c, request, provider: pacing._Snapshot(
        provider, request["model"], state.credential, c.base_url))
    monkeypatch.setattr(aux, "bypass_chat_sdk_request_transform", lambda request, c: request)
    monkeypatch.setattr(aux, "_relay_auxiliary_metadata", lambda **kwargs: None)
    monkeypatch.setattr(aux, "_acquire_sync_aux_semaphore", lambda task: None)
    monkeypatch.setattr(aux, "_acquire_async_aux_semaphore", lambda task: None)
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *a, **kw: None)

    def forbidden(*args, **kwargs):
        pytest.fail("stop must not refresh credentials, retry a provider or select fallback")

    for name in (
        "_refresh_provider_credentials", "_recover_provider_pool", "_retry_same_provider_sync",
        "_retry_same_provider_async", "_try_configured_fallback_chain", "_try_main_fallback_chain",
        "_try_payment_fallback", "_try_main_agent_model_fallback", "_evict_cached_client_instance",
    ):
        monkeypatch.setattr(aux, name, forbidden)

    def plan(task, *, messages, async_mode, **kwargs):
        state.plans.append(messages)
        request = {"model": "owned-model", "messages": messages, "timeout": 30}
        req = SimpleNamespace(
            client=async_client if async_mode else client, kwargs=request,
            request_provider="openai", resolved_provider="openai", final_model="owned-model",
            resolved_model="owned-model", base_info=client.base_url, resolved_base_url=client.base_url,
            resolved_api_key="owned-A", resolved_api_mode="chat_completions",
        )
        return req, {"max_tokens": None, "main_runtime": None}, {}

    monkeypatch.setattr(aux, "_plan_aux_call", plan)
    original_ladder = aux._start_recovery_ladder

    def ladder(error, *args, **kwargs):
        state.recovery.append(error)
        return original_ladder(error, *args, **kwargs)

    monkeypatch.setattr(aux, "_start_recovery_ladder", ladder)

    def bind(section, **selected):
        state.binds.append(selected)
        route = Route(selected["provider"], selected["model"], "owned-account")

        def policy(route, hold, budget):
            return state.decision

        async def apolicy(route, hold, budget):
            return policy(route, hold, budget)

        return SimpleNamespace(route=route, policy=policy, async_policy=apolicy)

    monkeypatch.setattr(native, "bind_request", bind)
    original_wait, original_async_wait = PendingAttempt.wait, PendingAttempt.wait_async

    def advance(seconds):
        state.clock += seconds
        state.steps += 1
        assert state.steps <= 4, "bounded proof: missing cancellation must not hang the lane"
        if state.on_wait:
            state.on_wait()
        # Let the broken baseline finish, rather than leaving a forever-held worker.
        if state.steps == 2:
            state.decision = Decision("allow", "fresh")

    def wait(attempt, policy):
        state.attempts.append(attempt)
        try:
            return original_wait(attempt, policy, clock=lambda: state.clock, wait=advance)
        finally:
            if state.on_exit:
                state.on_exit()

    async def await_tick(seconds):
        advance(seconds)
        await asyncio.sleep(0)

    async def await_gate(attempt, policy):
        state.attempts.append(attempt)
        try:
            return await original_async_wait(attempt, policy, clock=lambda: state.clock, wait=await_tick)
        finally:
            if state.on_exit:
                state.on_exit()

    monkeypatch.setattr(PendingAttempt, "wait", wait)
    monkeypatch.setattr(PendingAttempt, "wait_async", await_gate)
    owner = InterruptControlMixin()
    owner._execution_thread_id = threading.get_ident()
    owner._active_children_lock = threading.Lock()
    owner._active_children = []
    owner._hard_interrupt_requested = threading.Event()
    owner.quiet_mode = True
    state.owner = owner
    try:
        yield SimpleNamespace(state=state, client=client, async_client=async_client, response=response)
    finally:
        interrupt.set_interrupt(False, owner._execution_thread_id)
        interrupt.set_interrupt(False)


def _signal_from_host(owner, *, hard=True):
    errors = []

    def stop():
        try:
            owner.interrupt(hard_cancel=hard)
        except BaseException as error:
            errors.append(error)

    host = threading.Thread(target=stop, name="owned-stop-producer", daemon=True)
    host.start()
    host.join(2)
    assert not host.is_alive() and errors == []
    assert interrupt.is_thread_interrupted(owner._execution_thread_id)


@pytest.mark.parametrize("decision", [Decision("hold", "weekly_limit"), Decision("hold", "native_hold")])
def test_native_stop_settles_tool_result_without_losing_completed_receipt(lane, decision):
    state = lane.state
    state.decision = decision
    state.on_wait = lambda: _signal_from_host(state.owner)
    effects = []
    registry = ToolRegistry()

    def completed(args):
        effects.append("original effect")
        return json.dumps({"receipt": "done-once"})

    registry.register("owned_completed", "owned", {"parameters": {"type": "object"}}, completed)
    receipt = make_tool_result_message(
        "owned_completed", registry.dispatch("owned_completed", {}), "completed-call")
    messages = [receipt]

    def auxiliary(args):
        aux.call_llm(task="title_generation", messages=messages)
        return json.dumps({"unexpected": "sent after stop"})

    registry.register("owned_auxiliary", "owned", {"parameters": {"type": "object"}}, auxiliary)
    result = registry.dispatch("owned_auxiliary", {})
    messages.append(make_tool_result_message("owned_auxiliary", result, "pending-call"))
    assert "InterruptedError" in json.loads(result)["error"]
    assert state.owner._interrupt_requested and state.owner._turn_user_intervened
    assert state.sends == [] and effects == ["original effect"]
    assert messages[0] is receipt and json.loads(receipt["content"])["receipt"] == "done-once"
    assert [m["tool_call_id"] for m in messages] == ["completed-call", "pending-call"]
    assert state.plans == [messages]


def test_fresh_call_after_stop_can_resume_original_receipt_without_replaying_effect(lane):
    state = lane.state
    messages = [{"role": "tool", "tool_call_id": "completed", "content": "effect receipt"}]
    state.on_wait = lambda: _signal_from_host(state.owner)
    with pytest.raises(InterruptedError):
        aux.call_llm(task="title_generation", messages=messages)
    assert state.sends == []
    interrupt.set_interrupt(False, state.owner._execution_thread_id)
    state.decision = Decision("allow", "fresh")
    state.on_wait = None
    assert aux.call_llm(task="title_generation", messages=messages) is lane.response
    assert len(state.sends) == 1 and state.sends[0]["messages"] is messages
    assert messages == [{"role": "tool", "tool_call_id": "completed", "content": "effect receipt"}]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", ["route_changed", "route_unavailable"])
def test_unprotected_route_stop_is_normal_exception_and_not_recovered(lane, monkeypatch, failure, async_mode):
    state = lane.state
    if failure == "route_changed":
        state.on_wait = lambda: setattr(state, "credential", "owned-B")
    else:
        def unavailable(*args, **kwargs):
            raise ValueError("owned invalid route")
        monkeypatch.setattr(pacing, "_snapshot", unavailable)
        # The meter cannot set up: that is not a pace hold, so the call proceeds unmetered.
        if async_mode:
            asyncio.run(aux.async_call_llm(task="title_generation", messages=[]))
        else:
            aux.call_llm(task="title_generation", messages=[])
        assert len(state.sends) == 1
        return
    with pytest.raises(InterruptedError):
        if async_mode:
            asyncio.run(aux.async_call_llm(task="title_generation", messages=[]))
        else:
            aux.call_llm(task="title_generation", messages=[])
    assert state.sends == []
    assert all(isinstance(error, InterruptedError) for error in state.recovery)


@pytest.mark.parametrize("disable_pacing", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
def test_stop_latch_blocks_negotiation_retry_after_native_bit_clears(lane, monkeypatch, async_mode, disable_pacing):
    state = lane.state
    state.on_wait = lambda: _signal_from_host(state.owner)
    # Runs after the real PendingAttempt observed cancellation and unwound, before
    # stream negotiation catches Exception and tries its plain-request gate.
    def clear_native_bit():
        interrupt.set_interrupt(False, state.owner._execution_thread_id)
        if disable_pacing:
            monkeypatch.setattr(pacing, "_section", lambda: {"enabled": False})

    state.on_exit = clear_native_bit
    with aux.aux_progress_hook(lambda: None):
        with pytest.raises(InterruptedError):
            if async_mode:
                asyncio.run(aux.async_call_llm(task="title_generation", messages=[]))
            else:
                aux.call_llm(task="title_generation", messages=[])
    assert not interrupt.is_thread_interrupted(state.owner._execution_thread_id)
    assert state.sends == []
    assert all(isinstance(error, InterruptedError) for error in state.recovery)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", ["route_changed", "route_unavailable"])
def test_route_stop_latches_across_stream_to_plain_gate(lane, monkeypatch, failure, async_mode):
    state = lane.state
    if failure == "route_changed":
        state.on_wait = lambda: setattr(state, "credential", "owned-B")
    else:
        snapshot = pacing._snapshot
        calls = [0]

        def unavailable_once(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                raise ValueError("owned unavailable snapshot")
            return snapshot(*args, **kwargs)

        monkeypatch.setattr(pacing, "_snapshot", unavailable_once)
        with aux.aux_progress_hook(lambda: None):
            if async_mode:
                asyncio.run(aux.async_call_llm(task="title_generation", messages=[]))
            else:
                aux.call_llm(task="title_generation", messages=[])
        assert len(state.sends) == 1  # meter setup fault: proceed, never latch a stop
        return
    with aux.aux_progress_hook(lambda: None):
        with pytest.raises(InterruptedError):
            if async_mode:
                asyncio.run(aux.async_call_llm(task="title_generation", messages=[]))
            else:
                aux.call_llm(task="title_generation", messages=[])
    assert state.sends == []


def test_captured_native_source_latches_original_thread_and_acting_owner(lane):
    owner_tid = lane.state.owner._execution_thread_id
    token = interrupt.acting_for_tid.set(owner_tid)
    try:
        check = pacing._current_inputs(lane.client)["cancel_check"]
    finally:
        interrupt.acting_for_tid.reset(token)
    assert callable(check)
    _signal_from_host(lane.state.owner)
    seen = []

    def observe():
        seen.append(check())
        interrupt.set_interrupt(False, owner_tid)
        seen.append(check())

    worker = threading.Thread(target=observe, name="owned-cancellation-observer", daemon=True)
    worker.start()
    worker.join(2)
    assert not worker.is_alive() and seen == [True, True]


def test_sync_dispatch_binds_native_owner_before_callback_thread_hop(lane):
    state = lane.state
    state.on_wait = lambda: _signal_from_host(state.owner)
    outcomes = []

    def callback(request):
        # This callback stands in for a leaf's internal thread hop. The actual
        # protected daemon is covered separately through compression dispatch.
        context = contextvars.copy_context()

        def dispatch():
            try:
                aux._create_with_progress_once(lane.client, request)
            except BaseException as error:
                outcomes.append(error)

        worker = threading.Thread(target=context.run, args=(dispatch,), name="owned-leaf-hop", daemon=True)
        worker.start()
        worker.join(3)
        assert not worker.is_alive()

    with aux._relay_aux_call_scope((), {"task": "title_generation"}):
        aux._run_protected_sync_provider_call(callback, {"model": "owned-model", "messages": []})
    assert len(outcomes) == 1 and isinstance(outcomes[0], InterruptedError)
    assert state.sends == []


def test_external_async_loop_observes_owning_tool_thread_stop(lane):
    state = lane.state
    owner_tid = state.owner._execution_thread_id
    token = interrupt.acting_for_tid.set(owner_tid)
    try:
        context = contextvars.copy_context()
    finally:
        interrupt.acting_for_tid.reset(token)
    outcomes = []
    state.on_wait = lambda: _signal_from_host(state.owner)

    def execute():
        assert threading.get_ident() != owner_tid
        try:
            asyncio.run(aux.async_call_llm(task="vision", messages=[]))
        except BaseException as error:
            outcomes.append(error)

    worker = threading.Thread(target=context.run, args=(execute,), name="owned-external-async", daemon=True)
    worker.start()
    worker.join(3)
    assert not worker.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], InterruptedError)
    assert state.sends == []


def test_micro_compaction_without_explicit_source_stops_and_keeps_summary(lane):
    state = lane.state
    state.on_wait = lambda: _signal_from_host(state.owner)
    compressor = MicroCompactionMixin()
    compressor._micro_compact_rolling_summary = "original committed summary"
    compressor.max_summary_tokens = 100
    compressor.model = "owned-model"
    compressor.provider = "openai"
    compressor.base_url = lane.client.base_url
    compressor.api_key = "owned-A"
    compressor._apply_summary_route = lambda kwargs: None
    assert compressor._micro_summarize_one("completed tool receipt") is None
    assert compressor._micro_compact_rolling_summary == "original committed summary"
    assert state.sends == []


@pytest.mark.parametrize("hard", [False, True])
def test_real_compression_dispatch_keeps_soft_protection_and_hard_commit_fence(lane, hard):
    state = lane.state
    fence = compression.CompressionCommitFence(30)
    state.owner._active_compression_commit_fence = fence
    agent = SimpleNamespace(context_compressor=SimpleNamespace(), session_id="owned-session")
    messages = [{"role": "tool", "tool_call_id": "done", "content": "completed once"}]
    worker_done = threading.Event()
    worker_threads = []

    def on_wait():
        _signal_from_host(state.owner, hard=hard)
        if not hard:
            state.decision = Decision("allow", "fresh")

    state.on_wait = on_wait

    def summarize(history):
        def send(request):
            worker_threads.append(threading.current_thread().name)
            try:
                return aux._create_with_progress_once(lane.client, request, "compression")
            finally:
                worker_done.set()

        return aux._run_protected_sync_provider_call(send, {"model": "owned-model", "messages": history})

    def dispatch():
        return compression._run_summary_dispatch(
            agent, messages, summarize, {}, commit_fence=fence,
            attempt_generation=0, hard_cancel_event=state.owner._hard_interrupt_requested)

    if hard:
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            dispatch()
        state.owner._hard_interrupt_requested.clear()
        interrupt.set_interrupt(False, state.owner._execution_thread_id)
        assert not fence.begin_commit()
        assert state.sends == []
    else:
        assert dispatch() is lane.response
        assert len(state.sends) == 1 and state.sends[0]["messages"] is messages
        assert fence.begin_commit()
        fence.finish_commit()
    assert worker_done.wait(2)
    assert worker_threads == ["hermes-protected-aux-provider"]
    assert messages == [{"role": "tool", "tool_call_id": "done", "content": "completed once"}]
