"""Source-only candidate: real auxiliary dispatch, fake usage and counted sends.

Not execution permission. Run only in the parent's released contained lane.
"""
import asyncio
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import quota_pacing_aux as pacing
from agent import quota_pacing_native as native
from agent.quota_pacing import Decision, PendingAttempt, Route


@pytest.fixture
def lane(monkeypatch):
    sends, binds = [], []
    state = SimpleNamespace(credential="owned-A", provider="openai-codex", stopped=False,
                            decision=Decision("allow", "fresh"), on_wait=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))])

    def create(**request):
        sends.append(request)
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                             base_url="https://chatgpt.com/backend-api/codex",
                             _hermes_aux_effective_provider=state.provider)
    monkeypatch.setattr(pacing, "_section", lambda: {"enabled": True})
    monkeypatch.setattr(pacing, "_snapshot", lambda c, request, provider: pacing._Snapshot(
        provider, request["model"], state.credential, c.base_url))
    monkeypatch.setattr(aux, "bypass_chat_sdk_request_transform", lambda request, c: request)
    token = aux.set_runtime_main("openai-codex", "owned-model", session_id="owned-session")

    def bind(section, **selected):
        binds.append(selected)
        route = Route(selected["provider"], selected["model"], "stable-owned-account")

        def policy(route, hold, budget):
            return state.decision

        async def async_policy(route, hold, budget):
            return policy(route, hold, budget)

        return SimpleNamespace(route=route, policy=policy, async_policy=async_policy)

    monkeypatch.setattr(native, "bind_request", bind)

    # Exercise the actual gate loop without real sleeping/native usage reads.
    original_wait = PendingAttempt.wait
    clock = [0.0]

    def advance(seconds):
        clock[0] += seconds
        if state.on_wait:
            state.on_wait()

    monkeypatch.setattr(PendingAttempt, "wait", lambda self, policy: original_wait(
        self, policy, clock=lambda: clock[0], wait=advance))
    try:
        yield SimpleNamespace(client=client, state=state, sends=sends, binds=binds, response=response)
    finally:
        aux.reset_runtime_main(token)


def test_held_dispatch_keeps_original_messages_and_result(lane):
    original = [{"role": "tool", "tool_call_id": "done", "content": "effect already completed"}]
    lane.state.decision = Decision("hold", "weekly_limit")

    def release():
        assert lane.sends == []
        lane.state.decision = Decision("allow", "fresh")

    lane.state.on_wait = release
    result = aux._create_with_progress_once(lane.client, {"model": "owned-model", "messages": original})
    assert result is lane.response
    assert len(lane.sends) == 1
    assert lane.sends[0]["messages"] is original
    assert lane.binds[-1]["session_id"] == "owned-session"


def test_stream_rejection_rebinds_before_plain_retry(lane):
    seen = []

    def create(**request):
        seen.append(request)
        if request.get("stream"):
            lane.state.credential = "owned-B"
            raise ValueError("owned stream negotiation rejection")
        return lane.response

    lane.client.chat.completions.create = create
    request = {"model": "owned-model", "messages": [{"role": "user", "content": "original"}]}
    with aux.aux_progress_hook(lambda: None):
        assert aux._create_with_progress_once(lane.client, request) is lane.response
    assert len(seen) == 2
    assert {item["credential"] for item in lane.binds} == {"owned-A", "owned-B"}
    assert all(item["messages"] is request["messages"] for item in seen)


def test_cancellation_during_hold_cannot_enter_recovery_or_send(lane):
    lane.state.decision = Decision("hold", "weekly_limit")
    lane.state.on_wait = lambda: setattr(lane.state, "stopped", True)
    with aux.aux_interrupt_protection(cancel_check=lambda: lane.state.stopped):
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            aux._create_with_progress(lane.client, {"model": "owned-model", "messages": []})
    lane.state.stopped = False
    assert lane.sends == []


def test_route_change_during_hold_never_sends_old_snapshot(lane):
    lane.state.decision = Decision("hold", "weekly_limit")
    lane.state.on_wait = lambda: setattr(lane.state, "credential", "unassociated-B")
    with pytest.raises(InterruptedError):
        aux._create_with_progress_once(lane.client, {"model": "owned-model", "messages": []})
    assert lane.sends == []


def test_fallback_destination_takes_fresh_account_model_admission(lane, monkeypatch):
    request = {"model": "primary", "messages": []}
    aux._create_with_progress_once(lane.client, request)
    lane.state.credential = "fallback-key"
    lane.state.decision = Decision("hold", "native_hold")
    lane.state.on_wait = lambda: setattr(lane.state, "stopped", True)
    destination = aux._FallbackDestination("openai-codex", lane.client.base_url, "codex_responses", "fallback")
    monkeypatch.setattr(aux, "_plan_fallback_candidate", lambda *args, **kwargs: (
        destination, {**request, "model": "fallback"}, lambda **kwargs: None))
    monkeypatch.setattr(aux, "_relay_auxiliary_metadata", lambda **kwargs: None)
    with aux.aux_interrupt_protection(cancel_check=lambda: lane.state.stopped):
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            aux._call_fallback_candidate_sync(
                lane.client, "fallback", "owned", task="compression", messages=request["messages"],
                temperature=None, max_tokens=None, tools=None, effective_timeout=30,
                effective_extra_body={}, reasoning_config=None)
    assert len(lane.sends) == 1
    assert lane.binds[-1]["model"] == "fallback"
    assert lane.binds[-1]["credential"] == "fallback-key"


def test_disabled_retains_stock_dispatch_without_snapshot(lane, monkeypatch):
    monkeypatch.setattr(pacing, "_section", lambda: {"enabled": False})
    monkeypatch.setattr(pacing, "_snapshot", lambda *args: pytest.fail("disabled snapshot"))
    assert aux._create_with_progress_once(lane.client, {"model": "stock", "messages": []}) is lane.response
    assert lane.binds == []


def test_opaque_destination_proceeds_unmetered_without_warning(lane, monkeypatch, caplog):
    import logging
    monkeypatch.setattr(pacing, "_snapshot", lambda c, request, provider: pacing._Snapshot(
        provider, request["model"], "", "", "transport_snapshot_required"))
    lane.state.on_wait = lambda: pytest.fail("a meter that cannot check must not wait")
    with aux.aux_interrupt_protection(cancel_check=lambda: False):
        aux._create_with_progress_once(lane.client, {"model": "opaque", "messages": []})
    assert len(lane.sends) == 1
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "repair" not in caplog.text


def test_raw_stream_entry_is_gated_without_relay_context(lane, monkeypatch):
    monkeypatch.setattr(aux, "_RELAY_AUX_CALL_CONTEXT", SimpleNamespace(get=lambda: None))
    assert aux._relay_sync_stream(lane.client, {"model": "raw", "messages": []}) is lane.response
    assert len(lane.sends) == 1
    assert lane.binds[-1]["model"] == "raw"


def test_async_task_cancel_never_sends_or_becomes_fallback(lane, monkeypatch):
    async def run():
        held = asyncio.Event()

        async def policy(route, hold, budget):
            held.set()
            return Decision("hold", "weekly_limit")

        def bind(section, **selected):
            return SimpleNamespace(route=Route(selected["provider"], selected["model"], "owned"),
                                   async_policy=policy)

        monkeypatch.setattr(native, "bind_request", bind)
        task = asyncio.create_task(aux._acreate_with_progress(
            lane.client, {"model": "async", "messages": []}))
        await held.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert lane.sends == []

    asyncio.run(run())


def test_async_hold_releases_original_stream_attempt(lane, monkeypatch):
    lane.state.decision = Decision("hold", "weekly_limit")
    clock = [0.0]
    original_wait = PendingAttempt.wait_async

    async def advance(seconds):
        assert lane.sends == []
        clock[0] += seconds
        lane.state.decision = Decision("allow", "fresh")

    monkeypatch.setattr(PendingAttempt, "wait_async", lambda self, policy: original_wait(
        self, policy, clock=lambda: clock[0], wait=advance))

    async def create(**request):
        lane.sends.append(request)
        return lane.response

    lane.client.chat.completions.create = create
    original = [{"role": "user", "content": "retained input"}]
    result = asyncio.run(aux._acreate_with_progress(
        lane.client, {"model": "async-owned", "messages": original}, force_stream=True))
    assert result is lane.response
    assert len(lane.sends) == 1
    assert lane.sends[0]["messages"] is original
    assert lane.sends[0]["stream"] is True


def test_snapshot_uses_codex_leaf_not_stale_wrapper_credential(monkeypatch):
    # No SDK constructor or HTTP: the real selection function sees exact fake SDK
    # classes, while the existing Codex shim points at its actual selected leaf.
    import openai

    class FakeOpenAI:
        pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(pacing, "_http_snapshot_supported", lambda client: True)
    leaf = FakeOpenAI()
    leaf.api_key = "selected-B"
    leaf.base_url = "https://chatgpt.com/backend-api/codex"
    leaf.default_headers = {"Authorization": "Bearer selected-B"}

    client = aux.CodexAuxiliaryClient(leaf, "gpt-5")
    client.api_key = "stale-A"
    selected = pacing._snapshot(client, {"model": "gpt-5", "messages": []}, "openai-codex")
    assert selected.credential == "selected-B"
    assert selected.problem == ""
    bad = pacing._snapshot(client, {"model": "gpt-5", "extra_headers": {"Authorization": "Bearer other"}}, "openai-codex")
    assert bad.problem == "credential_snapshot_required"


def test_http_snapshot_rejects_auth_hooks_and_custom_transports(monkeypatch):
    import httpx

    class FakeHTTP:
        auth = None
        event_hooks = {}
        _mounts = {}

    class Transport:
        pass

    monkeypatch.setattr(httpx, "Client", FakeHTTP)
    monkeypatch.setattr(httpx, "HTTPTransport", Transport)
    http = FakeHTTP()
    http._transport = Transport()
    leaf = SimpleNamespace(_client=http)
    assert pacing._http_snapshot_supported(leaf)
    http.auth = object()
    assert not pacing._http_snapshot_supported(leaf)
    http.auth = None
    http.event_hooks = {"request": [lambda request: None]}
    assert not pacing._http_snapshot_supported(leaf)
    http.event_hooks = {}
    http._transport = object()
    assert not pacing._http_snapshot_supported(leaf)


def test_selected_workspace_headers_match_native_usage_identity(monkeypatch):
    from agent import codex_headers
    monkeypatch.setattr(codex_headers, "codex_account_headers", lambda token: {
        "ChatGPT-Account-ID": "owned-workspace", "x-openai-internal-codex-residency": "owned-region"})
    leaf = SimpleNamespace(default_headers={
        "Authorization": "Bearer owned-fake", "ChatGPT-Account-ID": "owned-workspace",
        "x-openai-internal-codex-residency": "owned-region"})
    assert pacing._headers_problem(leaf, {}, "owned-fake", "openai-codex") == ""
    assert pacing._headers_problem(leaf, {"extra_headers": {"chatgpt-account-id": "other"}},
                                   "owned-fake", "openai-codex") == "credential_snapshot_required"


@pytest.mark.parametrize("main_runtime", [None, {"provider": "openai-codex", "session_id": "owned-session"}])
def test_native_owner_warning_crosses_protected_context_once_per_status(lane, monkeypatch, caplog, main_runtime):
    import threading
    from agent.status_output import StatusOutputMixin
    from agent.turn_context import _publish_runtime_main
    from agent import prompt_cache_scope

    class Owner(StatusOutputMixin):
        provider = "openai-codex"
        model = "owned-model"
        session_id = "owned-session"
        suppress_status_output = True

    notices, contexts, threads = [], [], []
    owner = Owner()

    def status(kind, message):
        notices.append((kind, message))
        contexts.append(aux._runtime_main_value("session_id"))
        threads.append(threading.current_thread().name)
        # A reentrant read of native context/mirrors must not deadlock.
        with aux._RUNTIME_MAIN_COMPAT_LOCK:
            assert aux._runtime_main_value("quota_warning_callback").__self__ is owner

    owner.status_callback = status
    monkeypatch.setattr(prompt_cache_scope, "resolve_prompt_cache_scope_safe", lambda agent: "owned-scope")
    _publish_runtime_main(owner)
    first = "Pacing for OpenAI Codex is +9.0 hours over the pace. Pausing for about 30 minutes, then continuing automatically."
    second = "Pacing for OpenAI Codex: weekly usage is 99%, over the limit. Pausing for about 60 minutes until it resets, then continuing automatically."
    lane.state.decision = Decision("hold", "weekly_limit", note=first)
    ticks = [0]

    def change():
        assert lane.sends == []
        ticks[0] += 1
        if ticks[0] == 12:
            lane.state.decision = Decision("hold", "native_hold", note=second)
        if ticks[0] == 24:
            lane.state.decision = Decision("allow", "fresh")
        assert ticks[0] <= 24

    lane.state.on_wait = change
    with aux.scoped_runtime_main(main_runtime) as routing, aux.aux_interrupt_protection(cancel_check=lambda: False):
        assert "quota_warning_callback" not in routing
        result = aux._run_protected_sync_provider_call(
            lambda request: aux._create_with_progress_once(lane.client, request),
            {"model": "owned-model", "messages": []})
    assert result is lane.response
    assert len(lane.sends) == 1
    assert len(notices) == 2
    assert all(kind == "warn" for kind, message in notices)
    assert notices[0][1] == first and notices[1][1] == second
    assert contexts == ["owned-session", "owned-session"]
    assert threads == ["hermes-protected-aux-provider"] * 2
    assert caplog.text.count("Pacing for OpenAI Codex") == 2
    assert "repair" not in caplog.text
    assert "owned-A" not in caplog.text + repr(notices)
    assert "quota_warning_callback" not in aux._normalize_main_runtime(None)
    relay_token = aux._RELAY_AUX_CALL_CONTEXT.set({
        "provider": "openai-codex", "model": "owned-model", "request_id": "owned-request", "task": "compression"})
    try:
        route = aux._relay_auxiliary_metadata()
        assert route is not None
        metadata = route[2]
        assert "quota_warning_callback" not in metadata and "api_key" not in metadata
        assert all(not callable(value) for value in metadata.values())
    finally:
        aux._RELAY_AUX_CALL_CONTEXT.reset(relay_token)
    # A later physical attempt owns its own status dedup, not a process-global latch.
    lane.state.decision = Decision("hold", "weekly_limit", note=first)
    lane.state.on_wait = lambda: setattr(lane.state, "decision", Decision("allow", "fresh"))
    aux._create_with_progress_once(lane.client, {"model": "owned-model", "messages": []})
    assert len(notices) == 3


def test_warning_failure_is_presentation_only_and_no_owner_keeps_local_log(lane, caplog):
    notices = []

    def broken(message):
        notices.append(message)
        raise ValueError("presentation unavailable")

    token = aux.set_runtime_main("openai-codex", "owned-model", quota_warning_callback=broken)
    note = "Pacing for OpenAI Codex is +8.4 hours over the pace. Pausing for about 25 minutes, then continuing automatically."
    try:
        lane.state.decision = Decision("hold", "native_hold", note=note)
        lane.state.on_wait = lambda: setattr(lane.state, "decision", Decision("allow", "fresh"))
        aux._create_with_progress_once(lane.client, {"model": "owned-model", "messages": []})
    finally:
        aux.reset_runtime_main(token)
    lane.state.decision = Decision("hold", "native_hold", note=note)
    # A routing dictionary cannot invent a native owner/recipient.
    with aux.scoped_runtime_main({"provider": "openai-codex", "quota_warning_callback": broken}):
        aux._create_with_progress_once(lane.client, {"model": "owned-model", "messages": []})
    assert len(lane.sends) == 2
    assert notices == [note]
    assert caplog.text.count("Pacing for OpenAI Codex") == 2
