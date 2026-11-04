"""Authored source-only: real Anthropic auxiliary wrappers/adapter, inert SDK leaf.

No SDK construction, auth discovery, network or native usage. Execution needs the
parent's exact contained release; these cases have not been run by the author.
"""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import anthropic_adapter as adapter
from agent import quota_pacing_aux as pacing
from agent import quota_pacing_native as native
from agent.quota_pacing import Decision, PendingAttempt, Route


@pytest.fixture
def anthropic_lane(monkeypatch):
    import anthropic
    import httpx
    from hermes_cli import config
    from hermes_cli.config_providers import get_custom_provider_extra_headers

    sends, binds, attempts = [], [], []
    state = SimpleNamespace(decision=Decision("allow", "fresh"), stopped=False,
                            on_wait=None, async_policy=None, entries=[])

    class InertTransport:
        pass

    class InertHTTP:
        auth = None
        headers = {}
        event_hooks = {}
        _mounts = {}

        def __init__(self):
            self._transport = InertTransport()

    class InertSDK:
        def __init__(self, *, api_key=None, auth_token=None,
                     base_url="https://api.anthropic.com/", default_headers=None):
            self.api_key, self.auth_token = api_key, auth_token
            self.base_url = base_url
            self._custom_headers = default_headers or {}
            self._client = InertHTTP()
            self.messages = SimpleNamespace(create=self.create)
            self.credentials = self._token_cache = self._custom_auth = None
            self._middleware = ()
            self._middleware_chain = None

        @property
        def default_headers(self):
            # Same static header/Omit surface as the selected SDK; no auth provider.
            headers = {}
            if self.api_key is not None:
                headers["X-Api-Key"] = self.api_key
            if self.auth_token is not None:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            return {**headers, **self._custom_headers}

        def create(self, **request):
            body = {key: value for key, value in request.items() if key != "extra_body"}
            body.update(request.get("extra_body") or {})
            sends.append(SimpleNamespace(leaf=self, request=request, body=body,
                                         headers=self.default_headers))
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="summary")],
                                   stop_reason="end_turn", model=body["model"],
                                   usage=SimpleNamespace(input_tokens=2, output_tokens=1))

    monkeypatch.setattr(anthropic, "Anthropic", InertSDK)
    monkeypatch.setattr(httpx, "Client", InertHTTP)
    monkeypatch.setattr(httpx, "HTTPTransport", InertTransport)
    # Exercise the real constructor/header merge and config matching helper, but
    # replace its config input with owned literals, never a real profile read.
    monkeypatch.setattr(config, "get_custom_provider_extra_headers", lambda base_url:
                        get_custom_provider_extra_headers(base_url, custom_providers=state.entries))
    monkeypatch.setattr(pacing, "_section", lambda: {"enabled": True})
    monkeypatch.setattr(aux, "bypass_chat_sdk_request_transform", lambda request, client: request)

    def make(*, oauth=True):
        field = "auth_token" if oauth else "api_key"
        leaf = adapter._new_sdk_client(
            SimpleNamespace(Anthropic=InertSDK, Omit=anthropic.Omit),
            {field: "selected-B", "base_url": "https://api.anthropic.com/"},
            {"anthropic-beta": "owned-beta"})
        client = aux.AnthropicAuxiliaryClient(
            leaf, "anthropic/claude-sonnet-4.6", "stale-wrapper-A",
            "https://api.anthropic.com", is_oauth=oauth)
        client._hermes_aux_effective_provider = "anthropic"
        return client

    def bind(section, **selected):
        binds.append(selected)
        route = Route(selected["provider"], selected["model"], "stable-owned-account")

        def policy(route, hold, budget):
            return state.decision

        async def async_policy(route, hold, budget):
            if state.async_policy is not None:
                return await state.async_policy(route, hold, budget)
            return policy(route, hold, budget)

        return SimpleNamespace(route=route, policy=policy, async_policy=async_policy)

    monkeypatch.setattr(native, "bind_request", bind)
    original_attempt = pacing._attempt

    def attempt(*args, **kwargs):
        result = original_attempt(*args, **kwargs)
        attempts.append(result[0])
        return result

    monkeypatch.setattr(pacing, "_attempt", attempt)
    clock = [0.0]
    original_wait, original_async_wait = PendingAttempt.wait, PendingAttempt.wait_async

    def advance(seconds):
        clock[0] += seconds
        if state.on_wait:
            state.on_wait()
        else:
            pytest.fail("unexpected held attempt without owned release/cancel")

    async def advance_async(seconds):
        advance(seconds)

    monkeypatch.setattr(PendingAttempt, "wait", lambda self, policy: original_wait(
        self, policy, clock=lambda: clock[0], wait=advance))
    monkeypatch.setattr(PendingAttempt, "wait_async", lambda self, policy: original_async_wait(
        self, policy, clock=lambda: clock[0], wait=advance_async))
    token = aux.set_runtime_main("anthropic", "claude-sonnet-4-6", session_id="owned-session")
    try:
        yield SimpleNamespace(make=make, state=state, sends=sends, binds=binds, attempts=attempts,
                              sdk=InertSDK, http=InertHTTP, transport=InertTransport)
    finally:
        aux.reset_runtime_main(token)


@pytest.mark.parametrize("oauth", [True, False])
def test_actual_adapter_wait_retains_request_and_uses_selected_leaf(anthropic_lane, oauth):
    lane = anthropic_lane
    client = lane.make(oauth=oauth)
    client.base_url = "https://stale-wrapper.invalid"
    original = [
        {"role": "user", "content": "Retain input and completed effects"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "done", "type": "function",
         "function": {"name": "owned_effect", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "done", "content": "effect already completed"},
    ]
    before = deepcopy(original)
    request = {"model": "anthropic/claude-sonnet-4.6", "messages": original}
    lane.state.decision = Decision("hold", "weekly_limit")

    def release():
        assert lane.sends == []
        assert request["messages"] is original and original == before
        lane.state.decision = Decision("allow", "fresh")

    lane.state.on_wait = release
    response = aux._create_with_progress_once(client, request)
    assert response.choices[0].message.content == "summary"
    assert len(lane.sends) == 1 and original == before
    assert lane.sends[0].leaf is client._real_client
    assert "effect already completed" in repr(lane.sends[0].body["messages"])
    assert lane.sends[0].body["model"] == lane.binds[-1]["model"] == "claude-sonnet-4-6"
    assert lane.binds[-1]["credential"] == "selected-B"
    assert lane.binds[-1]["base_url"] == "https://api.anthropic.com/"
    assert lane.binds[-1]["session_id"] == "owned-session"


def test_wire_model_extra_body_and_adapter_default_are_actual(anthropic_lane):
    lane = anthropic_lane
    client = lane.make()
    request = {"messages": [{"role": "user", "content": "original"}],
               "extra_body": {"model": "claude-haiku-4-5"}}
    aux._create_with_progress_once(client, request)
    assert lane.sends[0].body["model"] == lane.binds[-1]["model"] == "claude-haiku-4-5"
    selected = pacing._snapshot(client, {}, "anthropic")
    assert selected.model == "claude-sonnet-4-6" and selected.problem == ""


def test_retry_fresh_admission_uses_new_literal_leaf_token(anthropic_lane):
    lane = anthropic_lane
    client = lane.make()
    request = {"model": "claude-sonnet-4.6", "messages": [{"role": "user", "content": "original"}]}
    aux._create_with_progress_once(client, request)
    client._real_client.auth_token = "selected-C"
    lane.state.decision = Decision("hold", "weekly_limit")

    def release():
        assert len(lane.sends) == 1
        lane.state.decision = Decision("allow", "fresh")

    lane.state.on_wait = release
    aux._create_with_progress_once(client, request)
    assert len(lane.sends) == 2 and len(lane.attempts) == 2
    assert lane.attempts[0].lifetime is not lane.attempts[1].lifetime
    assert {row["credential"] for row in lane.binds} == {"selected-B", "selected-C"}
    assert lane.sends[1].headers["Authorization"] == "Bearer selected-C"


@pytest.mark.parametrize("change", ["token", "endpoint", "adapter_leaf"])
def test_selected_route_change_during_hold_never_sends(anthropic_lane, change):
    lane = anthropic_lane
    client = lane.make()
    lane.state.decision = Decision("hold", "weekly_limit")

    def mutate():
        if change == "token":
            client._real_client.auth_token = "unassociated-C"
        elif change == "endpoint":
            client._real_client.base_url = "https://other.invalid/"
        else:
            client.chat.completions._client = lane.sdk(auth_token="other")

    lane.state.on_wait = mutate
    with pytest.raises(InterruptedError):
        aux._create_with_progress_once(client, {"model": "claude-sonnet-4.6", "messages": []})
    assert lane.sends == []


@pytest.mark.parametrize("header", ["Authorization", "authorization", "X-Api-Key", "HOST"])
def test_request_auth_or_host_override_proceeds_unmetered_quietly(anthropic_lane, header, caplog):
    import logging
    lane = anthropic_lane
    client = lane.make()
    lane.state.on_wait = lambda: pytest.fail("a meter that cannot verify must not wait")
    with aux.aux_interrupt_protection(cancel_check=lambda: False):
        aux._create_with_progress_once(client, {
            "model": "claude-sonnet-4.6", "messages": [], "extra_headers": {header: "foreign"}})
    assert len(lane.sends) == 1
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "foreign" not in caplog.text


@pytest.mark.parametrize("header", ["Authorization", "X-Api-Key", "host"])
def test_configured_auth_override_uses_real_constructor_config_helper(anthropic_lane, header):
    lane = anthropic_lane
    lane.state.entries = [{"name": "owned", "base_url": "https://api.anthropic.com/",
                           "extra_headers": {header: "foreign"}}]
    client = lane.make()
    assert client._real_client._custom_headers[header] == "foreign"
    selected = pacing._snapshot(client, {"model": "claude-sonnet-4.6"}, "anthropic")
    assert selected.problem in ("credential_snapshot_required", "transport_snapshot_required")


def test_omit_unused_credential_is_safe_but_dual_auth_is_not(anthropic_lane):
    lane = anthropic_lane
    client = lane.make()
    client._real_client.api_key = "unused-ambient-key"
    request = {"model": "claude-sonnet-4.6"}
    selected = pacing._snapshot(client, request, "anthropic")
    assert selected.credential == "selected-B" and selected.problem == ""
    client._real_client._custom_headers.pop("X-Api-Key")
    assert pacing._snapshot(client, request, "anthropic").problem == "credential_snapshot_required"


@pytest.mark.parametrize("field", ["credentials", "_token_cache", "_custom_auth", "_api_key_provider"])
def test_auth_providers_never_admitted_or_called(anthropic_lane, field):
    lane = anthropic_lane
    client = lane.make()

    def forbidden():
        pytest.fail("auth provider called")

    setattr(client._real_client, field, forbidden)
    assert pacing._snapshot(client, {"model": "claude-sonnet-4.6"}, "anthropic").problem == "credential_snapshot_required"


@pytest.mark.parametrize("change", ["http_auth", "hook", "transport", "http_header", "middleware", "custom_sdk"])
def test_custom_transport_or_sdk_is_unsupported(anthropic_lane, change):
    lane = anthropic_lane
    client = lane.make()
    leaf = client._real_client
    if change == "http_auth":
        leaf._client.auth = object()
    elif change == "hook":
        leaf._client.event_hooks = {"request": [lambda request: None]}
    elif change == "transport":
        leaf._client._transport = object()
    elif change == "http_header":
        leaf._client.headers = {"X-Api-Key": "foreign"}
    elif change == "middleware":
        leaf._middleware_chain = lambda request: None
    else:
        class CustomSDK(lane.sdk):
            pass
        client._real_client = CustomSDK(auth_token="selected-B")
        client.chat.completions._client = client._real_client
    assert pacing._snapshot(client, {"model": "claude-sonnet-4.6"}, "anthropic").problem == "transport_snapshot_required"


def test_standard_anthropic_http_wrapper_is_supported(anthropic_lane, monkeypatch):
    from anthropic import _base_client
    lane = anthropic_lane

    class InertSDKHTTPWrapper(lane.http):
        pass

    monkeypatch.setattr(_base_client, "SyncHttpxClientWrapper", InertSDKHTTPWrapper)
    client = lane.make()
    client._real_client._client = InertSDKHTTPWrapper()
    assert pacing._snapshot(client, {"model": "claude-sonnet-4.6"}, "anthropic").problem == ""


def test_sync_cancelled_hold_is_not_provider_fallback(anthropic_lane):
    lane = anthropic_lane
    lane.state.decision = Decision("hold", "weekly_limit")
    lane.state.on_wait = lambda: setattr(lane.state, "stopped", True)
    with aux.aux_interrupt_protection(cancel_check=lambda: lane.state.stopped):
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            aux._create_with_progress(lane.make(), {"model": "claude-sonnet-4.6", "messages": []})
    lane.state.stopped = False
    assert lane.sends == [] and len(lane.attempts) == 1


def test_async_wrapper_wait_releases_same_sync_adapter_leaf(anthropic_lane):
    lane = anthropic_lane
    client = aux.AsyncAnthropicAuxiliaryClient(lane.make())
    client._hermes_aux_effective_provider = "anthropic"
    request = {"model": "claude-sonnet-4.6", "messages": [{"role": "user", "content": "original"}]}
    lane.state.decision = Decision("hold", "weekly_limit")

    def release():
        assert lane.sends == []
        lane.state.decision = Decision("allow", "fresh")

    lane.state.on_wait = release
    response = asyncio.run(aux._acreate_with_progress(client, request))
    assert response.choices[0].message.content == "summary"
    assert len(lane.sends) == 1
    assert lane.sends[0].leaf is client._real_client is client.chat.completions._sync._client
    assert lane.binds[-1]["credential"] == "selected-B"
    assert lane.binds[-1]["model"] == lane.sends[0].body["model"]


def test_async_task_cancel_while_policy_held_never_reaches_adapter(anthropic_lane):
    lane = anthropic_lane

    async def run():
        held, release = asyncio.Event(), asyncio.Event()

        async def policy(route, hold, budget):
            held.set()
            await release.wait()
            return Decision("hold", "weekly_limit")

        lane.state.async_policy = policy
        client = aux.AsyncAnthropicAuxiliaryClient(lane.make())
        client._hermes_aux_effective_provider = "anthropic"
        task = asyncio.create_task(aux._acreate_with_progress(
            client, {"model": "claude-sonnet-4.6", "messages": []}))
        try:
            await asyncio.wait_for(held.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
        assert lane.sends == [] and len(lane.attempts) == 1

    asyncio.run(run())


def test_default_disabled_does_not_snapshot_or_bind(anthropic_lane, monkeypatch):
    lane = anthropic_lane
    monkeypatch.setattr(pacing, "_section", dict)
    monkeypatch.setattr(pacing, "_snapshot", lambda *args: pytest.fail("disabled snapshot"))
    response = aux._create_with_progress_once(lane.make(), {
        "model": "claude-sonnet-4.6", "messages": [{"role": "user", "content": "stock"}]})
    assert response.choices[0].message.content == "summary"
    assert len(lane.sends) == 1 and lane.binds == [] and lane.attempts == []
