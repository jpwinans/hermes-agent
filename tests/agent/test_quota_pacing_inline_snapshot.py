"""AUTHORED source-only regressions; parent executes in isolated containment.

Real owner gate, selector, dispatch, lifecycle/cache and request drivers. Only
SDK construction/send, native quota binding, clocks/watchdogs and Relay's outer
stream wrapper are inert boundaries. No credentials or network are discovered.
"""
from copy import deepcopy
from types import SimpleNamespace
import threading

import pytest

from agent import chat_completion_helpers as h
from agent import quota_pacing_host as host
from agent import quota_pacing_native as native
from agent.client_lifecycle import ClientLifecycleMixin
from agent.quota_pacing import Decision, PendingAttempt, Route


class OwnedSDK:
    """Inert SDK boundary: never constructs an HTTP client."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.api_key = kwargs["api_key"]
        self.base_url = kwargs["base_url"]
        self._custom_headers = dict(kwargs.get("default_headers") or {})
        self._custom_query = dict(kwargs.get("default_query") or {})
        self._api_key_provider = None
        self.is_closed = False
        self.sends = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        assert "_credential_snapshot" not in kwargs
        self.sends.append(kwargs)
        return SimpleNamespace(choices=[])

    def close(self):
        self.is_closed = True


class Owner(ClientLifecycleMixin):
    def _interruptible_api_call(self, kwargs, **private):
        return h.interruptible_api_call(self, kwargs, **private)

    def _create_openai_client(self, kwargs, *, reason, shared):
        assert not shared
        client = OwnedSDK(**kwargs)
        self.built.append(client)
        return client

    def _close_openai_client(self, client, *, reason, shared):
        client.close()

    def _touch_activity(self, message):
        pass

    def _stream_diag_init(self):
        return {}

    def _buffer_status(self, message):
        pass

    def _force_close_tcp_sockets(self, client):
        self.aborts.append(client)
        return 1

    def _run_codex_stream(self, kwargs, *, client, on_first_delta=None):
        self.codex_clients.append(client)
        return client.create(**kwargs)


def request():
    return {"model": "owned-model", "messages": [
        {"role": "system", "content": "Preserve original goal and context"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:owned-attachment"}}]},
        {"role": "assistant", "tool_calls": [{"id": "done", "type": "function", "function": {"name": "effect", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "done", "content": "completed side effect; do not repeat"},
    ], "timeout": 1}


@pytest.fixture
def owned(monkeypatch):
    import socket
    import openai
    from agent import provider_route_hooks
    def no_network(*args, **kwargs):
        pytest.fail("owned snapshot regression attempted network")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(openai, "OpenAI", OwnedSDK)
    monkeypatch.setattr(provider_route_hooks, "route_answer", lambda *a, **k: {})
    a = Owner()
    a.provider, a.api_mode, a.platform = "openai", "chat_completions", "cron"
    a.model, a.session_id = "owned-model", "owned-session"
    a.base_url = "https://original.invalid/v1"
    a._client_kwargs = {"api_key": "owned-original", "base_url": a.base_url,
                        "default_headers": {"X-Owned": "original"}, "default_query": {"tenant": "original"}}
    a.client = OwnedSDK(**a._client_kwargs)
    a.built, a.aborts, a.codex_clients = [], [], []
    a.ensures = []
    a._interrupt_requested = False
    a._pending_redirect_lock = threading.Lock()
    # An admitted constructor must never re-ensure/adopt a mutable primary.
    # Stock-only tests explicitly replace this boundary below.
    def ensure(**kwargs):
        a.ensures.append(kwargs)
        return a.client
    a._ensure_primary_openai_client = ensure
    monkeypatch.setattr(h._InlineRequest, "start_watchdogs", lambda self: None)
    monkeypatch.setattr(h._InlineRequest, "stop_watchdogs", lambda self: self.mark_done())
    monkeypatch.setattr(h, "_resolve_direct_stale_timeout", lambda *a: float("inf"))
    monkeypatch.setattr(h, "_resolve_nonstream_watchdogs", lambda *a: SimpleNamespace(
        codex=False, idle_requires_progress=False, ttfb_enabled=False, ttfb_timeout=0,
        progress_timeout=0, idle_enabled=False, idle_timeout=0, stale_timeout=float("inf")))
    return a


@pytest.fixture
def gate(monkeypatch):
    state = SimpleNamespace(now=0.0, during_wait=None, binds=[], holds=0)
    monkeypatch.setattr(host, "_section", lambda: {"enabled": True})
    real_wait = PendingAttempt.wait

    def advance(seconds):
        state.now += seconds
        if state.during_wait:
            state.during_wait()

    def wait(attempt, policy):
        return real_wait(attempt, policy, clock=lambda: state.now, wait=advance)

    def bind(section, **kwargs):
        state.binds.append(kwargs)
        route = Route(kwargs["provider"], kwargs["model"], "owned-account")

        def policy(actual, hold, budget):
            if state.holds:
                state.holds -= 1
                return Decision("hold", "owned_hold")
            return Decision("allow", "owned_allow")
        return SimpleNamespace(route=route, policy=policy)

    monkeypatch.setattr(PendingAttempt, "wait", wait)
    monkeypatch.setattr(native, "bind_request", bind)
    return state


def mutate(a, *, key=False, endpoint=False, headers=False):
    if key:
        a._client_kwargs["api_key"] = "owned-later"
        a.client.api_key = "owned-later"
    if endpoint:
        a._client_kwargs["base_url"] = a.base_url = "https://later.invalid/v1"
        a.client.base_url = a.base_url
    if headers:
        a._client_kwargs["default_headers"]["X-Owned"] = "later"
        a._client_kwargs["default_query"]["tenant"] = "later"


def admitted(a, req, mutation, entry=h.interruptible_api_call):
    def send(exact, **private):
        assert exact is req  # mutation is AFTER check_before_send, not a gate test
        mutation()
        return entry(a, exact, **private)
    return host.paced_call(a, req, send)


def assert_original(a, req, original):
    assert req == original
    assert len(a.built) == 1
    client = a.built[0]
    assert client.api_key == "owned-original"
    assert client.base_url == "https://original.invalid/v1"
    assert client.kwargs["default_headers"] == {"X-Owned": "original"}
    assert client.kwargs["default_query"] == {"tenant": "original"}
    assert client.kwargs["max_retries"] == 0
    assert len(client.sends) == 1
    assert client.sends[0]["messages"] == original["messages"]
    assert not a._request_client_cache["in_use"]
    assert a._quota_pacing_lifetimes == set()
    assert a.ensures == []
    return client


def test_cron_chat_late_credential_mutation(owned, gate):
    req = request(); original = deepcopy(req)
    assert h.should_use_direct_api_call(owned)
    admitted(owned, req, lambda: mutate(owned, key=True))
    assert_original(owned, req, original)


def test_delegated_context_late_endpoint_headers_preserves_completed_context(owned, gate):
    from agent.delegation_context import _DELEGATED_CHILD_CONTEXT
    owned.platform = "cli"
    token = _DELEGATED_CHILD_CONTEXT.set(True)
    try:
        assert h.should_use_direct_api_call(owned)
        req = request(); original = deepcopy(req)
        admitted(owned, req, lambda: mutate(owned, endpoint=True, headers=True))
        assert_original(owned, req, original)
    finally:
        _DELEGATED_CHILD_CONTEXT.reset(token)


def test_cron_codex_late_key_and_endpoint(owned, gate):
    owned.provider, owned.api_mode = "openai-codex", "codex_responses"
    req = request(); original = deepcopy(req)
    assert h.should_use_direct_api_call(owned)
    admitted(owned, req, lambda: mutate(owned, key=True, endpoint=True))
    assert owned.codex_clients == [assert_original(owned, req, original)]


def test_ordinary_threaded_dispatch_consumes_same_snapshot(owned, gate):
    owned.platform = "cli"
    assert not h.should_use_direct_api_call(owned)
    req = request(); original = deepcopy(req)
    admitted(owned, req, lambda: mutate(owned, key=True, endpoint=True, headers=True))
    assert_original(owned, req, original)


def test_streaming_wire_dispatch_consumes_snapshot(owned, gate, monkeypatch):
    from agent import relay_llm
    # Relay wrapper boundary only: real _call_wire -> _call_chat_completions ->
    # _open_chat_stream -> lifecycle -> inert SDK create. Empty owned stream with
    # a completed response avoids emulating provider chunks/consumer callbacks.
    class EmptyStream:
        def __init__(self, final):
            self.final_response = final
        def __iter__(self):
            return iter(())
        def close(self):
            pass
    monkeypatch.setattr(relay_llm, "stream", lambda req, open_stream, **kw: EmptyStream(open_stream(req)))
    monkeypatch.setattr(h._StreamingCall, "_stream_timeouts", lambda self: (1, 1, 1))
    req = request(); original = deepcopy(req)

    def entry(a, exact, **private):
        call = h._StreamingCall(a, exact, None, **private)
        try:
            return call._call_wire(call._start_stream_attempt())
        finally:
            call.clients.close_once("stream_request_complete")
    admitted(owned, req, lambda: mutate(owned, key=True, endpoint=True, headers=True), entry)
    assert_original(owned, req, original)


def test_fresh_next_admission_uses_new_identity_not_cached_snapshot(owned, gate):
    req = request()
    admitted(owned, req, lambda: mutate(owned, key=True, endpoint=True, headers=True))
    assert owned.built[0].api_key == "owned-original"
    admitted(owned, req, lambda: None)
    assert len(owned.built) == 2
    assert owned.built[0].is_closed
    assert owned.built[1].api_key == "owned-later"
    assert owned.built[1].base_url == "https://later.invalid/v1"
    assert owned.built[1].kwargs["default_headers"] == {"X-Owned": "later"}
    assert all(len(c.sends) == 1 for c in owned.built)


def test_snapshot_is_detached_immutable_and_reuses_then_aborts_real_slot(owned, gate):
    selected = host._selected(owned, request())
    snap = selected.client_key
    assert snap is not None
    assert "owned-original" not in repr(snap)
    with pytest.raises(AttributeError):
        snap.provider = "forbidden"
    with pytest.raises(TypeError):
        snap.kwargs["api_key"] = "forbidden"
    for name in ("default_headers", "default_query"):
        with pytest.raises(TypeError):
            snap.kwargs[name]["new"] = "forbidden"
    mutate(owned, key=True, endpoint=True, headers=True)
    one = owned._create_request_openai_client(reason="owned", _credential_snapshot=snap)
    owned._close_request_openai_client(one, reason="request_complete")
    two = owned._create_request_openai_client(reason="owned", _credential_snapshot=snap)
    assert one is two and one.api_key == "owned-original"
    owned._abort_request_openai_client(two, reason="owned_abort")
    assert owned._request_client_cache["poisoned"]
    owned._close_request_openai_client(two, reason="request_complete")
    assert two.is_closed and owned.aborts == [two]
    three = owned._create_request_openai_client(reason="owned", _credential_snapshot=snap)
    assert three is not two
    owned._close_request_openai_client(three, reason="request_complete")


def test_stock_no_snapshot_call_shape_and_cache_reuse(owned, monkeypatch):
    owned._ensure_primary_openai_client = lambda **kw: owned.client
    real = owned._create_request_openai_client
    # No private kw accepted: stock inline must retain the preexisting shape.
    def stock(*, reason, api_kwargs):
        return real(reason=reason, api_kwargs=api_kwargs)
    monkeypatch.setattr(owned, "_create_request_openai_client", stock)
    req = request()
    h.interruptible_api_call(owned, req)
    h.interruptible_api_call(owned, req)
    assert len(owned.built) == 1 and len(owned.built[0].sends) == 2
    assert not owned._request_client_cache["in_use"]


@pytest.mark.parametrize("field,value", [("provider", "moa"), ("api_mode", "anthropic_messages"),
                                         ("api_mode", "bedrock_converse")])
def test_late_unsafe_provider_mode_interrupts_before_any_dispatch(owned, gate, field, value):
    with pytest.raises(InterruptedError):
        admitted(owned, request(), lambda: setattr(owned, field, value))
    assert owned._interrupt_requested
    assert owned.built == [] and owned.codex_clients == []


@pytest.mark.parametrize("header", ["Authorization", "X-API-Key", "Host"])
def test_late_image_route_cannot_override_protected_headers(owned, gate, monkeypatch, header):
    from agent import provider_route_hooks
    monkeypatch.setattr(provider_route_hooks, "route_answer", lambda *a, **kw: {header: "owned-override"})
    with pytest.raises(InterruptedError):
        admitted(owned, request(), lambda: None)
    assert owned._interrupt_requested and owned.built == []


def test_constructor_header_change_while_waiting_interrupts_before_send(owned, gate):
    gate.holds = 1
    gate.during_wait = lambda: mutate(owned, headers=True)
    with pytest.raises(InterruptedError):
        host.paced_call(owned, request(), lambda *a, **kw: pytest.fail("changed route sent"))
    assert owned._interrupt_requested and owned.built == []


def test_unsupported_snapshot_never_silently_uses_live_primary(owned):
    with pytest.raises(InterruptedError):
        h.direct_api_call(owned, request(), _credential_snapshot=("not-openai",))
    assert owned._interrupt_requested and owned.built == []


def test_codex_streaming_passthrough_retains_admitted_constructor(owned, gate):
    owned.provider, owned.api_mode = "openai-codex", "codex_responses"
    req = request(); original = deepcopy(req)
    admitted(owned, req, lambda: mutate(owned, key=True, endpoint=True), h.interruptible_streaming_api_call)
    assert owned.codex_clients == [assert_original(owned, req, original)]
    assert owned._codex_on_first_delta is None


def test_stock_mock_primary_still_returns_without_reconstruction(owned):
    from unittest.mock import Mock
    primary = Mock()
    owned._ensure_primary_openai_client = lambda **kw: primary
    assert owned._create_request_openai_client(reason="stock") is primary
    assert owned.built == []


@pytest.mark.parametrize("cached", [False, True])
def test_late_builder_or_cached_header_override_fails_before_wire(owned, gate, monkeypatch, cached):
    req = request()
    if cached:
        # Populate the real slot through stock construction, then emulate an
        # unsafe late mutation of that owned client, not owner kwargs.
        owned._ensure_primary_openai_client = lambda **kw: owned.client
        h.interruptible_api_call(owned, req)
        owned.built[0]._custom_headers["Host"] = "later.invalid"
    else:
        original_builder = owned._create_openai_client
        def build(kwargs, **options):
            kwargs["default_headers"]["Authorization"] = "owned-override"
            return original_builder(kwargs, **options)
        monkeypatch.setattr(owned, "_create_openai_client", build)
    with pytest.raises(InterruptedError):
        admitted(owned, req, lambda: None)
    assert owned._interrupt_requested
    assert len(owned.built) == 1 and owned.built[0].is_closed
    assert len(owned.built[0].sends) == (1 if cached else 0)
    assert owned._request_client_cache["client"] is None


def test_stream_server_error_probe_keeps_admitted_constructor(owned, gate):
    owned._stream_5xx_probe_ts = None
    req = request(); original = deepcopy(req)
    class ServerFailure(Exception):
        status_code = 500
    def entry(a, exact, **private):
        call = h._StreamingCall(a, exact, None, **private)
        call._stream_stale_timeout = 1
        assert call._unmask_server_error_with_nonstreaming(ServerFailure())
        return call.result["response"]
    admitted(owned, req, lambda: mutate(owned, key=True, endpoint=True, headers=True), entry)
    assert_original(owned, req, original)


@pytest.mark.parametrize("change", ["credential", "endpoint", "callable", "http_client"])
def test_unsupported_selected_transport_proceeds_unmetered(owned, gate, change):
    if change == "credential":
        owned.client.api_key = "different-primary"
    elif change == "endpoint":
        owned.client.base_url = "https://different-primary.invalid/v1"
    elif change == "callable":
        owned.client._api_key_provider = lambda: "must-not-be-called"
    else:
        owned._client_kwargs["http_client"] = object()
    assert host._selected(owned, request()).problem
    gate.during_wait = lambda: pytest.fail("a meter that cannot verify must not wait")
    sent = []
    host.paced_call(owned, request(), lambda *args, **kwargs: sent.append(args[0]))
    assert len(sent) == 1
    assert owned._quota_wait_message is None
