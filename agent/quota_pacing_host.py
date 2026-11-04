"""Owner-thread quota admission for already-built main/side-question/summary requests.

C1/C2: retain the original kwargs and completed results on the caller's stack.
C3: reload operator repair while waiting. C4: bind the selected route afresh for
all logical attempts. Unsupported mutable transports fail closed, never inherit
an admission. This slice does not implement auxiliary/compression/cron ownership.
"""
from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Mapping
from dataclasses import dataclass
import logging

from agent.quota_pacing import Decision, Lifetime, PendingAttempt, PacingStopped


def _section():
    from agent.quota_pacing_config import section
    return section()


def _disabled(section):
    return isinstance(section, dict) and section.get("enabled", False) is False


@dataclass(frozen=True, repr=False)
class _Selected:
    provider: str
    model: str
    credential: object
    base_url: str
    mode: str
    configured_url: str
    problem: str = ""
    client_key: object = None


def _openai_snapshot(agent, request, kwargs):
    from openai import OpenAI
    client = getattr(agent, "client", None)
    credential, base_url = kwargs.get("api_key"), kwargs.get("base_url", "")
    if type(client) is not OpenAI or agent.provider == "moa":
        return credential, base_url, "transport_snapshot_required"
    if not isinstance(credential, str) or callable(getattr(client, "_api_key_provider", None)) or client.api_key != credential:
        return credential, base_url, "credential_snapshot_required"
    actual_url = str(client.base_url)
    if not isinstance(base_url, str) or not base_url or base_url.rstrip("/") != actual_url.rstrip("/"):
        return credential, actual_url, "transport_snapshot_required"
    problem = _openai_constructor_problem(kwargs)
    if not problem:
        problem = _transport_problem(
            {**kwargs, "default_headers": dict(kwargs.get("default_headers") or {})}, request)
    return credential, actual_url, problem


def _openai_constructor_problem(kwargs):
    # Only scalar identity-bearing settings are supported by the shallow freeze.
    headers = kwargs.get("default_headers") or {}
    if not isinstance(headers, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()
    ):
        return "transport_snapshot_required"
    query = kwargs.get("default_query")
    if query is not None and (not isinstance(query, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, (str, int, float, bool, type(None)))
        for key, value in query.items()
    )):
        return "transport_snapshot_required"
    return ""


def _transport_problem(kwargs, request):
    # Overrides can supersede a selected bearer/endpoint after admission.
    header_sets = (kwargs.get("default_headers") or {}, request.get("extra_headers") or {})
    if any(not isinstance(headers, dict) for headers in header_sets):
        return "transport_snapshot_required"
    if any(str(key).lower() in ("authorization", "x-api-key", "host") for headers in header_sets for key in headers):
        return "transport_snapshot_required"
    if any(kwargs.get(key) is not None for key in ("http_client", "azure_ad_token_provider")):
        return "transport_snapshot_required"
    return ""


def _selected(agent, request):
    mode = getattr(agent, "api_mode", "")
    provider = getattr(agent, "provider", "")
    extra_body = request.get("extra_body")
    model = extra_body.get("model", request.get("model", "")) if isinstance(extra_body, dict) else request.get("model", "")
    kwargs = getattr(agent, "_client_kwargs", None)
    client_key = None
    if mode == "anthropic_messages":
        # Freeze the native immutable key locally, without refresh or publication.
        client_key = agent._request_anthropic_client_key()
        credential, base_url = client_key[1:3]
        problem = _transport_problem(kwargs if isinstance(kwargs, dict) else {}, request)
        from agent.anthropic_adapter import _custom_provider_extra_headers
        problem = problem or _transport_problem(
            {"default_headers": _custom_provider_extra_headers(base_url)}, request)
        if client_key[0] != "direct" or not isinstance(credential, str) or not credential:
            problem = "transport_snapshot_required"
    elif mode in ("chat_completions", "codex_responses") and isinstance(kwargs, dict):
        from agent.client_lifecycle import _OpenAIConstructorSnapshot
        client_key = _OpenAIConstructorSnapshot(provider, mode, kwargs)
        # Validate the detached values against this exact selected SDK client.
        credential, base_url, problem = _openai_snapshot(agent, request, client_key.kwargs)
    else:
        credential, base_url, problem = None, "", "transport_snapshot_required"
    configured_url = str(kwargs.get("base_url", "")) if isinstance(kwargs, dict) else str(base_url or "")
    return _Selected(provider, model, credential, str(base_url or ""), mode, configured_url, problem, client_key)


def _redirect_lock(agent):
    lock = getattr(agent, "_pending_redirect_lock", None)
    return lock if lock is not None else nullcontext()


def paced_call(agent, request, send):
    """Gate one actual request before entering native workers/response clocks.

    send receives the exact original object. Bounded native transport retries are
    inside this admission; a later owner retry/summary call creates a fresh one.
    """
    section = _section()
    if _disabled(section):
        return send(request)
    from agent.quota_pacing_native import bind_request
    from agent.session_activity import quota_wait_message, set_quota_wait

    selected = _selected(agent, request)
    lifetime = Lifetime()
    with _redirect_lock(agent):
        lifetimes = vars(agent).setdefault("_quota_pacing_lifetimes", set())
        lifetimes.add(lifetime)
        if getattr(agent, "_interrupt_requested", False):
            lifetime.cancel()

    def cancelled():
        if getattr(agent, "_interrupt_requested", False):
            lifetime.cancel()
        try:
            lifetime.check(binding.route, lambda: binding.route)
        except PacingStopped:
            return True
        return False

    def bind(config):
        return bind_request(
            config, provider=selected.provider, model=selected.model,
            credential=selected.credential, base_url=selected.base_url,
            session_id=str(getattr(agent, "session_id", "") or ""), cancelled=cancelled,
        )

    def current_route():
        if cancelled():
            raise PacingStopped("cancelled")
        if _selected(agent, request) != selected:
            raise PacingStopped("route_changed")
        return binding.route

    def policy(route, active_hold, budget):
        nonlocal binding
        config = _section()
        if _disabled(config):
            # Explicit operator disable is the same stock opt-out as at entry.
            return Decision("allow", "operator_disabled", active_hold=active_hold)
        if selected.problem:
            return Decision("blocked", selected.problem, active_hold=active_hold)
        replacement = bind(config)
        if replacement.route != binding.route:
            raise PacingStopped("route_changed")
        binding = replacement
        return binding.policy(route, active_hold, budget)

    try:
        set_quota_wait(agent, Decision("hold", "checking_quota"))
        binding = bind(section)  # Local only; private bearer never stored on agent.
        attempt = PendingAttempt(
            route=binding.route, lifetime=lifetime, current_route=current_route,
            on_waiting=lambda decision: set_quota_wait(agent, decision),
        )
        attempt.wait(policy)
        if getattr(agent, "_quota_wait_status", None) is not None:
            set_quota_wait(agent, None)
        with _redirect_lock(agent):
            attempt.check_before_send()
        if selected.client_key is not None:
            return send(request, _credential_snapshot=selected.client_key)
        return send(request)
    except PacingStopped as stopped:
        if stopped.args and stopped.args[0] in ("route_changed", "route_unavailable"):
            logging.getLogger(__name__).info("request route changed during a pacing wait; stopping before send")
        # Native callers preserve/persist the input and summary cleanup on this
        # interruption. Never let quota control become empty success or fallback.
        with _redirect_lock(agent):
            agent._interrupt_requested = True
            if not getattr(agent, "_tool_interrupt_reason", None):
                agent._tool_interrupt_reason = "quota pacing stopped"
        raise InterruptedError("Quota pacing interrupted before provider request") from None
    finally:
        if getattr(agent, "_quota_wait_status", None) is not None:
            set_quota_wait(agent, None)
        with _redirect_lock(agent):
            lifetimes.discard(lifetime)
