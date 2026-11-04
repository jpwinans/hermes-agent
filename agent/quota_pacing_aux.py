"""Admission at auxiliary dispatch seams; no inherited permit or SDK replacement.

C1/C4: a hold retains the original request, including compression's history.
Opaque transports stay visibly repair-blocked rather than silently unmetered.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agent.quota_pacing import Decision, Lifetime, PendingAttempt, PacingStopped


@dataclass(frozen=True, repr=False)
class _Snapshot:
    provider: str
    model: str
    credential: object
    base_url: str
    problem: str = ""


def _section():
    from agent.quota_pacing_config import section
    return section()


def _disabled(section):
    return isinstance(section, dict) and section.get("enabled", False) is False


def _http_snapshot_supported(leaf, *, anthropic=False):
    import httpx
    if anthropic:
        from anthropic._base_client import SyncHttpxClientWrapper, AsyncHttpxClientWrapper
    else:
        from openai._base_client import SyncHttpxClientWrapper, AsyncHttpxClientWrapper
    http = getattr(leaf, "_client", None)
    if http is None or type(http) not in (httpx.Client, httpx.AsyncClient, SyncHttpxClientWrapper, AsyncHttpxClientWrapper):
        return False
    if getattr(http, "auth", None) is not None:
        return False
    if anthropic and any(str(key).lower() in ("authorization", "x-api-key", "host")
                         for key in getattr(http, "headers", {})):
        return False
    # Baseline keepalive clients use native httpx transports/shared pool views,
    # plus only the body-compression hook. Unknown HTTP auth/hooks/transports
    # could replace the bearer or destination AFTER SDK request construction.
    for hooks in http.event_hooks.values():
        for hook in hooks:
            if (hook.__module__, hook.__qualname__) not in (
                ("agent.codex_headers", "codex_request_compression_hooks.<locals>.compress"),
                ("agent.codex_headers", "codex_request_compression_hooks.<locals>.compress_async"),
            ):
                return False
    for transport in (http._transport, *http._mounts.values()):
        if transport is None:
            continue
        if (type(transport).__module__, type(transport).__qualname__) == (
            "agent.process_bootstrap", "_SharedTransport"
        ):
            transport = transport._inner
        if type(transport) not in (httpx.HTTPTransport, httpx.AsyncHTTPTransport):
            return False
    return True


def _headers_problem(leaf, request, credential, provider):
    from agent.codex_headers import codex_account_headers
    expected = {key.lower(): value for key, value in codex_account_headers(credential).items()} if provider == "openai-codex" else {}
    headers = (getattr(leaf, "default_headers", {}), request.get("extra_headers") or {})
    merged = {}
    if any(not hasattr(value, "items") for value in headers):
        return "transport_snapshot_required"
    for index, value in enumerate(headers):
        for key, content in value.items():
            key = str(key).lower()
            merged[key] = content
            if key in ("host", "x-api-key"):
                return "transport_snapshot_required"
            if key == "authorization" and (index != 0 or content != f"Bearer {credential}"):
                return "credential_snapshot_required"
            if key in ("chatgpt-account-id", "x-openai-internal-codex-residency") and content != expected.get(key):
                return "credential_snapshot_required"
    if any(merged.get(key) != value for key, value in expected.items()):
        return "credential_snapshot_required"
    return ""


def _anthropic_credential(leaf, request):
    from anthropic import Omit
    # SDK default_headers already includes static auth + configured Omit markers.
    # Drop Omit BEFORE case-folding, just as the SDK does: an omitted unused key
    # is safe, but duplicate active case variants or dual auth are ambiguous.
    headers = getattr(leaf, "default_headers", {})
    overrides = request.get("extra_headers") or {}
    if not hasattr(headers, "items") or not hasattr(overrides, "items"):
        return None, "transport_snapshot_required"
    active = {}
    for key, value in headers.items():
        key = str(key).lower()
        if isinstance(value, Omit):
            continue
        if key == "host":
            return None, "transport_snapshot_required"
        if key in ("authorization", "x-api-key"):
            if key in active:
                return None, "credential_snapshot_required"
            active[key] = value
    for key in overrides:
        if str(key).lower() in ("authorization", "x-api-key", "host"):
            return None, "credential_snapshot_required"
    if len(active) != 1:
        return None, "credential_snapshot_required"
    bearer = "authorization" in active
    credential = getattr(leaf, "auth_token" if bearer else "api_key", None)
    expected = f"Bearer {credential}" if bearer else credential
    if not isinstance(credential, str) or not credential or next(iter(active.values())) != expected:
        return None, "credential_snapshot_required"
    if any(getattr(leaf, field, None) is not None for field in (
        "credentials", "_token_cache", "_custom_auth", "custom_auth", "_api_key_provider",
    )):
        return None, "credential_snapshot_required"
    return credential, ""


def _anthropic_snapshot(client, request, provider):
    from anthropic import Anthropic
    from agent.auxiliary_client import (
        AsyncAnthropicAuxiliaryClient, _AnthropicCompletionsAdapter, _AsyncCompletionsAdapter,
    )
    from agent.anthropic_endpoints import _is_nous_portal_endpoint
    from agent.anthropic_message_convert import normalize_model_name

    leaf = client._real_client
    adapter = client.chat.completions
    if type(client) is AsyncAnthropicAuxiliaryClient:
        if type(adapter) is not _AsyncCompletionsAdapter:
            return _Snapshot(provider, "", None, "", "transport_snapshot_required")
        adapter = adapter._sync
    if (type(leaf) is not Anthropic or type(adapter) is not _AnthropicCompletionsAdapter
            or adapter._client is not leaf):
        return _Snapshot(provider, "", None, "", "transport_snapshot_required")
    credential, problem = _anthropic_credential(leaf, request)
    if (not _http_snapshot_supported(leaf, anthropic=True)
            or getattr(leaf, "_middleware", ()) or getattr(leaf, "_middleware_chain", None)):
        problem = "transport_snapshot_required"
    # Mirror the actual adapter's fixed model conversion and extra_body passthrough,
    # not the wrapper's display model/key/URL. No worker refreshes this SDK leaf.
    model = request.get("model", adapter._model)
    if not _is_nous_portal_endpoint(adapter._base_url):
        model = normalize_model_name(model)
    body = request.get("extra_body")
    if isinstance(body, dict):
        model = body.get("model", model)
    return _Snapshot(str(provider or ""), str(model or ""), credential,
                     str(getattr(leaf, "base_url", "") or ""), problem)


def _snapshot(client, request, provider):
    from openai import OpenAI, AsyncOpenAI
    from agent.auxiliary_client import (
        CodexAuxiliaryClient, AsyncCodexAuxiliaryClient,
        AnthropicAuxiliaryClient, AsyncAnthropicAuxiliaryClient,
    )

    if type(client) in (AnthropicAuxiliaryClient, AsyncAnthropicAuxiliaryClient):
        return _anthropic_snapshot(client, request, provider)
    # Only existing destinations whose selected SDK leaf is the actual sender.
    leaf = client
    codex = type(client) in (CodexAuxiliaryClient, AsyncCodexAuxiliaryClient)
    adapter = None
    problem = ""
    if codex:
        leaf = client._real_client
        adapter = client.chat.completions
        if type(client) is AsyncCodexAuxiliaryClient:
            adapter = adapter._sync
        if getattr(adapter, "_client", None) is not leaf:
            problem = "transport_snapshot_required"
    elif type(client) not in (OpenAI, AsyncOpenAI):
        problem = "transport_snapshot_required"
    if type(leaf) not in (OpenAI, AsyncOpenAI):
        problem = "transport_snapshot_required"
    if not _http_snapshot_supported(leaf):
        problem = "transport_snapshot_required"
    credential = getattr(leaf, "api_key", None)
    base_url = str(getattr(leaf, "base_url", "") or "")
    if not isinstance(credential, str) or not credential or callable(getattr(leaf, "_api_key_provider", None)):
        problem = "credential_snapshot_required"
    # Default and request overrides must not replace the bearer/host after binding.
    problem = _headers_problem(leaf, request, credential, provider) or problem
    body = request.get("extra_body")
    model = body.get("model", request.get("model", "")) if isinstance(body, dict) else request.get("model", "")
    if adapter is not None:
        from agent.codex_responses_adapter import _wire_model_identity
        model = _wire_model_identity(request.get("model", adapter._model))
    return _Snapshot(str(provider or ""), str(model or ""), credential, base_url, problem)


def _attempt(client, request, provider, cancel_check, wait_hook, session_id, warning_callback=None):
    from agent.quota_pacing_native import bind_request
    selected = _snapshot(client, request, provider)
    lifetime = Lifetime()
    stopped = False

    def cancelled():
        nonlocal stopped
        try:
            stopped = stopped or (cancel_check is not None and bool(cancel_check()))
        except Exception:
            stopped = True
        if stopped:
            lifetime.cancel()
        return stopped

    def bind(section):
        return bind_request(section, provider=selected.provider, model=selected.model,
                            credential=selected.credential, base_url=selected.base_url,
                            session_id=session_id, cancelled=cancelled)

    binding = bind(_section())

    def current_route():
        if cancelled():
            raise PacingStopped("cancelled")
        if _snapshot(client, request, provider) != selected:
            raise PacingStopped("route_changed")
        return binding.route

    last_status = None

    def waiting(decision):
        nonlocal last_status
        if wait_hook is not None and wait_hook(decision) is False:
            lifetime.cancel()
            raise PacingStopped("cancelled")
        status = (decision.action, decision.reason) if decision is not None else None
        if status == last_status:
            return
        last_status = status
        if decision is not None and decision.action == "hold":
            # A real pace pause only; never the selected route/secret.
            from agent.session_activity import quota_wait_message
            message = quota_wait_message(decision)
            logging.getLogger(__name__).warning("%s", message)
            # Context-captured native owner, outside all cancellation/runtime locks.
            # A UI failure is presentation-only, never permission or a send retry.
            if callable(warning_callback):
                try:
                    warning_callback(message)
                except Exception:
                    pass

    def replacement():
        nonlocal binding
        section = _section()
        if _disabled(section):
            return None
        if selected.problem:
            return Decision("blocked", selected.problem)
        updated = bind(section)
        if updated.route != binding.route:
            raise PacingStopped("route_changed")
        binding = updated
        return binding

    def policy(route, hold, budget):
        updated = replacement()
        if updated is None:
            return Decision("allow", "operator_disabled", active_hold=hold)
        if isinstance(updated, Decision):
            return updated
        return updated.policy(route, hold, budget)

    async def async_policy(route, hold, budget):
        updated = replacement()
        if updated is None:
            return Decision("allow", "operator_disabled", active_hold=hold)
        if isinstance(updated, Decision):
            return updated
        return await updated.async_policy(route, hold, budget)

    attempt = PendingAttempt(route=binding.route, lifetime=lifetime,
                             current_route=current_route, on_waiting=waiting)
    return attempt, policy, async_policy


def admit(client, request, *, provider="", cancel_check=None, wait_hook=None, session_id="", warning_callback=None):
    """Called immediately before the existing sync SDK dispatch, not around retries."""
    if _disabled(_section()):
        return
    try:
        attempt, policy, _ = _attempt(client, request, provider, cancel_check, wait_hook, session_id, warning_callback)
    except Exception:
        logging.getLogger(__name__).info("quota pacing unavailable for this auxiliary call; proceeding")
        return  # a meter that cannot set up is not a pace hold
    attempt.wait(policy)
    attempt.check_before_send()


async def admit_async(client, request, *, provider="", cancel_check=None, wait_hook=None, session_id="", warning_callback=None):
    if _disabled(_section()):
        return
    try:
        attempt, _, policy = _attempt(client, request, provider, cancel_check, wait_hook, session_id, warning_callback)
    except Exception:
        logging.getLogger(__name__).info("quota pacing unavailable for this auxiliary call; proceeding")
        return
    await attempt.wait_async(policy)
    attempt.check_before_send()


def _current_inputs(client) -> dict[str, Any]:
    from agent import auxiliary_client as aux
    context = aux._RELAY_AUX_CALL_CONTEXT.get()
    if context is not None and context.get("quota_stopped"):
        raise PacingStopped("cancelled")
    cancel_check = aux._capture_aux_cancel_check()
    if cancel_check is None:
        cancel_check = context.get("quota_cancel_check") if context is not None else None
        if cancel_check is None:
            cancel_check = aux._capture_aux_cancel_check(native_fallback=True)
            if context is not None:
                context["quota_cancel_check"] = cancel_check
    context = context or {}
    return dict(provider=aux._effective_provider_for_client(
        client, context.get("stream_provider") or context.get("provider") or ""),
        cancel_check=cancel_check,
        wait_hook=getattr(aux._aux_stream_deadline, "quota_wait", None),
        session_id=str(aux._runtime_main_value("session_id") or ""),
        warning_callback=aux._runtime_main_value("quota_warning_callback"))


def _stopped(stopped):
    from agent import auxiliary_client as aux
    if stopped.args and stopped.args[0] in ("route_changed", "route_unavailable"):
        aux.logger.info("auxiliary call's route changed during a pacing wait; stopping that call")
    if aux._aux_interrupt_protected() and aux._capture_aux_cancel_check() is not None:
        raise aux.AuxiliaryExplicitCancellation() from None
    # The stock stream-negotiation recovery catches Exception and may re-enter
    # admission. Freeze this logical call's stop, including route repair, so that
    # neither a cleared native bit nor a newly valid route can cause a late send.
    context = aux._RELAY_AUX_CALL_CONTEXT.get()
    if context is not None:
        context["quota_stopped"] = True
    raise InterruptedError("Auxiliary quota admission stopped") from None


def admit_current(client, request):
    from agent import auxiliary_client as aux
    context = aux._RELAY_AUX_CALL_CONTEXT.get() or {}
    if _disabled(_section()) and not context.get("quota_stopped"):
        return
    try:
        admit(client, request, **_current_inputs(client))
    except PacingStopped as stopped:
        _stopped(stopped)
    except Exception:
        _stopped(PacingStopped("route_unavailable"))


async def admit_current_async(client, request):
    from agent import auxiliary_client as aux
    context = aux._RELAY_AUX_CALL_CONTEXT.get() or {}
    if _disabled(_section()) and not context.get("quota_stopped"):
        return
    try:
        await admit_async(client, request, **_current_inputs(client))
    except PacingStopped as stopped:
        _stopped(stopped)
    except Exception:
        _stopped(PacingStopped("route_unavailable"))
