"""Source-only candidate: real compression fences/dispatch, fake quota and history.

No provider, SDK constructor, credentials, native reader or lock-store access.
"""
import concurrent.futures
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import conversation_compression as compression
from agent import quota_pacing_aux as pacing
from agent import quota_pacing_native as native
from agent.conversation_compression_deadline import compression_deadline_scope
from agent.quota_pacing import Decision, PendingAttempt, Route


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(compression, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


def test_live_wait_excludes_both_clocks_and_preserves_remaining_budget(clock):
    fence = compression.CompressionCommitFence(5)
    fence._idle_timeout_seconds = 2
    clock[0] += 1
    assert fence.quota_wait(Decision("hold", "weekly_limit"))
    with compression_deadline_scope(fence):
        clock[0] += 100
        assert fence.seconds_since_progress() == 1
        assert not fence.deadline_exceeded
        assert not fence.progress_observed
        assert aux._current_aux_stream_deadline() == 205
        assert fence.quota_wait(None)
        assert aux._current_aux_stream_deadline() - clock[0] == 4
    clock[0] += 1
    assert fence.seconds_since_progress() == 2
    assert fence.quota_wait(Decision("hold", "weekly_limit")) is False
    clock[0] += 3
    assert fence.deadline_exceeded
    assert fence.begin_commit() is False


@pytest.mark.parametrize("poison", ["expired", "cancelled", "revoked", "idle"])
def test_late_hold_never_revives_dead_fence(clock, poison):
    fence = compression.CompressionCommitFence(5)
    fence._idle_timeout_seconds = 2
    if poison == "expired":
        clock[0] += 5
    elif poison == "idle":
        clock[0] += 2
    elif poison == "cancelled":
        fence.cancel_before_commit()
    else:
        fence.revoke_commit_admission()
    before = fence.deadline_monotonic
    assert fence.quota_wait(Decision("hold", "weekly_limit")) is False
    clock[0] += 100
    assert fence.deadline_monotonic == before
    assert not fence.progress_observed


def test_host_poll_does_not_expire_while_verified_wait_is_live(clock):
    fence = compression.CompressionCommitFence(5)
    fence._idle_timeout_seconds = 2
    result = ([{"role": "user", "content": "summary"}], "prompt")

    class Future:
        polls = 0

        def result(self, timeout=None):
            self.polls += 1
            if self.polls == 1:
                assert fence.quota_wait(Decision("hold", "weekly_limit"))
                clock[0] += 100
                raise concurrent.futures.TimeoutError()
            assert not fence.deadline_exceeded
            assert fence.seconds_since_progress() == 0
            assert fence.quota_wait(None)
            return result

        def done(self):
            return False

    future = Future()
    settled, actual = compression._await_worker_within_budget(
        future, fence, idle=2, ceiling=5, wait_started=100)
    assert settled and actual is result
    assert future.polls == 2
    assert fence.deadline_monotonic == 205


@pytest.fixture
def summary_lane(monkeypatch, clock):
    state = SimpleNamespace(released=False, on_wait=None, sends=[], commits=[],
                            history=[{"role": "tool", "content": "completed action result"}],
                            fence=None, original=None)
    state.original = state.history
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))])

    def create(**request):
        assert state.released
        state.sends.append(request)
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                             base_url="https://chatgpt.com/backend-api/codex",
                             _hermes_aux_effective_provider="openai-codex")
    monkeypatch.setattr(pacing, "_section", lambda: {"enabled": True})
    monkeypatch.setattr(pacing, "_snapshot", lambda c, request, provider: pacing._Snapshot(
        provider, request["model"], "owned-fake", c.base_url))
    monkeypatch.setattr(aux, "bypass_chat_sdk_request_transform", lambda request, c: request)
    monkeypatch.setattr(aux, "_runtime_main_value", lambda name: None)
    monkeypatch.setattr(compression, "_retry_compression_on_fallback_chain",
                        lambda **kwargs: pytest.fail("a verified hold must not trigger fallback"))

    def bind(section, **selected):
        route = Route(selected["provider"], selected["model"], "stable-owned-account")

        def policy(route, hold, budget):
            return Decision("allow", "fresh") if state.released else Decision("hold", "weekly_limit")

        return SimpleNamespace(route=route, policy=policy)

    monkeypatch.setattr(native, "bind_request", bind)
    original_wait = PendingAttempt.wait

    def advance(seconds):
        assert state.history is state.original
        assert state.sends == [] and state.commits == []
        assert not state.fence.progress_observed
        clock[0] += 100
        assert not state.fence.deadline_exceeded
        assert state.fence.seconds_since_progress() == 0
        assert aux._current_aux_stream_deadline() > clock[0]
        if state.on_wait:
            state.on_wait()
        else:
            state.released = True

    monkeypatch.setattr(PendingAttempt, "wait", lambda self, policy: original_wait(
        self, policy, clock=lambda: clock[0], wait=advance))
    agent = SimpleNamespace(context_compressor=SimpleNamespace(), session_id="owned-session")

    def worker(fence):
        state.fence = fence

        def summarize(messages):
            # The real protected daemon boundary must carry the live fence hook.
            return aux._run_protected_sync_provider_call(
                lambda request: aux._create_with_progress_once(client, request, "compression"),
                {"model": "owned-summary", "messages": messages})

        compression._run_summary_dispatch(
            agent, state.history, summarize, {}, commit_fence=fence,
            attempt_generation=0, hard_cancel_event=None)
        if not fence.begin_commit():
            return state.history, "unchanged"
        try:
            state.history = [{"role": "user", "content": "committed summary"}]
            state.commits.append(state.history)
        finally:
            fence.finish_commit()
        return state.history, "prompt"

    state.worker = worker
    return state


def test_summary_releases_same_request_after_hold_beyond_idle_and_absolute(summary_lane, monkeypatch):
    state = summary_lane
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(compression, "_get_compress_timeout_executor", lambda: executor)
        messages, prompt = compression.run_compress_context_with_progress_timeout(
            worker=state.worker, messages=state.original, system_prompt_fallback="original prompt",
            idle_timeout_seconds=2, total_ceiling_seconds=5)
    assert messages is state.history
    assert prompt == "prompt"
    assert len(state.sends) == len(state.commits) == 1
    assert state.sends[0]["messages"] is state.original
    assert state.original == [{"role": "tool", "content": "completed action result"}]


@pytest.mark.parametrize("stop", ["cancel_before_commit", "revoke_commit_admission"])
def test_cancel_or_ownership_loss_during_wait_keeps_committed_original(summary_lane, monkeypatch, stop):
    state = summary_lane
    state.on_wait = lambda: getattr(state.fence, stop)()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(compression, "_get_compress_timeout_executor", lambda: executor)
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            compression.run_compress_context_with_progress_timeout(
                worker=state.worker, messages=state.original, system_prompt_fallback="original prompt",
                idle_timeout_seconds=2, total_ceiling_seconds=5)
    assert state.history is state.original
    assert state.sends == [] and state.commits == []
    assert not state.fence.begin_commit()


def test_stall_fallback_gets_only_remaining_post_quota_budget(clock, monkeypatch):
    fence = compression.CompressionCommitFence(5)
    clock[0] += 1
    assert fence.quota_wait(Decision("hold", "weekly_limit"))
    clock[0] += 100
    assert fence.quota_wait(None)
    # Real fallback entry: prove setup consumes rather than renews the shared budget.
    deadline = fence.deadline_monotonic
    assert deadline is not None
    observed = []
    from agent import context_compressor
    from contextlib import nullcontext
    monkeypatch.setattr(context_compressor, "pin_summary_route", lambda route: nullcontext())

    def run(**kwargs):
        observed.append(kwargs)
        return [{"role": "user", "content": "fallback summary"}], "prompt"

    monkeypatch.setattr(compression, "run_compress_context_with_progress_timeout", run)
    original = [{"role": "user", "content": "original"}]
    result = compression._run_pinned_compression_retry(
        {"provider": "fake", "model": "fallback", "label": "owned", "timeout": 10},
        worker=lambda fence: (original, "original prompt"), messages=original, system_prompt_fallback="original prompt",
        idle_timeout_seconds=2, total_ceiling_seconds=5, precommit_deadline_monotonic=deadline,
        on_commit_overrun=None, on_timeout_cause=None, telemetry_agent=None, new_fence=None)
    assert result is not None
    assert observed[0]["precommit_deadline_monotonic"] == 205
    assert observed[0]["total_ceiling_seconds"] == 4
    clock[0] = 205
    assert compression._run_pinned_compression_retry(
        {"provider": "fake", "model": "fallback", "label": "owned"},
        worker=lambda fence: pytest.fail("expired fallback ran"), messages=original,
        system_prompt_fallback="original prompt", idle_timeout_seconds=2, total_ceiling_seconds=5,
        precommit_deadline_monotonic=deadline, on_commit_overrun=None,
        on_timeout_cause=None, telemetry_agent=None, new_fence=None) is None
    assert len(observed) == 1
