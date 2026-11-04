"""Quota waits preserve native cron ownership, not semantic model progress.

Prepared for a contained canonical runner; no external worker/bootstrap or provider.
"""
from datetime import datetime, timezone, UTC
from types import SimpleNamespace
import threading

import pytest

from agent import session_activity
from cron import executions, scheduler, scheduler_liveness


@pytest.fixture
def clock(monkeypatch, tmp_path):
    clock = SimpleNamespace(now=1000.0)
    timer = SimpleNamespace(time=lambda: clock.now, monotonic=lambda: clock.now)
    for module in (session_activity, scheduler, scheduler_liveness):
        monkeypatch.setattr(module, "time", timer)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    monkeypatch.setattr(executions, "_hermes_now", lambda: datetime.fromtimestamp(clock.now, UTC))
    monkeypatch.setattr(executions, "_emit_execution_state", lambda *args, **kwargs: None)
    return clock


def test_watchdog_exempts_live_wait_then_detects_ordinary_idle(clock):
    agent = SimpleNamespace(_last_activity_ts=clock.now)
    session_activity.set_quota_wait(agent, SimpleNamespace(action="wait", reason="quota"))
    original_stamp = agent._last_activity_ts
    polls = []

    class Stop(threading.Event):
        def wait(self, timeout=None):
            clock.now += timeout or 0
            polls.append(clock.now)
            if len(polls) == 5:
                session_activity.set_quota_wait(agent, None)
            return len(polls) > 20

    assert scheduler_liveness._inactivity_watchdog_loop(
        get_idle_seconds=lambda: session_activity.quota_idle_seconds(agent, clock.now - original_stamp),
        limit_s=20, poll_s=10, stop=Stop(), future_done=lambda: False)
    assert len(polls) == 7  # held 50s, then 20s of actual idle
    assert agent._last_activity_ts == original_stamp


@pytest.mark.parametrize("action", ["wait", "blocked"], ids=["quota", "manual-repair"])
@pytest.mark.parametrize("lose_owner", [False, True], ids=["release", "ownership-loss"])
def test_native_monitor_keeps_same_fire_and_results(clock, monkeypatch, action, lose_owner):
    """Real worker, monitor, stamper and SQLite; foreign recovery only changes process identity.

    Fake the polling clock and inference wait, not persistence or the native cancellation seam.
    """
    started, release, exited, cancelled = (threading.Event() for _ in range(4))
    effects, sends, heartbeats, observed = [], [], [], []
    goal = "Publish the frozen brief once, then summarize its recorded receipt"
    job = {"id": "frozen-job", "prompt": goal, "schedule": {"kind": "once"},
           "run_claim": {"by": "original-claim"}}
    execution = executions.create_execution(job["id"], source="cron")
    executions.mark_execution_running(execution["id"])
    job["execution_id"] = execution["id"]
    before = executions.get_execution(execution["id"])
    assert before is not None
    owner_id = executions._PROCESS_ID
    monkeypatch.setattr(executions, "_owner_is_live", lambda *args: True)
    monkeypatch.setattr(executions, "_live_owner_stale_after_seconds", lambda: 20.0)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 10.0)
    monkeypatch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 20.0)
    monkeypatch.setattr(scheduler, "heartbeat_run_claim",
                        lambda jid, expected_owner: heartbeats.append((jid, expected_owner)))
    # The watchdog is separately driven above, so avoid an unsampled real 5s daemon here.
    real_thread = threading.Thread

    def thread(*args, **kwargs):
        if kwargs.get("name", "").startswith("cron-inactivity-"):
            return SimpleNamespace(start=lambda: None)
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(scheduler.threading, "Thread", thread)

    class Agent:
        session_id = "original-session"
        _last_activity_ts = 1000.0
        api_call_count = 0

        def get_activity_summary(self):
            return {"seconds_since_activity": clock.now - self._last_activity_ts}

        def run_conversation(self, prompt, **kwargs):
            try:
                assert prompt == goal
                effects.append("published-once")
                self.messages = [{"role": "user", "content": prompt},
                                 {"role": "tool", "tool_call_id": "published", "content": effects[0]}]
                session_activity.set_quota_wait(self, SimpleNamespace(action=action, reason="quota"))
                started.set()
                assert release.wait(10), "test did not release worker"
                if not cancelled.is_set():
                    session_activity.set_quota_wait(self, None)
                    sends.append(self.messages)
                    self.api_call_count += 1
                return {"final_response": "original completed", "messages": self.messages}
            finally:
                session_activity.set_quota_wait(self, None)
                exited.set()

        def interrupt(self, reason):
            cancelled.set()
            release.set()

    agent = Agent()
    native_wait = scheduler.concurrent.futures.wait

    def poll(futures, timeout=None):
        assert started.wait(5)
        clock.now += 10
        row = executions.get_execution(execution["id"])
        assert row is not None
        if row["progress_at"] is not None:
            # A different scheduler looks at the same live-owned row; it must not reclaim it.
            with monkeypatch.context() as foreign:
                foreign.setattr(executions, "_PROCESS_ID", "foreign-scheduler")
                assert executions.recover_interrupted_executions() == 0
            assert executions._PROCESS_ID == owner_id
        observed.append(row)
        assert agent.api_call_count == 0
        assert sends == []
        assert effects == ["published-once"]
        assert agent._last_activity_ts == 1000.0
        if len(observed) == 4:
            if lose_owner:
                cancelled.set()  # native fire-claim heartbeat's cancellation signal
                return set(), set(futures)
            release.set()
            return native_wait(futures, timeout=5)
        return set(), set(futures)

    monkeypatch.setattr(scheduler.concurrent.futures, "wait", poll)
    try:
        if lose_owner:
            with pytest.raises(RuntimeError, match="was stopped"):
                scheduler._run_agent_with_watchdog(
                    agent, goal, job, job["id"], job["id"], "task", cancelled)
        else:
            result = scheduler._run_agent_with_watchdog(
                agent, goal, job, job["id"], job["id"], "task", cancelled)
            assert result["messages"] is agent.messages
            assert sends == [agent.messages]
            assert sends[0][1]["content"] == "published-once"
        assert exited.wait(5)
        assert effects == ["published-once"]
        assert agent.api_call_count == (0 if lose_owner else 1)
        assert len(observed) == 4
        assert heartbeats and set(heartbeats) == {(job["id"], "original-claim")}
        after = executions.get_execution(execution["id"])
        assert after is not None
        assert after["status"] == "running"  # monitor does not book completion or create a new fire
        for key in ("id", "job_id", "process_id", "claimed_at", "started_at"):
            assert after[key] == before[key]
        assert after["progress_at"] is not None
        assert scheduler_liveness.owner_progress[execution["id"]] >= 1030
        assert job["prompt"] == goal and agent.session_id == "original-session"
    finally:
        release.set()
        assert exited.wait(5)
        scheduler_liveness.owner_progress.pop(execution["id"], None)


def test_confirmed_fire_claim_loss_sets_native_cancellation(monkeypatch):
    job = {"id": "frozen-job", "execution_id": "original-fire", "fire_claim": {"by": "original-owner"}}
    calls = []

    def heartbeat(job_id, *, expected_owner):
        calls.append((job_id, expected_owner))
        return len(calls) == 1  # startup owns it; both subsequent ownership probes fail

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", heartbeat)
    monkeypatch.setattr(scheduler, "self_removal_delivery_allowed", lambda job_id: False)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.001)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_MISS_CONFIRM_SECONDS", 0.001)
    entered = []

    def run(lost):
        entered.append(job["execution_id"])
        assert lost.wait(5), "confirmed ownership loss did not cancel the original run"
        return False

    assert scheduler._run_with_fire_claim_heartbeat(job, run) is False
    assert entered == ["original-fire"]
    assert calls == [("frozen-job", "original-owner")] * 3


def test_unmarked_idle_stops_stamping_and_foreign_owner_can_reclaim(clock, monkeypatch):
    row = executions.create_execution("wedged", source="cron")
    executions.mark_execution_running(row["id"])
    monkeypatch.setattr(executions, "_owner_is_live", lambda *args: True)
    monkeypatch.setattr(executions, "_live_owner_stale_after_seconds", lambda: 20.0)
    stamper = scheduler_liveness.ExecutionProgressStamper(
        row["id"], "wedged", idle_seconds=lambda: 100.0, every_seconds=10)
    clock.now += 100
    stamper.tick()
    current = executions.get_execution(row["id"])
    assert current is not None and current["progress_at"] is None
    monkeypatch.setattr(executions, "_PROCESS_ID", "foreign-scheduler")
    assert executions.recover_interrupted_executions() == 1
    current = executions.get_execution(row["id"])
    assert current is not None and current["status"] == "unknown"
    assert executions.recover_interrupted_executions() == 0
