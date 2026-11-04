"""Restart uncertainty remains manual reconciliation, never automatic effect replay.

Note wording is a guidance contract, not proof that an LLM follows instructions.
Native SQLite recovery below proves only retained history/terminal interruption.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.run import build_resume_recovery_note


@pytest.mark.parametrize("reason", ["restart_timeout", "shutdown_timeout", None])
def test_unattended_legacy_note_requires_reconciliation_before_repeating(reason):
    note = build_resume_recovery_note(reason, interactive=False)
    assert "Do NOT re-run tool calls whose results already appear in the history" in note
    assert "Before repeating any unanswered potentially effectful call, reconcile" in note
    assert "absence of a recorded result is not proof that the effect did not run" in note
    assert "CONTINUE only safe or reconciled work" in note
    assert "manual reconciliation" in note
    assert "resume from the first step that has no recorded result" not in note
    assert "Any restart/shutdown command in the history has already run" in note


def test_interactive_restore_behavior_unchanged():
    note = build_resume_recovery_note("restart_timeout")
    assert "ask what they would like to do next" in note
    assert "Do NOT re-execute old tool calls — skip any unfinished work" in note
    assert "CONTINUE only safe" not in note


@pytest.mark.parametrize("interactive", [True, False])
def test_new_message_takes_priority_without_old_work_replay(interactive):
    message = "Do not publish anything else; show the preserved receipt."
    note = build_resume_recovery_note("restart_timeout", message, interactive=interactive)
    assert "Address the user's NEW message below FIRST" in note
    assert "Do NOT re-execute old tool calls — skip any unfinished work" in note
    assert note.endswith("\n\n" + message)
    assert "CONTINUE only safe" not in note


def test_native_cron_status_forwards_ownership_stamp_without_completion(tmp_path, monkeypatch):
    from cron.scheduler_liveness import owner_progress
    from gateway import session_authorities, session_cron
    from hermes_state import SessionDB
    from hermes_state_keys import CRON_PRINCIPAL_ID, cron_request_id
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    db = SessionDB(tmp_path / "state.db")
    execution_id = "original-fire"
    sid = "original-cron-session"
    db.create_session(sid, source="cron")
    epoch = begin_runtime_epoch(db, instance_id="owner")
    receipt = admit_session_input(
        db, epoch=epoch, principal_id=CRON_PRINCIPAL_ID, session_id=sid,
        request_id=cron_request_id("job", execution_id), payload={"text": "frozen goal"})
    claim_session_input(db, epoch=epoch, session_id=sid)
    # Owned authority seam only: no gateway registration, process, connection or service.
    authority = SimpleNamespace(db=db, profile_id="p", authorize=lambda *args: None)
    monkeypatch.setattr(session_authorities, "require_serving", lambda authority: None)
    owner_progress[execution_id] = 1234.0
    try:
        status = asyncio.run(session_cron.operation(
            authority, "status", {"session_id": sid, "admission_id": receipt["admission_id"]}))
        assert status == {"status": "started", "result": None, "job_flags": {}, "progress_at": 1234.0}
    finally:
        owner_progress.pop(execution_id, None)
        db.close()


def test_native_cron_execute_uses_frozen_launch_not_changed_job(tmp_path, monkeypatch):
    import json
    from contextlib import nullcontext
    from cron import executions, scheduler
    from gateway import run, session_cron
    from hermes_state_keys import CRON_PRINCIPAL_ID, cron_request_id

    frozen = {"id": "job", "prompt": "original frozen goal", "model": "original-model"}
    request_id = cron_request_id("job", "original-fire")
    policy = SimpleNamespace(request_json=json.dumps({
        "cron_job": frozen, "extra_prompt": "original input", "request_id": request_id}))
    row = {"payload": {"text": "original input"}, "request_id": request_id,
           "principal_id": CRON_PRINCIPAL_ID, "admission_id": "admission"}
    authority = SimpleNamespace(
        db=SimpleNamespace(db_path=str(tmp_path / "state.db")), pending_results={},
        sessions={"session": SimpleNamespace(source=SimpleNamespace(user_id=CRON_PRINCIPAL_ID))})
    calls = []

    def run_job(job, *, execution_id, extra_prompt, cancel_event):
        calls.append((job, execution_id, extra_prompt))
        assert session_cron.current_execution() == (authority, "session", "job", "original-fire")
        assert job == dict(frozen, execution_id="original-fire")
        return True, "saved output", "preserved receipt", ""

    monkeypatch.setattr(scheduler, "run_job", run_job)
    monkeypatch.setattr(executions, "get_execution", lambda execution_id: {})
    monkeypatch.setattr(run, "_profile_runtime_scope", lambda home: nullcontext())
    result = asyncio.run(session_cron.execute(authority, SimpleNamespace(session_id="session"), row, policy))
    assert result == "preserved receipt"
    assert len(calls) == 1 and calls[0][1:] == ("original-fire", "original input")
    saved = authority.pending_results["admission"]["result"]
    assert saved["completed"] is True and saved["cron_result"][2] == "preserved receipt"
    assert session_cron.current_execution() is None


def test_crashed_cron_keeps_goal_results_and_unanswered_effect_for_manual_reconciliation(tmp_path):
    from gateway.session_unattended_unknown import settle_recovered_unattended
    from hermes_state import SessionDB
    from hermes_state_keys import CRON_PRINCIPAL_ID, cron_request_id
    from hermes_state_runtime import (
        admit_session_input, begin_runtime_epoch, claim_session_input,
        get_session_admission, recover_session_inputs,
    )

    path = tmp_path / "state.db"
    goal = "Publish once, then summarize the preserved receipt"
    sid, effects = "frozen-cron-session", []
    db = SessionDB(path)
    try:
        db.create_session(sid, source="cron")
        epoch = begin_runtime_epoch(db, instance_id="first-owner")
        receipt = admit_session_input(
            db, epoch=epoch, principal_id=CRON_PRINCIPAL_ID, session_id=sid,
            request_id=cron_request_id("job", "fire"), payload={"text": goal})
        claim_session_input(db, epoch=epoch, session_id=sid)
        db.append_message(sid, "user", goal)
        db.append_message(sid, "assistant", tool_calls=[{
            "id": "recorded", "type": "function",
            "function": {"name": "publish", "arguments": "{}"}}])
        effects.append("published")
        db.append_message(sid, "tool", "published receipt", tool_call_id="recorded")
        db.append_message(sid, "assistant", tool_calls=[{
            "id": "unanswered", "type": "function",
            "function": {"name": "deliver", "arguments": "{}"}}])
        effects.append("delivered-without-result")  # crash window after effect, before result persistence
    finally:
        db.close()

    db = SessionDB(path)
    try:
        epoch = begin_runtime_epoch(db, instance_id="replacement-owner")
        assert recover_session_inputs(db, epoch=epoch) == 1
        row = get_session_admission(db, admission_id=receipt["admission_id"])
        assert row is not None and row["status"] == "unknown"
        assert settle_recovered_unattended(db, epoch) == 1
        row = get_session_admission(db, admission_id=receipt["admission_id"])
        assert row is not None
        assert row["status"] == "terminal" and row["outcome"] == "interrupted"
        assert row["payload"] == {"text": goal}
        assert claim_session_input(db, epoch=epoch, session_id=sid) is None
        assert settle_recovered_unattended(db, epoch) == 0
        messages = db.get_messages(sid)
        assert any(m["content"] == goal for m in messages)
        assert any(m["tool_call_id"] == "recorded" and m["content"] == "published receipt" for m in messages)
        assert any(m.get("tool_calls") and "unanswered" in str(m["tool_calls"]) for m in messages)
        assert not any(m["role"] == "tool" and m["tool_call_id"] == "unanswered" for m in messages)
        assert effects == ["published", "delivered-without-result"]
        # This is intentionally NOT a successful same-fire continuation; the frozen goal/history
        # support manual reconciliation. No scheduler resume/re-fire or synthetic agent is invoked.
    finally:
        db.close()
