"""M-F1 native regressions, prepared only; parent owns OS-fenced red/green execution.

Real newly owned SessionDB, capture, admissions/results, compaction, rewind,
turn leases, transcript loader and collectors. No AST doubles, inference, SDK
clients, credentials, native boot, scheduling, sleeps, or executed tool effects.
The receipt is a newly owned local fixture, not proof of provider/tool execution.
"""
import hashlib
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

CAPTION = 'Keep this untranscribed attachment; continue only the remaining summary.'
BODY = b'owned M-F1 untranscribed audio fixture'
HOLDER = 'owned-media-handoff-live-turn'  # unstructured: never a foreign-PID probe
CALL_ID = 'owned-effect-once'
RECEIPT = 'completed-owned-effect-receipt'


@pytest.fixture
def owned(tmp_path, monkeypatch):
    # Native conftest replaces the initial nonempty marker with its owned HOME.
    # Check ancestry, not equality to the runner's literal marker "1".
    assert os.environ.get('HERMES_DISABLE_LAZY_INSTALLS') == '1'
    assert os.environ.get('HERMES_TEST_ISOLATION')
    scratch = Path(os.environ['TMPDIR']).resolve()
    isolation = Path(os.environ['HERMES_TEST_ISOLATION']).resolve()
    assert isolation == Path(os.environ['HERMES_HOME']).resolve()
    assert isolation.is_relative_to(scratch)
    assert tmp_path.resolve().is_relative_to(scratch)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    # No product imports happen in this file before the owned-path assertions.
    from gateway import session_ingress_media as media
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    monkeypatch.setattr(media, '_last_sweep', {})
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s', source='local')
    state = SimpleNamespace(db=db, media=media, root=tmp_path, sid='s',
        epoch=begin_runtime_epoch(db, instance_id='owned-handoff'), now=time.time())
    try:
        yield state
    finally:
        db.close()


def failed_input(state, outcome):
    from gateway.platforms.base import get_document_cache_dir
    from gateway.session_results import finish_result, admission_result
    from hermes_state_runtime import admit_session_input, claim_session_input
    staged = get_document_cache_dir() / 'owned-handoff.ogg'
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(BODY)
    state.refs = state.media.capture_native_media([staged])
    row = admit_session_input(state.db, epoch=state.epoch, principal_id='owned-human',
        session_id=state.sid, request_id='owned-failed-input',
        payload={'text': CAPTION, 'native_text_v1': {'media': state.refs}})
    claim = claim_session_input(state.db, epoch=state.epoch, session_id=state.sid)
    assert isinstance(row, dict) and isinstance(claim, dict)
    assert claim['admission_id'] == row['admission_id']
    captured = {'result': {'final_response': '', 'messages': [], 'completed': False,
                          'failed' if outcome == 'failed' else 'interrupted': True}, 'usage': {}}
    state.row = finish_result(state.db, epoch=state.epoch, row=claim, response='',
        outcome='completed', result=captured)[0]
    assert state.row['status'] == 'terminal' and state.row['outcome'] == outcome
    state.saved_result = admission_result(state.db, row['admission_id'])
    assert state.saved_result['result'][outcome]
    assert not state.saved_result['result']['completed']
    # Ensure sweep's real age-eligible collector is exercised, not just its fresh-file branch.
    stamp = state.now - 30 * 3600
    for ref in state.refs:
        os.utime(ref['path'], (stamp, stamp))


def rows(state):
    with state.db._read_ctx() as conn:
        return [tuple(row) for row in conn.execute('SELECT * FROM messages ORDER BY id')]


def proof(state):
    with state.db._read_ctx() as conn:
        return state.media._failed_input_history(conn, state.row, state.refs)


def holds(state):
    with state.db._read_ctx() as conn:
        return state.refs[0]['path'] in state.media._held_media_paths(conn)


def intact(state):
    for ref in state.refs:
        assert Path(ref['path']).read_bytes() == BODY
        assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256']
        assert state.media.restore_native_media([ref]) == [ref['path']]
    from gateway.session_results import admission_result
    from hermes_state_runtime import get_session_admission
    assert admission_result(state.db, state.row['admission_id']) == state.saved_result
    assert get_session_admission(state.db, admission_id=state.row['admission_id']) == state.row


def sweep(state, step):
    # Advance the existing sweep eligibility only, NOT the actual lease wall clock.
    # No clearing the interval to pretend a blocked handoff retries immediately.
    return state.media.sweep_settled_media(state.db, now=state.now + step * 3601)


def compact(state, sid='s'):
    state.db.archive_and_compact(sid, [{'role': 'assistant', 'content': 'Owned summary; older input archived.'}])


def start_round(state):
    assert state.db.try_acquire_session_turn_lease(state.sid, HOLDER, ttl_seconds=3600)
    state.db.append_message(state.sid, 'user', 'Summarize only the remaining work.', turn_lease_holder=HOLDER)
    state.db.append_message(state.sid, 'assistant', tool_calls=[{
        'id': CALL_ID, 'type': 'function',
        'function': {'name': 'owned_effect', 'arguments': '{}'}}], turn_lease_holder=HOLDER)
    # The effect receipt already exists; its result has not yet flushed to history.
    state.effect = state.root / 'owned-effect-receipt.txt'
    state.effect.write_text(RECEIPT + '\n')


def finish_round(state):
    state.db.append_message(state.sid, 'tool', RECEIPT, tool_name='owned_effect',
        tool_call_id=CALL_ID, turn_lease_holder=HOLDER)
    state.db.release_session_turn_lease(state.sid, HOLDER)
    assert state.db.session_turn_lease_expires_at(state.sid, HOLDER) is None


def check_pair_and_receipt(state):
    from gateway.session_transcript import SessionTranscriptMixin
    from agent.agent_runtime_helpers import _pair_tool_calls_positionally
    store = SimpleNamespace(_db_for_session_id=lambda sid: state.db, _follow_reroutes=lambda sid: sid)
    history = SessionTranscriptMixin.load_transcript(store, state.sid)
    call = next(i for i, message in enumerate(history)
                if any(tc.get('id') == CALL_ID for tc in message.get('tool_calls') or []))
    assert history[call + 1]['role'] == 'tool'
    assert history[call + 1]['tool_call_id'] == CALL_ID
    assert history[call + 1]['content'] == RECEIPT
    paired = _pair_tool_calls_positionally(history)
    assert paired == history
    assert len([m for m in paired if m.get('role') == 'tool' and m.get('content') == RECEIPT]) == 1
    assert not any('Result unavailable' in str(m.get('content')) for m in paired)
    assert state.effect.read_text() == RECEIPT + '\n'
    intact(state)


@pytest.mark.parametrize('outcome', ['failed', 'interrupted'])
def test_compacted_failed_input_sweep_cannot_splice_live_tool_round(owned, outcome):
    failed_input(owned, outcome)
    original = owned.db.append_message(owned.sid, 'user', CAPTION + '\n' + owned.refs[0]['path'])
    compact(owned)
    with owned.db._read_ctx() as conn:
        assert tuple(conn.execute('SELECT active, compacted FROM messages WHERE id=?', (original,)).fetchone()) == (0, 1)
    start_round(owned)
    before = rows(owned)
    assert sweep(owned, 0) == 0
    assert rows(owned) == before  # predicted current-source RED: background assistant insertion
    assert proof(owned) == (owned.sid, True)  # also requires archived caption AND path proof
    assert not holds(owned)
    assert owned.db.session_turn_lease_expires_at(owned.sid, HOLDER) is not None
    intact(owned)
    finish_round(owned)
    completed = rows(owned)
    assert sweep(owned, 1) == 0
    assert rows(owned) == completed  # compaction is durable evidence, not postponed re-insertion
    check_pair_and_receipt(owned)


@pytest.mark.parametrize('outcome', ['failed', 'interrupted'])
def test_missing_handoff_waits_for_lease_then_existing_sweep_makes_progress(owned, outcome):
    failed_input(owned, outcome)
    assert proof(owned) == (owned.sid, False) and holds(owned)
    start_round(owned)
    before = rows(owned)
    assert sweep(owned, 0) == 0
    assert rows(owned) == before  # predicted current-source RED, independent of compaction fix
    assert proof(owned) == (owned.sid, False) and holds(owned)
    assert owned.media.release_unheld_media(owned.db, owned.refs) == 0
    intact(owned)
    # A same-interval sweep is not a new retry framework.
    assert owned.media.sweep_settled_media(owned.db, now=owned.now + 1) == 0
    assert rows(owned) == before
    finish_round(owned)
    completed = rows(owned)
    assert sweep(owned, 1) == 0
    assert proof(owned) == (owned.sid, True) and not holds(owned)
    assert len(rows(owned)) == len(completed) + 1
    check_pair_and_receipt(owned)
    after = rows(owned)
    assert sweep(owned, 2) == 0
    assert rows(owned) == after  # exactly one non-executable handoff; no admission replay
    from hermes_state_runtime import claim_session_input
    assert claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid) is None


@pytest.mark.parametrize('evidence', [
    'active', 'caption-compacted', 'path-compacted', 'both-compacted', 'rewound', 'foreign-compacted',
])
def test_handoff_proof_counts_compaction_but_not_rewind_or_other_session(owned, evidence):
    failed_input(owned, 'failed')
    target = owned.sid
    if evidence == 'foreign-compacted':
        target = 'other'
        owned.db.create_session(target, source='local')
    first = owned.db.append_message(target, 'user', CAPTION)
    if evidence == 'caption-compacted':
        compact(owned, target)
    owned.db.append_message(target, 'user', 'Retained path: ' + owned.refs[0]['path'])
    if evidence == 'path-compacted':
        # Archive both first, then restore ONLY caption as an active row.
        compact(owned, target)
        owned.db.append_message(target, 'user', CAPTION)
    elif evidence in ('both-compacted', 'foreign-compacted'):
        compact(owned, target)
    elif evidence == 'rewound':
        owned.db.rewind_to_message(target, first)
        with owned.db._read_ctx() as conn:
            flags = [tuple(r) for r in conn.execute('SELECT active, compacted FROM messages WHERE session_id=?', (target,))]
        assert flags and all(flag == (0, 0) for flag in flags)
    expected = evidence not in ('rewound', 'foreign-compacted')
    assert proof(owned) == (owned.sid, expected)  # three compacted variants predicted RED
    assert holds(owned) is not expected
    # Unfiltered history remains a conservative byte owner, even for rewind/foreign rows.
    assert owned.media.release_unheld_media(owned.db, owned.refs) == 0
    intact(owned)
    if not expected:
        assert sweep(owned, 0) == 0
        assert proof(owned) == (owned.sid, True)
        assert not holds(owned)
        intact(owned)


def test_settlement_handoff_keeps_existing_prompt_availability_under_own_lease(owned):
    failed_input(owned, 'failed')
    assert owned.db.try_acquire_session_turn_lease(owned.sid, HOLDER, ttl_seconds=3600)
    # Not a background sweep: preserve the existing settlement call's default behavior.
    assert owned.media.release_admission_media(owned.db, owned.row['admission_id']) == 0
    assert proof(owned) == (owned.sid, True)
    assert owned.db.session_turn_lease_expires_at(owned.sid, HOLDER) is not None
    assert len(rows(owned)) == 1
    intact(owned)
    before = rows(owned)
    owned.media.release_admission_media(owned.db, owned.row['admission_id'])
    assert rows(owned) == before
    owned.db.release_session_turn_lease(owned.sid, HOLDER)
