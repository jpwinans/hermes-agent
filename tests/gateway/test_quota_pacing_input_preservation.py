"""Prepared C1/C2 regressions; execute only after the parent's OS fence release.

Real owned SQLite/capture/admission/history/settlement/collector; fake inference,
policy, environment preparation and event transport. No native boot or SDK client.
DB reopen below is NOT a clean gateway restart. No same-fire replay is invented.
"""
import asyncio
import base64
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import session_ingress_media as media
from gateway.session_results import admission_result, execution_result, finish_result
from gateway.session_settlement_recovery import commit_and_publish
from gateway.session_transcript import SessionTranscriptMixin, TranscriptReadError
from hermes_state import SessionDB
from hermes_state_runtime import (
    admit_session_input, begin_runtime_epoch, cancel_session_input, claim_session_input,
    get_session_admission, recover_session_inputs, resolve_unknown_session_input,
    settle_session_input,
)

GOAL = 'Compare the retained attachment, preserve receipt-effect-once, and summarize only the remaining work.'
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aO4sAAAAASUVORK5CYII=')


@pytest.fixture
def owned(tmp_path, monkeypatch):
    # Fail before any fixture write if someone bypasses the released runner/fence.
    assert os.environ.get('HERMES_DISABLE_LAZY_INSTALLS') == '1'
    # Native autouse isolation replaces the marker with its owned HERMES_HOME.
    assert os.environ.get('HERMES_TEST_ISOLATION')
    isolation = Path(os.environ['HERMES_TEST_ISOLATION']).resolve()
    assert isolation == Path(os.environ['HERMES_HOME']).resolve()
    assert isolation.is_relative_to(Path(os.environ['TMPDIR']).resolve())
    assert tmp_path.resolve().is_relative_to(Path(os.environ['TMPDIR']).resolve())
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(media, '_last_sweep', {})
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s', source='local')
    state = SimpleNamespace(db=db, epoch=begin_runtime_epoch(db, instance_id='owned'),
                            root=tmp_path, sid='s', number=0)
    yield state
    state.db.close()


def capture(state, name='voice.ogg', body=b'owned untranscribed audio'):
    from gateway.platforms.base import get_document_cache_dir
    staged = get_document_cache_dir() / name
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(body)
    refs = media.capture_native_media([staged])
    assert Path(refs[0]['path']).read_bytes() == body
    return refs


def admit(state, refs, *, text=GOAL, sid=None, kind='native_text_v1'):
    state.number += 1
    return admit_session_input(state.db, epoch=state.epoch, principal_id='owned-human',
        session_id=sid or state.sid, request_id=f'owned-{state.number}',
        payload={'text': text, kind: {'media': refs}})


def old(refs):
    stamp = time.time() - 26 * 3600
    for ref in refs:
        os.utime(ref['path'], (stamp, stamp))


def digest_survives(refs):
    for ref in refs:
        assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256']
        assert media.restore_native_media([ref]) == [ref['path']]


def force_sweep(state):
    media._last_sweep.clear()
    return media.sweep_settled_media(state.db)


def history(state):
    # Real messaging history loader/projection; only DB routing is injected.
    store = SimpleNamespace(_db_for_session_id=lambda sid: state.db, _follow_reroutes=lambda sid: sid)
    return SessionTranscriptMixin.load_transcript(store, state.sid)


def original_history(state, refs, *, image=False):
    note = GOAL + '\n' + ('[screenshot]\n[Image attached at: ' + refs[0]['path'] + ']' if image else
                           'Untranscribed attachment. It is saved at: ' + refs[0]['path'])
    state.db.append_message(state.sid, 'user', note)
    state.db.append_message(state.sid, 'assistant', tool_calls=[{
        'id': 'effect-once', 'type': 'function',
        'function': {'name': 'owned_effect', 'arguments': '{}'}}])
    state.db.append_message(state.sid, 'tool', 'receipt-effect-once', tool_call_id='effect-once')


def settle(state, row, *, failed=False, publish=False):
    captured = {'result': {'final_response': '', 'messages': [], 'completed': False,
                           'failed' if failed else 'interrupted': True}, 'usage': {}}
    if failed:
        captured['result']['error'] = 'Owned transcript read failure'
    if not publish:
        return finish_result(state.db, epoch=state.epoch, row=row, response='',
                             outcome='completed', result=captured)[0]
    return publish_captured(state, row, captured)


def publish_captured(state, row, captured):
    events = []
    live = SimpleNamespace(settling={}, controls=SimpleNamespace(snapshot=lambda *args: None),
        event_stream=SimpleNamespace(lock=threading.RLock(), publish=lambda sid, event: events.append(event)))
    authority = SimpleNamespace(db=state.db, epoch=state.epoch, sessions={state.sid: live},
                                steer_transfers={}, adopted={})
    result = {}
    commit_and_publish(authority, live, SimpleNamespace(session_id=state.sid), row,
                       '', 'completed', captured, result)
    assert events and events[-1]['admission_id'] == row['admission_id']
    assert events[-1]['status'] in ('interrupted', 'error')
    return result['settled']


@pytest.mark.parametrize('image', [False, True], ids=['untranscribed-audio', 'native-image'])
def test_old_transcript_owned_bytes_survive_native_settlement(owned, image):
    refs = capture(owned, 'photo.png', PNG) if image else capture(owned)
    row = admit(owned, refs)
    claim = claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid)
    assert claim['admission_id'] == row['admission_id']
    original_history(owned, refs, image=image)
    before = history(owned)
    old(refs)
    settle(owned, claim, publish=True)
    force_sweep(owned)
    digest_survives(refs)  # genuine baseline red: the old sweep ignores transcript owners
    after = history(owned)
    assert after == before
    assert GOAL in str(after) and 'receipt-effect-once' in str(after)
    if image:
        from agent.image_replay import replay_recent_images
        wire = deepcopy(after)
        assert replay_recent_images(wire, keep=3) == 1
        urls = [part['image_url']['url'] for message in wire if isinstance(message.get('content'), list)
                for part in message['content'] if part.get('type') == 'image_url']
        assert base64.b64decode(urls[0].split(',', 1)[1]) == PNG


@pytest.mark.asyncio
async def test_actual_prepare_history_failure_preserves_fresh_input_and_truthful_result(owned, monkeypatch):
    from gateway import run, run_turn_prepare as prepare
    from gateway.config import Platform
    refs = capture(owned, 'clip.mp4', b'owned video bytes, not a decoder fixture')
    row = admit(owned, refs)
    claim = claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid)
    entry = SimpleNamespace(session_id=owned.sid, session_key='owned-key', yolo=None)
    source = SimpleNamespace(platform=Platform.LOCAL)
    runner = SimpleNamespace(config=SimpleNamespace(),
        _hmwa_open_session=AsyncMock(return_value=(False, False)),
        _set_session_env=lambda context: (), _clear_session_env=lambda tokens: None,
        _hmwa_acquire_turn_lease=AsyncMock(), _mark_durable_active_turn=AsyncMock(),
        _hmwa_first_contact_notes=AsyncMock(side_effect=AssertionError('must fail before enrichment/agent')),
        async_session_store=SimpleNamespace(load_transcript=AsyncMock(side_effect=TranscriptReadError(owned.sid))))
    monkeypatch.setattr(prepare, 'build_session_context', lambda *args: {})
    monkeypatch.setattr(run, '_load_gateway_config', dict)
    monkeypatch.setattr('tools.approval_yolo.restore_gateway_yolo', lambda *args: None)
    captured = {}
    token = execution_result.set(captured)
    try:
        response, tokens = await prepare.GatewayTurnPrepareMixin._hmwa_prepare_turn(
            runner, SimpleNamespace(text=GOAL, internal=False), source, entry, 'owned-key', 'owned-key', 1)
    finally:
        execution_result.reset(token)
    assert isinstance(response, str) and tokens == ()
    runner._hmwa_first_contact_notes.assert_not_called()
    assert captured['result']['failed'] and not captured['result']['completed']
    assert history(owned) == []
    publish_captured(owned, claim, captured)
    digest_survives(refs)  # genuine fresh prehistory baseline red
    saved = admission_result(owned.db, row['admission_id'])
    assert saved['result']['failed'] and not saved['result']['completed']
    assert not saved['result']['messages']
    assert GOAL in str(history(owned)) and refs[0]['path'] in str(history(owned))
    assert claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid) is None
    media.release_admission_media(owned.db, row['admission_id'])
    assert len([m for m in history(owned) if 'Original user input:' in str(m.get('content'))]) == 1


@pytest.mark.parametrize('fault', ['append', 'result-metadata'])
def test_storage_failure_cannot_drop_last_input_owner(owned, fault):
    refs = capture(owned)
    row = admit(owned, refs)
    claim = claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid)
    if fault == 'result-metadata':
        owned.db._execute_write(lambda conn: conn.execute(
            "CREATE TRIGGER fail_result BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.admission.result.%' "
            "BEGIN SELECT RAISE(ABORT, 'owned result write failure'); END"))
        with pytest.raises(sqlite3.DatabaseError):
            settle(owned, claim, failed=True)
        assert get_session_admission(owned.db, admission_id=row['admission_id'])['status'] == 'started'
    else:
        settle(owned, claim, failed=True)
        owned.db._execute_write(lambda conn: conn.execute(
            "CREATE TRIGGER fail_history BEFORE INSERT ON messages "
            "BEGIN SELECT RAISE(ABORT, 'owned history write failure'); END"))
        with pytest.raises(sqlite3.DatabaseError):
            media.release_admission_media(owned.db, row['admission_id'])
        assert history(owned) == []
    old(refs)
    # Bypass cleanup's handoff attempt: even a collector called directly must hold.
    assert media.release_unheld_media(owned.db, refs) == 0
    force_sweep(owned)
    digest_survives(refs)
    if fault == 'append':
        owned.db._execute_write(lambda conn: conn.execute('DROP TRIGGER fail_history'))
        force_sweep(owned)  # retries history handoff, never retries the failed execution
        digest_survives(refs)
        assert GOAL in str(history(owned)) and refs[0]['path'] in str(history(owned))
        assert admission_result(owned.db, row['admission_id'])['result']['failed']
    # Faults and trigger removal above touch only this new isolated test DB.


def test_committed_handoff_then_raised_is_read_back_not_duplicated(owned, monkeypatch):
    refs = capture(owned)
    row = admit(owned, refs)
    claim = claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid)
    settle(owned, claim, failed=True)
    execute = owned.db._execute_write
    armed = True

    def committed_then_raised(callback, *args, **kwargs):
        nonlocal armed
        result = execute(callback, *args, **kwargs)
        if armed:
            armed = False
            raise OSError('owned commit acknowledgement lost')
        return result

    monkeypatch.setattr(owned.db, '_execute_write', committed_then_raised)
    with pytest.raises(OSError, match='acknowledgement'):
        media.release_admission_media(owned.db, row['admission_id'])
    first = history(owned)
    assert GOAL in str(first) and refs[0]['path'] in str(first)
    media.release_admission_media(owned.db, row['admission_id'])
    assert history(owned) == first
    old(refs)
    force_sweep(owned)
    digest_survives(refs)


def test_queued_unknown_and_explicit_queued_discard_are_distinct(owned):
    refs = capture(owned)
    running = admit(owned, refs)
    claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid)
    queued = admit(owned, refs)
    old(refs)
    owned.db.close()
    owned.db = SessionDB(owned.root / 'state.db')
    owned.epoch = begin_runtime_epoch(owned.db, instance_id='owned-reopened')
    assert recover_session_inputs(owned.db, epoch=owned.epoch) == 1
    unknown = get_session_admission(owned.db, admission_id=running['admission_id'])
    assert unknown['status'] == 'unknown'
    force_sweep(owned)
    assert media.release_unheld_media(owned.db, refs) == 0
    digest_survives(refs)
    cancel_session_input(owned.db, epoch=owned.epoch, admission_id=queued['admission_id'])
    media.release_admission_media(owned.db, queued['admission_id'])
    digest_survives(refs)  # unknown is still a holder, not a replayable turn
    resolve_unknown_session_input(owned.db, epoch=owned.epoch, admission_id=running['admission_id'],
                                  generation=unknown['generation'])
    media.release_admission_media(owned.db, running['admission_id'])
    digest_survives(refs)  # resolution is interrupted, NOT explicit input discard
    assert claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid) is None
    unrelated = capture(owned, 'unwanted.ogg', b'explicit queued discard')
    discard = admit(owned, unrelated)
    cancel_session_input(owned.db, epoch=owned.epoch, admission_id=discard['admission_id'])
    media.release_admission_media(owned.db, discard['admission_id'])
    assert not Path(unrelated[0]['path']).exists()


def test_unrelated_unreferenced_completed_media_still_collects(owned):
    refs = capture(owned)
    row = admit(owned, refs)
    claim = claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid)
    settle_session_input(owned.db, epoch=owned.epoch, admission_id=row['admission_id'],
                         generation=claim['generation'], outcome='completed')
    old(refs)
    assert force_sweep(owned) == 1
    assert not Path(refs[0]['path']).exists()


@pytest.mark.parametrize('image', [False, True], ids=['path-only', 'image-base64'])
def test_native_hold_stop_fake_repair_manual_continuation_does_not_replay_effect(owned, monkeypatch, image):
    from agent import quota_pacing_host as host, quota_pacing_native as native
    from agent.interrupt_control import InterruptControlMixin
    from agent.quota_pacing import Decision, PendingAttempt, Route
    refs = capture(owned, 'photo.png', PNG) if image else capture(owned)
    initial = admit(owned, refs)
    claim = claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid)
    original_history(owned, refs, image=image)
    effects = owned.root / 'effect-receipt.txt'
    effects.write_text('receipt-effect-once\n')  # the only completed local effect
    sends = []
    config = {'enabled': True, 'association': 'broken'}
    selected = host._Selected('fake-provider', 'fake-model', 'fake-key', 'https://invalid.example',
                              'chat_completions', 'https://invalid.example')
    monkeypatch.setattr(host, '_section', lambda: dict(config))
    monkeypatch.setattr(host, '_selected', lambda agent, request: selected)
    route = Route('fake-provider', 'fake-model', 'owned-account')
    monkeypatch.setattr(native, 'bind_request', lambda section, **kwargs: SimpleNamespace(
        route=route, policy=lambda *args: Decision('allow', 'repaired') if section['association'] == 'repaired'
        else Decision('hold', 'native_hold')))
    original_wait = PendingAttempt.wait
    clock = [0.0]
    waiting = []

    class Owner(InterruptControlMixin):
        def __init__(self):
            self.session_id = owned.sid
            self._interrupt_requested = False
            self._execution_thread_id = None
            self._active_children_lock = threading.Lock()
            self._active_children = []
            self.quiet_mode = True
            self._last_activity_ts = 100.0

    agent = Owner()

    def advance(seconds):
        clock[0] += 30 * 3600  # fake age/idle advance, not sleep
        old(refs)
        force_sweep(owned)
        assert media.release_unheld_media(owned.db, refs) == 0
        digest_survives(refs)
        assert agent._quota_wait_status == ('hold', 'native_hold')
        assert sends == [] and effects.read_text() == 'receipt-effect-once\n'
        waiting.append(True)
        assert agent.interrupt(hard_cancel=True)

    def wait(attempt, policy):
        return original_wait(attempt, policy, clock=lambda: clock[0], wait=advance)

    monkeypatch.setattr(PendingAttempt, 'wait', wait)
    first_request = {'model': 'fake-model', 'messages': history(owned)}
    if image:
        from agent.image_replay import replay_recent_images
        assert replay_recent_images(first_request['messages'], keep=3) == 1
    with pytest.raises(InterruptedError):
        host.paced_call(agent, first_request, sends.append)
    assert waiting and sends == []
    settle(owned, claim, publish=True)
    force_sweep(owned)
    digest_survives(refs)
    # Cold DB boundary only; do not label this a clean gateway restart.
    owned.db.close()
    owned.db = SessionDB(owned.root / 'state.db')
    owned.epoch = begin_runtime_epoch(owned.db, instance_id='manual-followup')
    assert recover_session_inputs(owned.db, epoch=owned.epoch) == 0
    assert claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid) is None
    assert admission_result(owned.db, initial['admission_id'])['result']['interrupted']
    config['association'] = 'repaired'  # fake config repair, not clearing a real account hold
    followup = admit(owned, [], text='Continue only the remaining summary; do not repeat the completed effect.')
    next_claim = claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid)
    assert next_claim['admission_id'] == followup['admission_id'] != initial['admission_id']
    next_request = {'model': 'fake-model', 'messages': history(owned) + [
        {'role': 'user', 'content': followup['payload']['text']}]}
    assert GOAL in str(next_request) and 'receipt-effect-once' in str(next_request)
    if image:
        from agent.image_replay import replay_recent_images
        assert replay_recent_images(next_request['messages'], keep=3) == 1
        urls = [p['image_url']['url'] for m in next_request['messages'] if isinstance(m.get('content'), list)
                for p in m['content'] if p.get('type') == 'image_url']
        assert base64.b64decode(urls[0].split(',', 1)[1]) == PNG
    else:
        assert refs[0]['path'] in str(next_request)
    host.paced_call(Owner(), next_request, sends.append)
    assert sends == [next_request] and effects.read_text() == 'receipt-effect-once\n'
    digest_survives(refs)


def test_capture_lease_and_physical_alias_holders_remain(owned):
    with media.capture_lease():
        refs = capture(owned)
        assert media.release_unheld_media(owned.db, refs) == 0
        digest_survives(refs)
    alias = Path(refs[0]['path']).with_name('alias.ogg')
    os.link(refs[0]['path'], alias)
    alias_ref = dict(refs[0], path=str(alias))
    holder = admit(owned, [alias_ref])
    assert media.release_unheld_media(owned.db, refs) == 0
    digest_survives(refs)
    cancel_session_input(owned.db, epoch=owned.epoch, admission_id=holder['admission_id'])
    assert media.release_unheld_media(owned.db, refs) == 1
    assert media.release_unheld_media(owned.db, [alias_ref]) == 1


def test_terminal_api_holder_and_foreign_profile_database_deny_collection(owned):
    refs = capture(owned, 'api_' + hashlib.sha256(PNG).hexdigest()[:32] + '.png', PNG)
    row = admit(owned, refs, kind='api_turn_v1')
    claim = claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid)
    settle_session_input(owned.db, epoch=owned.epoch, admission_id=row['admission_id'],
                         generation=claim['generation'], outcome='completed')
    old(refs)
    assert media.release_unheld_media(owned.db, refs) == 0
    foreign_root = owned.root / 'foreign-owned-profile'
    foreign_root.mkdir()
    with SessionDB(foreign_root / 'state.db') as foreign:
        assert media.collect_unheld_api_images(foreign) == 0
    digest_survives(refs)


def test_failed_input_release_policy_requires_history_or_explicit_session_discard(owned):
    refs = capture(owned)
    row = admit(owned, refs)
    claim = claim_session_input(owned.db, epoch=owned.epoch, session_id=owned.sid)
    settle(owned, claim, failed=True)
    # A different session containing the same text/path is not this input's handoff.
    owned.db.create_session('other', source='local')
    owned.db.append_message('other', 'assistant', GOAL + '\n' + refs[0]['path'])
    with owned.db._read_ctx() as conn:
        assert refs[0]['path'] in media._held_media_paths(conn)
    media.release_admission_media(owned.db, row['admission_id'])
    with owned.db._read_ctx() as conn:
        assert refs[0]['path'] not in media._held_media_paths(conn)
    digest_survives(refs)  # now history, not a permanent all-failed ledger hold
    # Native explicit deletion of BOTH owned sessions retires admissions/history;
    # it is the deliberate discard route, never stop/repair/continue.
    assert owned.db.delete_session('other')
    assert owned.db.delete_session(owned.sid)
    media.release_unheld_media(owned.db, refs)
    assert not Path(refs[0]['path']).exists()
