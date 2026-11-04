"""Source-only fresh-capture regressions, prepared, not execution receipts.

Extract exact native function ASTs, not product modules or SDKs. SQLite schema,
profile routing, redaction, ACL/fsync and refused admission are narrow doubles;
capture, lease, reconciliation, holder/history checks and unlink are native code.
Run only through the separately released canonical runner and OS fence. These
cases do not test admission policy, gateway restart, inference or quota behavior.
"""
import ast
import asyncio
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
GOAL = 'Compare original audio; continue only unfinished work.'
BODY = b'owned audio fixture, no decoder or provider involved'
PATHS = ['retry', 'refused-write', 'webhook-preparation']


class Refused(Exception):
    pass


class OwnedDB:
    """Only the SQL surface the extracted media functions actually use."""

    def __init__(self, path):
        # The actual webhook leaf intentionally calls the collector off-loop.
        # Serialize this narrow double without bypassing that native thread hop.
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript('''
            CREATE TABLE session_admissions (
                principal_id TEXT, target_session_id TEXT, request_id TEXT,
                status TEXT, outcome TEXT, payload_json TEXT);
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT, active INTEGER,
                compacted INTEGER NOT NULL DEFAULT 0, role TEXT, content TEXT);
        ''')

    @contextmanager
    def _read_ctx(self):
        with self.lock:
            yield self.conn

    def _execute_write(self, fn):
        with self.lock, self.conn:
            return fn(self.conn)

    def row(self, request, refs, *, status='terminal', outcome='completed'):
        payload = {'text': GOAL, 'native_text_v1': {'media': refs}}
        with self.conn:
            self.conn.execute('INSERT INTO session_admissions VALUES (?,?,?,?,?,?)',
                              ('principal', 's', request, status, outcome, json.dumps(payload)))
        return payload

    def history(self):
        return self.conn.execute('SELECT * FROM messages ORDER BY id').fetchall()


def extract(relative, names, namespace):
    """Compile selected function/class bodies only, never module top-level code."""
    tree = ast.parse((ROOT / relative).read_text(), filename=relative)
    selected = []
    wanted = set(names)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in wanted:
                selected.append(node)
            elif isinstance(node, ast.ClassDef):
                selected.extend(child for child in node.body
                                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and child.name in wanted)
    assert {node.name for node in selected} == wanted
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative, 'exec'), namespace)


def fake_module(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def owned(tmp_path, monkeypatch):
    # Check runner isolation before this fixture writes anything.
    assert os.environ.get('HERMES_DISABLE_LAZY_INSTALLS') == '1'
    assert os.environ.get('HERMES_TEST_ISOLATION')
    isolation = Path(os.environ['HERMES_TEST_ISOLATION']).resolve()
    scratch = Path(os.environ['TMPDIR']).resolve()
    assert isolation == Path(os.environ['HERMES_HOME']).resolve()
    assert isolation.is_relative_to(scratch)
    assert tmp_path.resolve().is_relative_to(scratch)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    root = tmp_path / 'native-inputs'
    root.mkdir()
    db = OwnedDB(tmp_path / 'owned.sqlite')

    def validate(size, *, max_bytes):
        if size > max_bytes:
            raise ValueError('owned size limit')

    fake_module(monkeypatch, 'gateway.platforms.base',
                get_inbound_media_max_bytes=lambda: 1024 * 1024,
                validate_inbound_media_size=validate)
    fake_module(monkeypatch, 'agent.redact', redact_sensitive_text=lambda text: text)
    fake_module(monkeypatch, 'hermes_state_local_lineage',
                local_physical_target=lambda conn, sid: sid)
    fake_module(monkeypatch, 'hermes_state_runtime',
                _canonical_chain=lambda conn, sid: [sid])
    namespace = dict(Counter=Counter, contextmanager=contextmanager, ContextVar=ContextVar,
                     hashlib=hashlib, json=json, os=os, Path=Path, re=re, stat=stat,
                     tempfile=tempfile, threading=threading, RuntimeStoreError=Refused,
                     ADMITTED_REQUEST_CONFLICT={},
                     make_user_dir=lambda path, **kw: Path(path).mkdir(**kw),
                     fsync_directory=lambda path: None,
                     native_inputs_store=lambda: root, _media_root=lambda: root,
                     _leases=Counter(), _lease_lock=threading.Lock(),
                     _lease_scope=ContextVar('owned-media-capture', default=None),
                     _SHA256_NAME=re.compile(r'[0-9a-f]{64}'),
                     _API_IMAGE_NAME=re.compile(r'api_[0-9a-f]{32}\.(png|jpg|gif|webp)'),
                     _HOSTED_DOCUMENT_RE=re.compile(r'\n\[Shared attachment\] file: ([^\n]+)\n'))
    extract('gateway/session_ingress_media.py', [
        '_CaptureScope', 'capture_lease', 'drop_capture_lease', 'capture_native_media',
        '_capture_file', '_open_regular', 'restore_native_media', 'ingress_fingerprint',
        'reconcile_native_retry', 'hosted_document_references', '_failed_input_history',
        '_held_media_paths', '_file_identity', '_held_file_identities', '_history_mentions',
        '_collectable', 'release_unheld_media'], namespace)
    media = fake_module(monkeypatch, 'gateway.session_ingress_media', **namespace)
    gate_calls = []

    def refuse(db, **kwargs):
        gate_calls.append(kwargs['request_id'])
        raise Refused('owned-admission-refusal')

    native = {'admit_session_input': refuse}
    extract('gateway/session_authority.py', ['_admit_native_write'], native)
    webhook = {'asyncio': asyncio}
    extract('gateway/platforms/webhook_ingress.py', ['_release_preparation'], webhook)
    state = SimpleNamespace(root=tmp_path, db=db, media=media,
                            native=native['_admit_native_write'],
                            webhook=webhook['_release_preparation'], gate_calls=gate_calls,
                            authority=SimpleNamespace(db=db, epoch='owned',
                                                      _current_owner=lambda sid: sid))
    try:
        yield state
    finally:
        db.conn.close()


def capture(state, name):
    # Real native copying/content addressing; separate staging path on each call.
    directory = state.root / ('staging-' + str(len(list(state.root.glob('staging-*')))))
    directory.mkdir()
    staged = directory / name
    staged.write_bytes(BODY)
    return state.media.capture_native_media([staged])


def seed(state, *, outcome='completed', transcript=True, live=False):
    canonical = capture(state, 'original.ogg')
    state.db.row('retry-id', canonical)
    earlier = capture(state, 'incoming.ogg')
    state.db.row('earlier-context', earlier, outcome=outcome)
    if transcript:
        with state.db.conn:
            state.db.conn.executemany('INSERT INTO messages (id,session_id,active,role,content) VALUES (?,?,?,?,?)', [
                (1, 's', 1, 'assistant', GOAL + '\nRetained attachment: ' + earlier[0]['path']),
                (2, 's', 1, 'tool', 'receipt-effect-once')])
    if live:
        state.db.row('other-live', earlier, status='queued', outcome=None)
    return canonical, earlier


def cleanup(state, path, fresh):
    payload = {'text': GOAL, 'native_text_v1': {'media': fresh}}
    if path == 'retry':
        resolved = state.media.reconcile_native_retry(
            state.db, principal_id='principal', session_id='s', request_id='retry-id', payload=payload)
        assert Path(resolved['native_text_v1']['media'][0]['path']).name == 'original.ogg'
    elif path == 'refused-write':
        with pytest.raises(Refused, match='owned-admission-refusal'):
            state.native(state.authority, principal_id='principal', session_id='s',
                         request_id='new-refused-id', payload=payload, authorize=lambda conn: None)
    else:
        assert path == 'webhook-preparation'
        asyncio.run(state.webhook(state.authority, payload))


def survive(state, refs):
    assert Path(refs[0]['path']).exists(), 'fresh-capture cleanup erased prior retained context'
    assert hashlib.sha256(Path(refs[0]['path']).read_bytes()).hexdigest() == refs[0]['sha256']
    assert state.media.restore_native_media(refs) == [refs[0]['path']]


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('outcome', ['completed', 'failed', 'interrupted'])
def test_fresh_cleanup_preserves_prior_transcript_owner(owned, path, outcome):
    _, earlier = seed(owned, outcome=outcome)
    history = owned.db.history()
    if outcome != 'completed':
        # Real handoff-proof query: saved caption/refs mean the ledger no longer holds.
        assert owned.media._failed_input_history(owned.db.conn,
            {'target_session_id': 's', 'payload': {'text': GOAL}}, earlier) == ('s', True)
    assert earlier[0]['path'] not in owned.media._held_media_paths(owned.db.conn)
    with owned.media.capture_lease():
        fresh = capture(owned, 'incoming.ogg')
        assert fresh == earlier  # Same bytes AND basename reuse the exact prior file.
        if path == 'webhook-preparation':
            # _prepared exits its capture lease before _release_preparation is called.
            owned.media.drop_capture_lease()
        cleanup(owned, path, fresh)
    assert owned.db.history() == history  # Receipt/goal survive textually even on the bug.
    survive(owned, earlier)  # Predicted red for all three paths on current candidate.
    # Repeated cleanup must not rewrite history, replay a tool, or lose context.
    cleanup(owned, path, fresh)
    assert owned.db.history() == history
    survive(owned, earlier)
    assert owned.gate_calls == (['new-refused-id'] * 2 if path == 'refused-write' else [])


@pytest.mark.parametrize('path', PATHS)
def test_fresh_cleanup_still_collects_unowned_capture(owned, path):
    _, earlier = seed(owned, transcript=False)
    with owned.media.capture_lease():
        fresh = capture(owned, 'incoming.ogg')
        if path == 'webhook-preparation':
            owned.media.drop_capture_lease()
        cleanup(owned, path, fresh)
    assert not Path(earlier[0]['path']).exists()
    cleanup(owned, path, fresh)  # Missing-file cleanup is harmless.
    assert owned.db.history() == []


@pytest.mark.parametrize('path', PATHS)
def test_fresh_cleanup_keeps_other_live_admission_holder(owned, path):
    _, earlier = seed(owned, transcript=False, live=True)
    with owned.media.capture_lease():
        fresh = capture(owned, 'incoming.ogg')
        if path == 'webhook-preparation':
            owned.media.drop_capture_lease()
        cleanup(owned, path, fresh)
    survive(owned, earlier)


def test_retry_does_not_collect_its_exact_committed_path(owned):
    canonical, _ = seed(owned)
    with owned.media.capture_lease():
        fresh = capture(owned, 'original.ogg')
        assert fresh == canonical
        cleanup(owned, 'retry', fresh)
    survive(owned, canonical)
    assert owned.gate_calls == []
