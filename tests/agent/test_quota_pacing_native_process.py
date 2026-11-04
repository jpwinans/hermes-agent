"""C3/C4 proposed real-owned-child proofs. NOT approved for execution yet.

Run only in the parent's separately released OS-contained canonical lane.
No real native programs/stores/GETs; the driver arms its own 30s alarm before
stdin/importing the adapter and never spawns. This is not descendant containment.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from agent import quota_pacing_native as adapter
from tests.waiting import async_wait_until

pytestmark = pytest.mark.platforms('posix')
REAL_POPEN = subprocess.Popen
FAKE_CREDENTIAL = 'owned-process-selected-fake-bearer'
DRIVER = Path(__file__).resolve().parents[1] / 'fixtures' / 'quota_pacing_owned_child.py'


def usage(lead):
    return {'rate_limit': {'primary_window': {
        'used_percent': 50, 'limit_window_seconds': 604800,
        'reset_at': time.time() + (5040 + lead * 60) * 60}}}


@pytest.fixture
def owned(tmp_path, monkeypatch):
    # The OS sandbox remains mandatory; these checks do not substitute for it.
    assert os.environ.get('HERMES_DISABLE_LAZY_INSTALLS') == '1'
    # Canonical conftest uses the isolated home path as its active marker.
    assert os.environ.get('HERMES_TEST_ISOLATION') in ('1', os.environ['HERMES_HOME'])
    assert Path(os.environ['HERMES_HOME']).resolve().is_relative_to(tmp_path.resolve())
    scratch = Path(os.environ['TMPDIR']).resolve()
    assert tmp_path.resolve().is_relative_to(scratch)
    state = tmp_path / 'state'
    section = {'enabled': True, 'accounts': [{
        'provider': 'openai-codex', 'account': 'owned-account',
        'credential_sha256': hashlib.sha256(FAKE_CREDENTIAL.encode()).hexdigest(),
        'state_dir': str(state)}]}
    bound = adapter.bind_request(section, provider='openai-codex', model='gpt-owned',
        credential=FAKE_CREDENTIAL, base_url='https://chatgpt.com/backend-api/codex',
        session_id='owned-session')
    lane = SimpleNamespace(binding=bound, state=state, children=[], records=[],
        mode='quota', lead=9, cleanup_entered=threading.Event(),
        cleanup_release=threading.Event(), supervisor_joined=threading.Event(),
        hold_cleanup=False)

    def spawn(argv, **kwargs):
        # Runtime contract, not source-shape inspection; reject any other launch.
        assert argv == [sys.executable, '-I', '-B', str(Path(adapter.__file__).absolute())]
        assert kwargs == dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        assert FAKE_CREDENTIAL not in repr(argv)
        root = tmp_path / ('child-' + str(len(lane.children)))
        root.mkdir()
        (root / 'plan.json').write_text(json.dumps({
            'credential_sha256': hashlib.sha256(FAKE_CREDENTIAL.encode()).hexdigest(),
            'usage': usage(lane.lead)}))
        fixed_argv = [sys.executable, '-I', '-B', str(DRIVER),
                      str(Path(adapter.__file__).absolute()), str(root), lane.mode]
        assert FAKE_CREDENTIAL not in repr(fixed_argv)
        # Keep conftest's parent marker untouched; the stdlib-only owned driver
        # uses the contract's literal boundary flags and inherits the OS fence.
        child_env = dict(os.environ)
        child_env.update(HERMES_TEST_ISOLATION='1', HERMES_DISABLE_LAZY_INSTALLS='1')
        assert child_env['HERMES_TEST_ISOLATION'] == '1'
        assert child_env['HERMES_DISABLE_LAZY_INSTALLS'] == '1'
        child = REAL_POPEN(fixed_argv, env=child_env, **kwargs)
        lane.children.append(child)  # Register ownership before any instrumentation.
        (root / 'spawn.json').write_text(json.dumps({'pid': child.pid, 'driver': str(DRIVER)}))
        record: dict[str, Any] = dict(pid=child.pid, root=root, argv=argv, fixed_argv=fixed_argv,
                      killed=[], waited=[], stdin_selected=False, output=b'')
        lane.records.append(record)
        communicate, kill, wait = child.communicate, child.kill, child.wait

        def observed_wait(*args, **kw):
            result = wait(*args, **kw)
            record['waited'].append((child.pid, result))
            return result

        def observed_kill():
            record['killed'].append(child.pid)
            return kill()

        def observed_communicate(payload=None, timeout=None):
            if payload is not None:
                request = json.loads(payload)
                record['stdin_selected'] = request['credential'] == FAKE_CREDENTIAL
            if timeout is None and lane.hold_cleanup:
                lane.cleanup_entered.set()
                assert lane.cleanup_release.wait(10), 'owned cleanup barrier not released'
            output = communicate(payload, timeout=timeout)
            record['output'] = output[0]
            if timeout is None:
                lane.supervisor_joined.set()
            return output

        child.wait = observed_wait
        child.kill = observed_kill
        child.communicate = observed_communicate
        return child

    # Do not mutate Python's shared subprocess module or other owners' Popen calls.
    monkeypatch.setattr(adapter, 'subprocess', SimpleNamespace(
        Popen=spawn, PIPE=subprocess.PIPE, DEVNULL=subprocess.DEVNULL,
        TimeoutExpired=subprocess.TimeoutExpired))
    try:
        yield lane
    finally:
        lane.cleanup_release.set()
        for child in lane.children:
            if child.poll() is None:
                child.kill()  # Only exact Popen handles created above; no pid search.
            child.communicate(timeout=5)
            for stream in (child.stdin, child.stdout):
                if stream is not None:
                    stream.close()


def observed(lane):
    if not lane.records:
        return False
    record = lane.records[-1]
    path = record['root'] / 'observed.json'
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:  # Writer can be observed between open and close.
        return False
    assert data == dict(pid=record['pid'], selected_bearer_matched=True,
                        isolation='1', lazy_installs='1')
    return True


def assert_reaped(lane, killed=False):
    # waitpid(ECHILD) distinguishes reaping from a merely dead/zombie child.
    for child, record in zip(lane.children, lane.records, strict=True):
        assert child.returncode is not None
        assert record['waited'] and all(pid == child.pid for pid, _ in record['waited'])
        if killed:
            assert record['killed'] == [child.pid]
            assert child.returncode == -signal.SIGKILL
        else:
            assert not record['killed']
            assert child.returncode == 0
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)


def test_real_stdin_native_hold_then_same_account_release(owned, caplog, capsys):
    b = owned.binding
    first = b.policy(b.route, None, 10)
    assert (first.action, first.reason) == ('hold', 'native_hold')
    assert first.active_hold
    assert observed(owned)
    persisted = json.loads((owned.state / 'hold.json').read_text())
    assert persisted
    # Expire only the owned fake cache's refresh timestamp; native thresholds unchanged.
    path = owned.state / 'limits.json'
    data = json.loads(path.read_text())
    data['checked_at'] = 0
    path.write_text(json.dumps(data))
    owned.lead = 3
    released = b.policy(b.route, first.active_hold, 10)
    assert (released.action, released.reason, released.active_hold) == ('allow', 'native_allow', {})
    assert json.loads((owned.state / 'hold.json').read_text()) == {}
    assert b.route.account == 'owned-account'
    assert observed(owned)
    assert all(r['stdin_selected'] for r in owned.records)
    assert all(FAKE_CREDENTIAL.encode() not in r['output'] for r in owned.records)
    assert FAKE_CREDENTIAL not in repr(b) + repr(first) + repr(released)
    captured = capsys.readouterr()
    assert FAKE_CREDENTIAL not in captured.out + captured.err + caplog.text
    assert_reaped(owned)


def test_real_deadline_kills_and_reaps_exact_child(owned):
    owned.mode = 'hang'
    held = {'reset': 123}
    b = owned.binding
    result = b.policy(b.route, held, 5)
    assert observed(owned), 'child must have reached injected GET, not just failed startup'
    assert (result.action, result.reason) == ('blocked', 'native_deadline')
    assert result.active_hold is held
    assert_reaped(owned, killed=True)


def test_real_sync_cancellation_kills_and_reaps_exact_child(owned):
    owned.mode = 'hang'
    b = owned.binding
    b._cancelled = lambda: bool(observed(owned))
    held = {'reset': 123}
    result = b.policy(b.route, held, 10)
    assert (result.action, result.reason) == ('blocked', 'cancelled')
    assert result.active_hold is held
    assert_reaped(owned, killed=True)


def test_real_async_deadline_yields_and_joins(owned):
    owned.mode = 'hang'
    b = owned.binding

    async def exercise():
        task = asyncio.create_task(b.async_policy(b.route, None, 5))
        await async_wait_until(lambda: observed(owned), timeout=10, what='owned child GET')
        # This callback must run while the policy is still pending on the child.
        pulse = asyncio.Event()
        asyncio.get_running_loop().call_soon(pulse.set)
        await asyncio.wait_for(pulse.wait(), 2)
        assert not task.done()
        result = await asyncio.wait_for(task, 10)
        assert result.reason == 'native_deadline'
        assert owned.supervisor_joined.is_set()
        assert_reaped(owned, killed=True)

    asyncio.run(exercise())


def test_real_async_cancellation_joins_supervisor_before_return(owned):
    owned.mode = 'hang'
    b = owned.binding

    async def exercise():
        task = asyncio.create_task(b.async_policy(b.route, None, 20))
        try:
            await async_wait_until(lambda: observed(owned), timeout=10, what='owned child GET')
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
            assert owned.supervisor_joined.is_set()
            assert_reaped(owned, killed=True)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_repeated_async_cancellation_cannot_escape_pending_join(owned):
    """Proposed strict regression, not xfail: may expose a frozen-adapter gap.

    Delay only the real child's final communicate/join after kill; every kill,
    wait and communication still uses the real Popen. No fake supervisor.
    """
    owned.mode = 'hang'
    owned.hold_cleanup = True
    b = owned.binding

    async def exercise():
        task = asyncio.create_task(b.async_policy(b.route, None, 20))
        try:
            await async_wait_until(lambda: observed(owned), timeout=10, what='owned child GET')
            task.cancel()
            await async_wait_until(owned.cleanup_entered.is_set, timeout=10,
                                   what='supervisor final real communicate')
            assert not task.done()
            task.cancel()  # Second cancellation while the actual supervisor is joining.
            # FIFO callbacks allow cancellation delivery, without a clock-based race.
            turn = asyncio.Event()
            asyncio.get_running_loop().call_soon(turn.set)
            await asyncio.wait_for(turn.wait(), 2)
            assert not task.done(), 'repeated cancellation escaped the supervisor join'
            owned.cleanup_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
            assert owned.supervisor_joined.is_set()
            assert_reaped(owned, killed=True)
        finally:
            owned.cleanup_release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # Even a red proposed assertion may not leave our supervisor/child behind.
            joined = await asyncio.to_thread(owned.supervisor_joined.wait, 10)
            assert joined

    asyncio.run(exercise())
