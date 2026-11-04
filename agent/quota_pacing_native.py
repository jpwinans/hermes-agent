"""Explicit account binding and bounded native policy reads (criteria 3 and 4).

No auth discovery: the operator associates a selected bearer fingerprint with a
stable account label and the SAME directories used by that account's native gate.
Native code is pinned, restricted to policy/cache functions, and runs in one
short-lived child. No inference, CLI, Keychain, refresh or credential writes.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

# Trusted installed programs, not configuration-supplied executable paths.
_SOURCES = {
    'claude': ('/Users/jameswinans/.claude/hooks/pace.py',
               'd5cfdae2bfaf81d3a8f7835ded2a20db229ed1df27dfc4875aaf27cb0a7efc9c'),
    'codex': ('/Users/jameswinans/Dropbox/Development/codex-pacing/pace.py',
              '9b5fbd1b4af21f77d904f649d9539181c03684a5e23987b673cf40945da33792'),
    'usage': ('/Users/jameswinans/Dropbox/Development/codex-pacing/claude_usage.py',
              '07c79e2c19ad8dced37fdbc60a355b261528c4c1022eaad7e1a9fa1bf0ec57e5'),
}
_ENDPOINTS = {'anthropic': {'https://api.anthropic.com', 'https://api.anthropic.com/v1'},
              'openai-codex': {'https://chatgpt.com/backend-api/codex'}}
_FUNCTIONS = {
    'claude': 'is_window label applies live_windows session_pace evaluate _evaluate read_json write_json read_limits iso_epoch stale_pause',
    'codex': 'normalize read_cache atomic_json snapshot ideal evaluate weekly_pause stale_pause bucket_for',
    'usage': 'normalize as_limits retry_after_sec _short problem snapshot',
}
_CONSTANTS = {'HARD_PCT', 'WEEKLY_LEAD_MIN', 'WEEKLY_LEAD_RESUME', 'WEEKLY_MIN',
              'POLL_SEC', 'REFRESH_SEC', 'MAX_AGE_SEC', 'SCHEMA',
              'WEEKLY_LEAD_TRIGGER_HOURS', 'WEEKLY_LEAD_RESUME_HOURS'}


def _native(kind, directory):
    """Load only audited policy/cache definitions; exclude all auth/CLI code."""
    import email.utils
    import fcntl
    import os
    import tempfile
    import urllib.request
    import urllib.error
    from datetime import datetime, timezone

    path, pin = _SOURCES[kind]
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError('native_source_changed')
    tree = ast.parse(raw, filename=path)
    functions = set(_FUNCTIONS[kind].split())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            names = {n.id for target in node.targets for n in ast.walk(target)
                     if isinstance(n, ast.Name)}
            if names and names <= _CONSTANTS:
                nodes.append(node)
    namespace = dict(json=json, math=math, time=time, Path=Path, os=os,
                     fcntl=fcntl, tempfile=tempfile, datetime=datetime,
                     timezone=timezone, email=email, urllib=urllib,
                     CACHE_DIR=directory, HOLD=str(directory / 'hold.json'),
                     LIMITS=str(directory / 'limits.json'), USAGE_DIR=str(directory),
                     SESSION_PACE=str(directory / 'session-pace.json'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)
    return SimpleNamespace(**namespace)


_PROVIDERS = {'openai-codex': 'OpenAI Codex', 'anthropic': 'Anthropic Claude'}


def _note(provider, pause, now, label=''):
    """One line for a real pause, from the native (resume_at, kind, detail). Y is the native estimate;
    the release itself is judged live, so the wording says about."""
    resume_at, kind, detail = pause
    who = label or _PROVIDERS.get(provider, provider)
    minutes = max(1, math.ceil((resume_at - now) / 60))
    span = f'about {minutes} minutes' + (f' ({minutes / 60:.1f} hours)' if minutes >= 120 else '')
    if kind == 'credits-floor':
        return f'Pacing for {who}: {detail}. Pausing for {span} until they renew, then continuing automatically.'
    if kind.endswith('-unverified'):
        return (f'Pacing for {who} is holding until a fresh usage reading shows it back on pace. '
                'Checking again shortly, then continuing automatically.')
    lead = re.search(r'([0-9]+(?:\.[0-9]+)?)h ahead', detail)
    if lead:
        return f'Pacing for {who} is +{lead.group(1)} hours over the pace. Pausing for {span}, then continuing automatically.'
    used = re.search(r'^(.*?) ([0-9]+(?:\.[0-9]+)?)%', detail)
    if used:
        return (f'Pacing for {who}: {used.group(1)} usage is {used.group(2)}%, over the limit. '
                f'Pausing for {span} until it resets, then continuing automatically.')
    return f'Pacing for {who}: {detail}. Pausing for {span}, then continuing automatically.'


def _cycle(renews_day, now):
    """[start, end) of the monthly billing cycle renewing on renews_day, at local midnight."""
    import calendar
    from datetime import datetime

    def on(year, month):
        return datetime(year, month, min(renews_day, calendar.monthrange(year, month)[1])).timestamp()

    year, month = datetime.fromtimestamp(now).year, datetime.fromtimestamp(now).month
    if on(year, month) > now:
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    following = (year + 1, 1) if month == 12 else (year, month + 1)
    return on(year, month), on(*following)


def _credits_spent(state_dbs, start):
    """Dollars this cycle on the Anthropic API, from Hermes's own per-model usage rows (main turns and helper
    calls alike). A row is counted whole if it was last used this cycle."""
    # ponytail: a session running across renewal counts entirely in the new cycle (overcounts, protecting the floor).
    import sqlite3
    total = 0.0
    for path in state_dbs:
        uri = Path(path).resolve(strict=True).as_uri() + '?mode=ro'
        with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=2)) as db:
            total += db.execute(
                # Hermes stores actual_cost_usd as 0.0, not NULL, when only an estimate exists: take the larger.
                "SELECT coalesce(sum(max(coalesce(actual_cost_usd, 0), coalesce(estimated_cost_usd, 0))), 0) "
                "FROM session_model_usage "
                "WHERE billing_provider = 'anthropic' AND billing_base_url LIKE 'https://api.anthropic.com%' "
                "AND last_seen >= ?", (start,)).fetchone()[0]
    return float(total)


def _evaluate_credits(request, now=None):
    """Pace a monthly API-credit grant: hold when spend runs ahead of an even burn of (grant - floor), keep the
    account-wide latch until back to the resume point, and stop at the floor until renewal."""
    import fcntl
    now = time.time() if now is None else now
    config = request['credits']
    start, end = _cycle(int(config['renews_day']), now)
    budget = float(config['monthly_usd']) - float(config['floor_usd'])
    spent = _credits_spent(config['state_dbs'], start)
    lead = (spent - budget * (now - start) / (end - start)) / budget * (end - start) / 3600
    directory = Path(request['state_dir'])
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / 'credits-hold.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / 'credits-hold.json'
        try:
            state = json.loads(path.read_text())
        except FileNotFoundError:
            state = {}
        held = state.get('cycle') == start or (request.get('active_hold') or {}).get('cycle') == start
        pause = None
        if spent >= budget:
            pause = (end, 'credits-floor', f"${spent:.2f} of the ${budget:.0f} budget spent this cycle, "
                                           f"keeping ${float(config['floor_usd']):.0f} in reserve")
        elif lead >= float(config['trigger_hours']) or (held and lead > float(config['resume_hours'])):
            held = True
            pause = (now + (lead - float(config['resume_hours'])) * 3600, 'credits-lead',
                     f'API credits {lead:.1f}h ahead of pace')
        else:
            held = False
        latch = {'cycle': start} if held else {}
        if latch != state:
            path.write_text(json.dumps(latch))
    result = {'action': 'hold' if pause else 'allow', 'reason': 'native_hold' if pause else 'native_allow',
              'retry_after': 30, 'active_hold': latch}
    if pause:
        result['note'] = _note(request['provider'], pause, now, request.get('label', ''))
    return result


def _codex_shape(raw):
    """HTTP quota shape only; Spark never borrows the ordinary quota bucket."""
    buckets = {}
    rows = [('codex', raw.get('rate_limit') or {})]
    additional = raw.get('additional_rate_limits', [])
    if additional is None:
        additional = []
    rows += [(r.get('limit_id'), r.get('rate_limit') or {})
             for r in additional if isinstance(r, dict)]
    for key, limits in rows:
        if not isinstance(key, str) or not key:
            continue
        bucket = {}
        for source, target in [('primary_window', 'primary'), ('secondary_window', 'secondary')]:
            window = limits.get(source)
            if isinstance(window, dict):
                seconds = window.get('limit_window_seconds')
                bucket[target] = dict(usedPercent=window.get('used_percent'),
                                      resetsAt=window.get('reset_at'),
                                      windowDurationMins=seconds / 60 if isinstance(seconds, (int, float)) else None)
        buckets[key] = bucket
    return {'rateLimitsByLimitId': buckets}


def _evaluate_native(request, *, load=_native, get=None):
    """DI seam for owned fake state/quota; not a public fixture permission mode."""
    import urllib.request
    import urllib.error

    if request.get('credits'):
        return _evaluate_credits(request)
    directory = Path(request['state_dir'])
    cache = Path(request.get('cache_dir', request['state_dir']))
    provider = request['provider']
    repair = False

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    def selected_get(url, headers):
        nonlocal repair
        try:
            if get is not None:
                return get(url, headers)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            with opener.open(urllib.request.Request(url, headers=headers), timeout=3) as response:
                raw = response.read(1048577)
                if len(raw) > 1048576:
                    raise ValueError('quota_response_too_large')
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            repair = exc.code == 401
            raise

    if provider == 'anthropic':
        native = load('claude', directory)
        usage = load('usage', cache)
        # Native snapshot is reused verbatim, including completeness, backoff,
        # locks and atomic writes. Replace its auth lookup with the selected
        # bearer only; the original keychain function is never loaded.
        native.keychain_token = lambda now: request['credential']
        native.USAGE_API = 'https://api.anthropic.com/api/oauth/usage'
        usage.snapshot.__globals__['helper'] = lambda: native

        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return json.dumps(selected_get(native.USAGE_API, {
                    'Authorization': 'Bearer ' + request['credential'],
                    'anthropic-beta': 'oauth-2025-04-20'})).encode()

        # Scope urllib replacement to this extracted module, never process globals.
        usage.snapshot.__globals__['urllib'] = SimpleNamespace(
            request=SimpleNamespace(Request=urllib.request.Request,
                                    urlopen=lambda *args, **kwargs: Response()),
            error=urllib.error)
        data = usage.snapshot(cache)
        lead, resume, hold = native.session_pace({'session_id': request['session_id']})
        limits = usage.as_limits(data)
        stale = usage.problem(data, time.time())
        pause = (native.stale_pause(limits, time.time(), request['model'], hold, lead, resume)
                 if stale else native.evaluate(limits, time.time(), request['model'], hold,
                                               lead_min=lead, resume_min=resume))
        repair = repair or 'HTTP 401' in str(data.get('error'))
        active = request.get('active_hold')
    else:
        native = load('codex', directory)
        native.snapshot.__globals__['rpc'] = lambda: _codex_shape(selected_get(
            'https://chatgpt.com/backend-api/wham/usage',
            request['headers']))
        data = native.snapshot(directory)
        bucket = native.bucket_for(request['model'])
        row = data['buckets'].get(bucket) or {}
        windows = row.get('windows', [])
        now = time.time()
        unavailable = (data.get('error') or row.get('complete') is False or not windows
                       or not 0 <= now - data.get('observed_at', 0) <= native.MAX_AGE_SEC
                       or any(w['reset'] <= now for w in windows))
        active = dict(request.get('active_hold') or {})
        pause = native.weekly_pause(directory, bucket, windows, now,
                                    unavailable=bool(unavailable), active_hold=active)
        repair = repair or '401' in str(data.get('error'))
    result = {'action': 'blocked' if repair else 'hold' if pause else 'allow',
              'reason': 'credential_repair' if repair else 'native_hold' if pause else 'native_allow',
              'retry_after': native.POLL_SEC, 'active_hold': active}
    if pause and not repair:
        result['note'] = _note(provider, pause, time.time())
    return result


def _child_main():
    try:
        request = json.loads(sys.stdin.buffer.read(65537))
        result = _evaluate_native(request)
    except Exception:
        result = dict(action='blocked', reason='native_unavailable', retry_after=5)
    # No raw native detail, credential, exception or usage payload crosses stdout.
    sys.stdout.write(json.dumps(result))


def _binding_row(section, provider, model, credential, base_url):
    if not isinstance(section, dict) or section.get('enabled') is not True:
        return {}, 'configuration_repair'
    endpoint = base_url.rstrip('/') if isinstance(base_url, str) else ''
    if not isinstance(provider, str) or not provider.strip() or not isinstance(model, str) or not model.strip():
        return {}, 'route_repair'
    for row in section.get('unmetered', []):
        if row == {'provider': provider, 'base_url': endpoint}:
            return row, 'explicit_unmetered'
    rows = [row for row in section.get('accounts', [])
            if isinstance(row, dict) and row.get('provider') == provider]
    # Association selects a label; a pool reference or token-derived identity cannot.
    digest = hashlib.sha256(credential.encode()).hexdigest() if isinstance(credential, str) else ''
    matches = [row for row in rows if isinstance(row.get('credential_sha256'), str)
               and hmac.compare_digest(row['credential_sha256'], digest)]
    row = matches[0] if len(matches) == 1 else (rows[0] if len(rows) == 1 else {})
    if endpoint not in _ENDPOINTS.get(provider, set()):
        return row, 'endpoint_repair'
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', str(row.get('account', ''))):
        return {}, 'account_repair'
    paths = ['state_dir'] + (['cache_dir'] if provider == 'anthropic' else [])
    if any(not isinstance(row.get(key), str) or not Path(row[key]).is_absolute() for key in paths):
        return row, 'state_repair'
    if len(matches) != 1 or not credential:
        return row, 'credential_repair'
    # All credentials explicitly associated with a label must share native stores.
    siblings = [r for r in rows if r.get('account') == row['account']]
    if any(any(r.get(key) != row.get(key) for key in paths) for r in siblings):
        return row, 'state_repair'
    return row, ''


class _Binding:
    def __init__(self, section, provider, model, credential, base_url, session_id, cancelled):
        from agent.quota_pacing import Route
        self._section = section
        self._provider, self._model = provider, model
        self._credential, self._base_url = credential, base_url
        self._session_id, self._cancelled = session_id, cancelled
        try:
            row, _ = _binding_row(section, provider, model, credential, base_url)
        except (TypeError, ValueError, AttributeError):
            row = {}
        account = row.get('account')
        account = account if isinstance(account, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', account) else 'unbound'
        self.route = Route(provider if isinstance(provider, str) and provider.strip() else 'unknown',
                           model if isinstance(model, str) and model.strip() else 'unknown', account)

    def __repr__(self):
        return '<NativeQuotaBinding>'

    def _request(self, route, active_hold, budget):
        from agent.quota_pacing import Decision
        if route != self.route:
            return Decision('blocked', 'route_repair', active_hold=active_hold)
        try:
            row, reason = _binding_row(self._section, self._provider, self._model,
                                       self._credential, self._base_url)
            if row.get('account', 'unbound') != self.route.account:
                reason = 'account_repair'
        except (TypeError, ValueError, AttributeError):
            row, reason = {}, 'configuration_repair'
        if self._cancelled and self._cancelled():
            reason = 'cancelled'
        if not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
            reason = 'native_deadline'
        if reason:
            return Decision('allow' if reason == 'explicit_unmetered' else 'blocked', reason,
                            active_hold=active_hold)
        request = dict(provider=self._provider, model=self._model, credential=self._credential,
                       state_dir=row['state_dir'], cache_dir=row.get('cache_dir', row['state_dir']),
                       session_id=self._session_id, active_hold=active_hold)
        if isinstance(row.get('credits'), dict):
            request.update(credits=row['credits'], label='Anthropic API credits')
        if self._provider == 'openai-codex':
            from agent.codex_headers import codex_account_headers
            request['headers'] = {'Authorization': 'Bearer ' + self._credential,
                                  'Accept': 'application/json', 'User-Agent': 'codex-cli',
                                  **codex_account_headers(self._credential)}
        return request

    def policy(self, route, active_hold, budget):
        from agent.quota_pacing import Decision
        request = self._request(route, active_hold, budget)
        if isinstance(request, Decision):
            return request
        deadline = time.monotonic() + budget
        child = subprocess.Popen([sys.executable, '-I', '-B', str(Path(__file__).absolute())],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL)
        try:
            payload = json.dumps(request).encode()
            while True:
                if self._cancelled and self._cancelled():
                    return Decision('blocked', 'cancelled', active_hold=active_hold)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return Decision('blocked', 'native_deadline', active_hold=active_hold)
                try:
                    output, _ = child.communicate(payload, timeout=min(.05, remaining))
                    return self._decision(output, active_hold)
                except subprocess.TimeoutExpired:
                    payload = None
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate()

    @staticmethod
    def _decision(output, active_hold):
        from agent.quota_pacing import Decision
        try:
            result = json.loads(output)
            if result['action'] not in ('allow', 'hold', 'blocked') or result['reason'] not in (
                    'native_allow', 'native_hold', 'credential_repair', 'native_unavailable'):
                raise ValueError('invalid_native_reply')
            return Decision(result['action'], result['reason'], result['retry_after'],
                            result.get('active_hold', active_hold), str(result.get('note', ''))[:240])
        except (ValueError, KeyError, TypeError):
            return Decision('blocked', 'native_unavailable', active_hold=active_hold)

    async def async_policy(self, route, active_hold, budget):
        # One owned thread supervises one child. Cancellation is joined, not orphaned.
        import threading
        stopped = threading.Event()
        original = self._cancelled
        twin = _Binding(self._section, self._provider, self._model, self._credential,
                        self._base_url, self._session_id,
                        lambda: stopped.is_set() or bool(original and original()))
        task = asyncio.create_task(asyncio.to_thread(twin.policy, route, active_hold, budget))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            stopped.set()
            try:
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
            finally:
                raise


def bind_request(section, *, provider, model, credential, base_url,
                 credential_ref='', session_id='', cancelled=None):
    """Local-only bind; credential_ref is deliberately NOT account evidence.

    section: enabled: true; accounts: [{provider, account, credential_sha256,
    state_dir, cache_dir (Claude shared snapshot directory)}]. Explicit unmetered
    rows contain exactly provider and base_url. Host alone skips disabled mode.
    """
    return _Binding(section, provider, model, credential, base_url, session_id, cancelled)


if __name__ == '__main__':
    _child_main()
