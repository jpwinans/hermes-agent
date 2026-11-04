"""Criteria 3/4: explicit binding, native hysteresis and selected-only refresh.

Pinned policy CODE below is copied without execution from the approved programs.
Every cache/hold is owned tmp_path state; every quota GET is injected. No real
native program, credential, account snapshot, child or provider is exercised.
"""
import asyncio
import hashlib
import json
import time
import urllib.error
from email.message import Message

import pytest
from agent import quota_pacing_native as adapter

PINNED_CODE = {'claude': 'HARD_PCT = 98\n\nWEEKLY_LEAD_MIN = 8 * 60\n\nWEEKLY_LEAD_RESUME = 4 * 60\n\nWEEKLY_MIN = 7 * 24 * 60\n\nPOLL_SEC = 30\n\ndef is_window(key):\n    # Weekly only: Anthropic dropped the five-hour session limit on 2026-10-31.\n    return key.startswith("seven_day")\n\ndef label(key):\n    """How a window reads in a message: \'weekly\', \'weekly Fable\', \'weekly opus\'."""\n    name = key.replace("seven_day", "", 1).lstrip(":_")\n    return f"weekly {name}" if name else "weekly"\n\ndef applies(key, model):\n    """Does this window limit an agent running `model`? Session and plan-wide\n    weekly windows limit every model; \'seven_day:Fable\' only a Fable model."""\n    if ":" not in key:\n        return True\n    return not model or key.startswith("seven_day:surface:") or key.split(":", 1)[1].lower() in model.lower()\n\ndef live_windows(limits, now, model=None):\n    """(key, used, resets, lead) for every window limiting `model` that is\n    present and not yet reset.\n\n    lead is minutes of usage ahead of a flat burn of the window — the weekly\n    pacing signal, and meaningless for the session window.\n    """\n    out = []\n    for key, w in limits.items():\n        if not is_window(key) or not isinstance(w, dict) or not applies(key, model):\n            continue\n        used, resets = w.get("used_percentage"), w.get("resets_at")\n        if used is None or not resets or now >= resets:\n            continue\n        out.append((key, used, resets, used / 100 * WEEKLY_MIN - (WEEKLY_MIN - (resets - now) / 60)))\n    return out\n\ndef session_pace(payload):\n    """(lead_min, resume_min, hold_path) for the calling session. A session listed in\n    session-pace.json paces by its own weekly lead and resume points and latches in its\n    own hold file (keyed by those points), so its stricter hold never holds a session on\n    the defaults. Subagents carry their parent\'s session_id. Anything unreadable or out\n    of order (resume not below lead) falls back to the defaults."""\n    entry = read_json(SESSION_PACE).get((payload or {}).get("session_id") or "")\n    try:\n        lead, resume = float(entry["lead_hours"]) * 60, float(entry["resume_hours"]) * 60\n    except (TypeError, KeyError, ValueError):\n        return WEEKLY_LEAD_MIN, WEEKLY_LEAD_RESUME, HOLD\n    if not 0 <= resume < lead:\n        return WEEKLY_LEAD_MIN, WEEKLY_LEAD_RESUME, HOLD\n    return lead, resume, os.path.join(USAGE_DIR, f"hold-{lead:g}-{resume:g}.json")\n\ndef evaluate(limits, now, model=None, hold_path=None, persist=True, unavailable=False,\n             lead_min=WEEKLY_LEAD_MIN, resume_min=WEEKLY_LEAD_RESUME):\n    """Serialize shared hold updates; read-only meters never change the latch."""\n    hold_path = HOLD if hold_path is None else hold_path\n    if not persist:\n        return _evaluate(limits, now, model, hold_path, False, unavailable, lead_min, resume_min)\n    os.makedirs(os.path.dirname(hold_path), exist_ok=True)\n    with open(hold_path + \'.lock\', \'a\') as lock:\n        fcntl.flock(lock, fcntl.LOCK_EX)\n        return _evaluate(limits, now, model, hold_path, True, unavailable, lead_min, resume_min)\n\ndef _evaluate(limits, now, model, hold_path, persist, unavailable,\n              lead_min=WEEKLY_LEAD_MIN, resume_min=WEEKLY_LEAD_RESUME):\n    """Active pause for an agent running `model`, as (resume_at, kind, detail),\n    or None. Only the windows limiting that model count; the longest hold wins.\n\n    A function of the limits and the clock, save for the weekly-lead hysteresis:\n    holds are read from and written to hold_path, one per window, so every\n    process shares them — and a gate only ever updates the windows it judged,\n    so an Opus agent never releases a Fable agent\'s hold. A hold\'s resume_at is\n    an estimate for the log and the statusline — the release is judged live,\n    from the lead at that moment.\n    """\n    holds = {k: h for k, h in read_json(hold_path).items()\n             if isinstance(h, dict) and is_window(k)}\n    before = dict(holds)\n    pauses = []\n    if unavailable:\n        # Stale data never pauses by itself. A latched weekly hold survives a failed\n        # refresh only while the LAST KNOWN reading, projected to the clock now, still\n        # shows the full +8h trigger lead; a stale token or a dead endpoint must not\n        # hold an agent forever on a lead nobody can verify (2026-09-21: an expired\n        # OAuth token held a session overnight on \'awaiting fresh lead\').\n        for key, used, resets, lead in live_windows(limits, now, model):\n            if used > HARD_PCT:\n                pauses.append((resets, \'stale-hard\', f\'{label(key)} {used}%, until reset\'))\n            elif ((holds.get(key) or {}).get("resets_at") == resets\n                  and lead >= lead_min - 1e-7):\n                pauses.append((now + POLL_SEC, \'weekly-lead-unverified\',\n                               f\'{label(key)} last seen {lead / 60:.1f}h ahead; awaiting fresh lead\'))\n        return max(pauses) if pauses else None\n    for key, used, resets, lead in live_windows(limits, now, model):\n        held = (holds.get(key) or {}).get("resets_at") == resets\n        if lead >= lead_min - 1e-7 or (held and lead > resume_min + 1e-7):\n            holds[key] = {"resets_at": resets, "since": holds[key]["since"] if held else now}\n        else:\n            holds.pop(key, None)\n        if used > HARD_PCT:                             # beats a lead hold, which stays recorded\n            pauses.append((resets, "weekly-hard", f"{label(key)} {used}%, until reset"))\n        elif key in holds:\n            pauses.append((min(resets, now + (lead - resume_min) * 60),\n                           "weekly-lead", f"{label(key)} {lead / 60:.1f}h ahead of pace"))\n    if persist and holds != before:                     # trigger, release, or new week\n        write_json(hold_path, holds)\n    return max(pauses) if pauses else None\n\ndef read_json(path):\n    try:\n        with open(path) as f:\n            return json.load(f)\n    except (OSError, ValueError):\n        return {}\n\ndef write_json(path, data):\n    os.makedirs(os.path.dirname(path), exist_ok=True)\n    tmp = f"{path}.{os.getpid()}"\n    with open(tmp, "w") as f:\n        json.dump(data, f)\n    os.replace(tmp, path)\n\ndef read_limits():\n    return read_json(LIMITS)\n\ndef iso_epoch(s):\n    try:\n        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))\n    except (AttributeError, ValueError):\n        return None\n    ts = (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()\n    return round(ts / 60) * 60\n\ndef stale_pause(limits, now, model=None, hold_path=None,\n                lead_min=WEEKLY_LEAD_MIN, resume_min=WEEKLY_LEAD_RESUME):\n    # The whole last-known snapshot: the unavailable branch needs each window\'s lead,\n    # not only the rows already over the hard limit.\n    known = {key: row for key, row in limits.items()\n             if is_window(key) and isinstance(row, dict)\n             and isinstance(row.get(\'used_percentage\'), (int, float))\n             and 0 <= row[\'used_percentage\'] <= 100}\n    return evaluate(known, now, model, hold_path, persist=False, unavailable=True,\n                    lead_min=lead_min, resume_min=resume_min)\n', 'codex': 'REFRESH_SEC = 60\n\nPOLL_SEC = 5\n\nHARD_PCT = 98\n\nMAX_AGE_SEC = 90\n\nWEEKLY_LEAD_TRIGGER_HOURS, WEEKLY_LEAD_RESUME_HOURS = 8, 4\n\ndef normalize(result):\n    """Keep quota fields only. Do not assume primary means five hours."""\n    buckets = result.get("rateLimitsByLimitId") or {}\n    single = result.get("rateLimits") or {}\n    if not buckets and single:\n        buckets = {single.get("limitId") or "codex": single}\n    out = {}\n    for key, bucket in buckets.items():\n        windows = []\n        for slot in ("primary", "secondary"):\n            w = bucket.get(slot) or {}\n            values = [w.get(k) for k in ("usedPercent", "windowDurationMins", "resetsAt")]\n            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)\n                       and math.isfinite(v) for v in values):\n                continue\n            used, minutes, reset = values\n            if not 0 <= used <= 100 or minutes <= 0 or reset <= 0:\n                continue\n            windows.append({"used": used, "minutes": minutes, "reset": reset})\n        out[key] = {"name": bucket.get("limitName"), "windows": windows,\n                    "complete": len(windows) == sum(bool(bucket.get(slot)) for slot in ("primary", "secondary"))}\n    return out\n\ndef read_cache(directory):\n    try:\n        data = json.loads((directory / "limits.json").read_text())\n        if isinstance(data.get("buckets"), dict) and isinstance(data.get("checked_at"), (int, float)):\n            return data\n    except (OSError, ValueError, AttributeError):\n        pass\n    return {"buckets": {}, "checked_at": 0}\n\ndef atomic_json(path, data):\n    fd, name = tempfile.mkstemp(prefix=".limits-", dir=path.parent)\n    try:\n        with os.fdopen(fd, "w") as f:\n            json.dump(data, f)\n        os.replace(name, path)\n    finally:\n        if os.path.exists(name):\n            os.unlink(name)\n\ndef snapshot(directory, force=False, refresh_sec=REFRESH_SEC):\n    directory.mkdir(parents=True, exist_ok=True, mode=0o700)\n    # Serialize refreshes across all sessions. Readers cannot see partial JSON.\n    with (directory / "refresh.lock").open("a") as lock:\n        fcntl.flock(lock, fcntl.LOCK_EX)\n        data = read_cache(directory)\n        if not force and time.time() - data["checked_at"] < refresh_sec:\n            return data\n        try:\n            buckets = normalize(rpc())\n            if not buckets or not any(b["windows"] for b in buckets.values()):\n                raise RuntimeError("Codex returned no usable quota windows")\n            data = {"buckets": buckets, "observed_at": time.time(),\n                    "checked_at": time.time(), "error": None}\n        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:\n            # Preserve the last successful reading; throttle failed refreshes too.\n            data.update(checked_at=time.time(), error=str(exc))\n        atomic_json(directory / "limits.json", data)\n        return data\n\ndef ideal(w, now):\n    return max(0, min(100, 100 * (1 - (w["reset"] - now) / (w["minutes"] * 60))))\n\ndef evaluate(windows, now, holds=None):\n    """Keep the bucket latched until every fresh weekly lead is at most +4h.\n\n    A reset timestamp is quota metadata, not the identity of an active hold.\n    Provider corrections must not turn a latched +6h reading into a new call.\n    """\n    holds = {} if holds is None else holds\n    pauses, weekly = [], []\n    for w in windows:\n        used, minutes, reset = w["used"], w["minutes"], w["reset"]\n        if reset <= now:\n            continue\n        label = w.get("label") or ("weekly" if minutes == 10080 else "session" if minutes == 300 else f"{minutes:g}m")\n        if minutes == 10080:\n            lead = (used - ideal(w, now)) / 100 * minutes / 60\n            weekly.append((lead, w))\n        if used > HARD_PCT:\n            pauses.append((reset, f"{label}-hard", f"{label} {used:g}%, until reset"))\n    if weekly:\n        lead, w = max(weekly, key=lambda item: item[0])\n        if lead >= WEEKLY_LEAD_TRIGGER_HOURS - 1e-9 or (\n                holds and lead > WEEKLY_LEAD_RESUME_HOURS + 1e-9):\n            holds.update(reset=w["reset"])\n            resume = w["reset"] - w["minutes"] * 60 * (1 - w["used"] / 100) - WEEKLY_LEAD_RESUME_HOURS * 3600\n            pauses.append((min(w["reset"], resume), "weekly-lead", f"weekly {lead:.1f}h ahead of pace"))\n        else:\n            holds.clear()\n    return max(pauses) if pauses else None\n\ndef weekly_pause(directory, bucket, windows, now, unavailable=False, persist=True, active_hold=None):\n    """Serialize account-wide hysteresis; displays read it without changing it.\n\n    Missing/stale data cannot clear an established hold, even at its old ETA.\n    Only a fresh live weekly reading (including a new week) can release it.\n    """\n    def decide():\n        path = directory / \'hold.json\'\n        try:\n            state = json.loads(path.read_text())\n        except FileNotFoundError:\n            state = {}\n        if not isinstance(state, dict) or not isinstance(state.get(bucket, {}), dict):\n            raise ValueError(\'Invalid pacing hold state\')\n        held = dict(state.get(bucket, {}))\n        before = dict(held)\n        # A waiting hook also remembers its own hold. An older process or another\n        # writer clearing the shared file cannot prematurely release this call.\n        if not held and active_hold:\n            held.update(active_hold)\n        weekly = any(w[\'minutes\'] == 10080 and w[\'reset\'] > now for w in windows)\n        if unavailable or (held and not weekly):\n            pause = stale_pause(windows, now)\n            if held:\n                pending = (now + POLL_SEC, \'weekly-lead-unverified\',\n                           \'weekly hold: awaiting fresh lead <=4h\')\n                pause = max(pause, pending) if pause else pending\n        else:\n            pause = evaluate(windows, now, held)\n        if persist and held != before:\n            if held:\n                state[bucket] = held\n            else:\n                state.pop(bucket, None)\n            atomic_json(path, state)\n        if persist and active_hold is not None:\n            active_hold.clear()\n            active_hold.update(held)\n        return pause\n    if not persist:\n        return decide()\n    directory.mkdir(parents=True, exist_ok=True, mode=0o700)\n    with (directory / \'hold.lock\').open(\'a\') as lock:\n        fcntl.flock(lock, fcntl.LOCK_EX)\n        return decide()\n\ndef stale_pause(windows, now):\n    """Old pacing leads are ignored; only known usage >98% can hold until reset."""\n    high = [w for w in windows if isinstance(w, dict)\n            and isinstance(w.get(\'used\'), (int,float)) and HARD_PCT < w[\'used\'] <= 100\n            and isinstance(w.get(\'reset\'), (int,float)) and w[\'reset\'] > now]\n    return evaluate(high, now)\n\ndef bucket_for(model):\n    # Reserve is a fallback bucket, not a replacement for the ordinary quota.\n    return "codex_bengalfox" if "spark" in model.lower() else "codex"\n', 'usage': 'REFRESH_SEC = 60\n\nMAX_AGE_SEC = 90\n\nSCHEMA = 2\n\ndef normalize(source, iso_epoch):\n    rows = {}\n    for key, row in source.items():\n        if (key == \'five_hour\' or key.startswith(\'seven_day\')) and isinstance(row, dict) and any(field in row for field in (\'utilization\', \'used_percentage\', \'resets_at\')):\n            if key not in (\'five_hour\', \'seven_day\') and row.get(\'utilization\', row.get(\'used_percentage\')) is None and row.get(\'resets_at\') is None:\n                continue\n            name = key.removeprefix(\'seven_day\').lstrip(\':_\') if key != \'five_hour\' else \'\'\n            canonical = \'five_hour\' if key == \'five_hour\' else \'seven_day\' + (\':\' + name if name else \'\')\n            rows[canonical] = (row.get(\'utilization\', row.get(\'used_percentage\')), row.get(\'resets_at\'))\n    for row in source.get(\'limits\') or []:\n        scope = row.get(\'scope\') or {}\n        model = (scope.get(\'model\') or {}).get(\'display_name\')\n        surface = (scope.get(\'surface\') or {}).get(\'display_name\')\n        kind = row.get(\'kind\', \'\')\n        if kind == \'session\' and not scope:\n            key = \'five_hour\'\n        elif kind == \'weekly_all\' and not scope:\n            key = \'seven_day\'\n        elif kind.startswith(\'weekly\') and (model or surface):\n            # Surface limits conservatively apply to all local Claude models.\n            key = \'seven_day:\' + model if model else \'seven_day:surface:\' + surface\n        else:\n            continue\n        rows[key] = (row.get(\'percent\'), row.get(\'resets_at\'))\n    # Weekly only: Anthropic dropped the five-hour session limit on 2026-10-31.\n    rows.pop(\'five_hour\', None)\n    windows = []\n    for key, (used, reset) in rows.items():\n        if isinstance(reset, str):\n            reset = iso_epoch(reset)\n        if reset is None and used == 0:\n            # An idle window has no reset yet; a full window from now is the latest it could be.\n            reset = time.time() + (300 if key == \'five_hour\' else 10080) * 60\n        if (all(isinstance(v, (int, float)) and not isinstance(v, bool)\n                and math.isfinite(v) for v in (used, reset))\n                and 0 <= used <= 100 and reset > 0):\n            name = key.removeprefix(\'seven_day:\') if \':\' in key else None\n            windows.append({\'key\': key, \'used\': used, \'minutes\': 300 if key == \'five_hour\' else 10080,\n                            \'reset\': reset, \'scope\': name,\n                            \'label\': \'5H session\' if key == \'five_hour\' else \'7D weekly \' + (name or \'all models\')})\n    return {\'name\': \'Claude\', \'windows\': windows,\n            \'complete\': len(windows) == len(rows) and \'seven_day\' in {w[\'key\'] for w in windows}}\n\ndef as_limits(data):\n    return {w[\'key\']: {\'used_percentage\': w[\'used\'], \'resets_at\': w[\'reset\']}\n            for w in data.get(\'windows\', []) if \'key\' in w}\n\ndef retry_after_sec(header, now, default=300, floor=REFRESH_SEC, cap=3600):\n    """Seconds to wait from a Retry-After header: delta-seconds or an HTTP-date\n    (RFC 9110 10.2.3). Absent or unparseable waits the default."""\n    try:\n        delay = float(header)\n    except (TypeError, ValueError):\n        try:\n            delay = email.utils.parsedate_to_datetime(header).timestamp() - now\n        except (TypeError, ValueError, IndexError):\n            delay = default\n    if not math.isfinite(delay):\n        delay = default\n    return min(max(delay, floor), cap)\n\ndef _short(seconds):\n    return f\'{seconds / 3600:.0f}h\' if seconds >= 5400 else f\'{max(1, round(seconds / 60))}m\'\n\ndef problem(data, now, max_age=MAX_AGE_SEC):\n    """None when the reading is fresh and whole; else a terse reason for the statusline."""\n    error = data.get(\'error\')\n    if error:\n        detail = error.removeprefix(\'Claude usage refresh unavailable: \').removeprefix(\'Claude \')\n        wait = data.get(\'retry_after\', 0) - now\n        return f\'{detail}, retry {_short(wait)}\' if wait > 0 else detail\n    if not data.get(\'complete\') or data.get(\'schema\') != SCHEMA:\n        return \'incomplete\'\n    age = now - data.get(\'observed_at\', 0)\n    if not 0 <= age <= max_age:\n        return f\'{_short(age)} old\' if age > 0 else \'clock skew\'\n    if any(w[\'reset\'] <= now for w in data.get(\'windows\', [])):\n        return \'reset passed\'\n    return None\n\ndef snapshot(directory=CACHE_DIR, refresh_sec=REFRESH_SEC, force=False):\n    directory.mkdir(parents=True, exist_ok=True)\n    cache = directory / \'claude-v2.json\'\n    with (directory / \'claude-v2.lock\').open(\'a\') as lock:\n        fcntl.flock(lock, fcntl.LOCK_EX)\n        try:\n            data = json.loads(cache.read_text())\n            if not isinstance(data, dict):\n                raise ValueError(\'Invalid cache\')\n        except (OSError, ValueError):\n            data = {\'name\': \'Claude\', \'windows\': [], \'checked_at\': 0}\n        now = time.time()\n        if data.get(\'schema\') == SCHEMA and now < data.get(\'retry_after\', 0):\n            return data\n        if not force and data.get(\'schema\') == SCHEMA and 0 <= now - data.get(\'checked_at\', 0) < refresh_sec:\n            return data\n        try:\n            source = helper()\n            if not data.get(\'windows\'):\n                fallback = normalize(source.read_limits(), source.iso_epoch)\n                data.update(fallback)  # display only; never mark fallback fresh\n            token = source.keychain_token(now)\n            if not token:\n                raise RuntimeError(\'Claude login unavailable or expired\')\n            request = urllib.request.Request(source.USAGE_API, headers={\n                \'Authorization\': \'Bearer \' + token,\n                \'anthropic-beta\': \'oauth-2025-04-20\'})\n            with urllib.request.urlopen(request, timeout=10) as response:\n                raw = json.load(response)\n                bucket = normalize(raw, source.iso_epoch)\n            if not bucket[\'complete\']:\n                # Kept so a reply this parser can\'t read can be looked at (it holds no token).\n                (directory / \'claude-incomplete.json\').write_text(json.dumps(raw))\n                raise RuntimeError(\'Claude quota response incomplete\')\n            known = {w.get(\'key\') for w in data.get(\'windows\', []) if w.get(\'reset\', 0) > now and w.get(\'scope\')}\n            present = {w[\'key\'] for w in bucket[\'windows\']}\n            if known - present:\n                raise RuntimeError(\'Claude model quota missing from refresh\')\n            data.update(bucket, observed_at=time.time(), error=None, retry_after=0)\n        except (OSError, ValueError, RuntimeError, TypeError, AttributeError) as exc:\n            detail = str(exc) if isinstance(exc, RuntimeError) else \'HTTP \'+str(exc.code) if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__\n            data[\'error\'] = \'Claude usage refresh unavailable: \' + detail\n            header = exc.headers.get(\'Retry-After\') if isinstance(exc, urllib.error.HTTPError) and exc.headers else None\n            # A 429 always backs off; a 503 only when the server says how long.\n            if isinstance(exc, urllib.error.HTTPError) and (exc.code == 429 or (exc.code == 503 and header)):\n                data[\'retry_after\'] = time.time() + retry_after_sec(header, time.time())\n        data.update(checked_at=time.time(), schema=SCHEMA)\n        temporary = directory / \'claude-v2.json.tmp\'\n        temporary.write_text(json.dumps(data))\n        temporary.replace(cache)\n        return data\n'}

@pytest.fixture
def native(tmp_path, monkeypatch):
    sources = {}
    for kind, code in PINNED_CODE.items():
        path = tmp_path / (kind + '.py')
        path.write_text(code)
        sources[kind] = (str(path), hashlib.sha256(code.encode()).hexdigest())
    monkeypatch.setattr(adapter, '_SOURCES', sources)
    return adapter._native


def config(tmp_path, provider='openai-codex', credential='owned-fake-bearer'):
    return {'enabled': True, 'accounts': [{
        'provider': provider, 'account': 'owned-account',
        'credential_sha256': hashlib.sha256(credential.encode()).hexdigest(),
        'state_dir': str(tmp_path / 'state'), 'cache_dir': str(tmp_path / 'cache')}]}


def binding(section, credential='owned-fake-bearer', provider='openai-codex', model='gpt-test'):
    return adapter.bind_request(section, provider=provider, model=model,
        credential=credential, base_url=next(iter(adapter._ENDPOINTS[provider])),
        session_id='owned-session', credential_ref='not-account-evidence')


def request(tmp_path, provider='openai-codex', model='gpt-test', active=None):
    return dict(provider=provider, model=model, credential='owned-fake-bearer',
                state_dir=str(tmp_path / 'state'), cache_dir=str(tmp_path / 'cache'),
                session_id='owned-session', active_hold=active,
                headers={'Authorization': 'Bearer owned-fake-bearer',
                         'Accept': 'application/json', 'User-Agent': 'codex-cli'})


def codex_usage(lead=9, used=50, bucket='codex'):
    reset = time.time() + (10080 * (1 - used / 100) + lead * 60) * 60
    limits = {'primary_window': {'used_percent': used, 'limit_window_seconds': 604800,
                                 'reset_at': reset}}
    return {'rate_limit': limits} if bucket == 'codex' else {
        'additional_rate_limits': [{'limit_id': bucket, 'rate_limit': limits}]}


@pytest.mark.parametrize('case', ['absent', 'null', 'empty', 'populated', 'ordinary_null'])
def test_codex_shape_optional_additional_limits(case):
    # Explicit synthetic counterexamples, not a claimed captured live response.
    window = {'used_percent': 10, 'limit_window_seconds': 604800, 'reset_at': 2000000000}
    raw: dict = {'rate_limit': {'primary_window': window}}
    if case == 'null':
        raw['additional_rate_limits'] = None
    elif case == 'empty':
        raw['additional_rate_limits'] = []
    elif case == 'populated':
        raw['additional_rate_limits'] = [{'limit_id': 'codex_bengalfox',
                                          'rate_limit': {'primary_window': window}}]
    elif case == 'ordinary_null':
        raw = {'rate_limit': None, 'additional_rate_limits': []}
    buckets = adapter._codex_shape(raw)['rateLimitsByLimitId']
    assert ('codex_bengalfox' in buckets) == (case == 'populated')
    if case == 'ordinary_null':
        assert buckets['codex'] == {}
    else:
        assert buckets['codex']['primary'] == {
            'usedPercent': 10, 'resetsAt': 2000000000, 'windowDurationMins': 10080}


def test_codex_shape_false_additional_is_not_reinterpreted_as_absent():
    with pytest.raises(TypeError):
        adapter._codex_shape({'rate_limit': {}, 'additional_rate_limits': False})


def test_codex_null_additional_still_uses_ordinary_weekly_hold(tmp_path, native):
    raw = dict(codex_usage(), additional_rate_limits=None)
    seen = []
    def get(url, headers):
        seen.append(url)
        return raw
    out = adapter._evaluate_native(request(tmp_path), load=native, get=get)
    assert out['action'] == 'hold'
    assert out['active_hold']
    assert len(seen) == 1


def test_binding_local_no_discovery_and_private(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter.subprocess, 'Popen', lambda *a, **k: pytest.fail('bind spawned'))
    b = binding(config(tmp_path))
    assert b.route.account == 'owned-account'
    assert 'owned-fake-bearer' not in repr(b)
    assert isinstance(b._request(b.route, None, 5), dict)


def test_mismatch_retains_account_and_blocks(tmp_path):
    b = binding(config(tmp_path), credential='different-fake-bearer')
    assert b.route.account == 'owned-account'
    held = {'reset': 123}
    d = b.policy(b.route, held, 5)
    assert (d.action, d.reason, d.active_hold) == ('blocked', 'credential_repair', held)


@pytest.mark.parametrize('change', ['unknown', 'endpoint', 'state', 'duplicate'])
def test_unknown_routes_and_invalid_associations_block(tmp_path, change):
    section = config(tmp_path)
    b = binding(section)
    if change == 'unknown':
        section['accounts'] = []
    elif change == 'endpoint':
        b._base_url = 'https://untrusted.invalid'
    elif change == 'state':
        section['accounts'][0]['state_dir'] = 'relative'
    else:
        section['accounts'].append(dict(section['accounts'][0]))
    assert b.policy(b.route, None, 5).action == 'blocked'


def test_explicit_unmetered_only(tmp_path):
    b = adapter.bind_request({'enabled': True, 'unmetered': [
        {'provider': 'local-owned', 'base_url': 'http://localhost:9999'}]},
        provider='local-owned', model='owned', credential='', base_url='http://localhost:9999')
    assert b.policy(b.route, None, 5).action == 'allow'
    b._base_url = 'http://localhost:9998'
    assert b.policy(b.route, None, 5).action == 'blocked'


def test_codex_refresh_alone_and_account_latch(tmp_path, native):
    r = request(tmp_path)
    seen = []
    def get(url, headers):
        seen.append((url, headers))
        return codex_usage()
    first = adapter._evaluate_native(r, load=native, get=get)
    assert first['action'] == 'hold'
    assert first['active_hold']
    assert len(seen) == 1
    assert seen[0][1]['Authorization'] == 'Bearer owned-fake-bearer'
    # Same cache and latch, new wait: no token-derived state namespace.
    again = adapter._evaluate_native(r, load=native, get=lambda *a: pytest.fail('cache miss'))
    assert again['action'] == 'hold'
    path = tmp_path / 'state' / 'limits.json'
    data = json.loads(path.read_text()); data['checked_at'] = 0
    path.write_text(json.dumps(data))
    released = adapter._evaluate_native(dict(r, active_hold=first['active_hold']),
                                        load=native, get=lambda *a: codex_usage(lead=3))
    assert released['action'] == 'allow'
    assert released['active_hold'] == {}


def test_codex_stale_hold_and_initial_unknown_differ(tmp_path, native):
    def unavailable(*args):
        raise OSError('owned fake unavailable')
    unknown = adapter._evaluate_native(request(tmp_path), load=native, get=unavailable)
    assert unknown['action'] == 'allow'  # Native initial-unknown quota policy.
    held = adapter._evaluate_native(request(tmp_path, active={'reset': 123}),
                                    load=native, get=unavailable)
    assert held['action'] == 'hold'
    assert held['active_hold'] == {'reset': 123}


def test_codex_spark_is_separate(tmp_path, native):
    out = adapter._evaluate_native(request(tmp_path, model='spark-owned'),
                                   load=native, get=lambda *a: codex_usage())
    assert out['action'] == 'allow'  # Ordinary quota is not Spark quota.
    other = tmp_path / 'other'
    out = adapter._evaluate_native(request(other, model='spark-owned'), load=native,
        get=lambda *a: codex_usage(bucket='codex_bengalfox'))
    assert out['action'] == 'hold'


def test_codex_active_hold_survives_other_writer_clear(tmp_path, native):
    first = adapter._evaluate_native(request(tmp_path), load=native, get=lambda *a: codex_usage())
    (tmp_path / 'state' / 'hold.json').write_text('{}')
    data_path = tmp_path / 'state' / 'limits.json'
    data = json.loads(data_path.read_text()); data['checked_at'] = 0
    data_path.write_text(json.dumps(data))
    out = adapter._evaluate_native(request(tmp_path, active=first['active_hold']),
                                   load=native, get=lambda *a: codex_usage(lead=6))
    assert out['action'] == 'hold'


def test_claude_full_limits_and_session_pace(tmp_path, native):
    r = request(tmp_path, provider='anthropic', model='claude-fable')
    state = tmp_path / 'state'; state.mkdir()
    (state / 'session-pace.json').write_text(json.dumps({
        'owned-session': {'lead_hours': 2, 'resume_hours': 0}}))
    now = time.time()
    raw = {'seven_day': {'utilization': 20, 'resets_at': now + 604800 * .7},
           'limits': [{'kind': 'weekly_model', 'scope': {'model': {'display_name': 'fable'}},
                       'percent': 50, 'resets_at': now + 604800 * .5 + 3 * 3600}]}
    out = adapter._evaluate_native(r, load=native, get=lambda *a: raw)
    assert out['action'] == 'hold'
    assert (state / 'hold-120-0.json').exists()
    opus = adapter._evaluate_native(dict(r, model='claude-opus'), load=native,
                                    get=lambda *a: pytest.fail('cache miss'))
    assert opus['action'] == 'allow'


def test_claude_stale_is_not_codex_stale(tmp_path, native):
    r = request(tmp_path, provider='anthropic')
    def fail(*a):
        raise OSError('owned fake')
    out = adapter._evaluate_native(r, load=native, get=fail)
    assert out['action'] == 'allow'
    helper = native('claude', tmp_path / 'state')
    now = time.time()
    limits = {'seven_day': {'used_percentage': 50, 'resets_at': now + 604800 * .5 + 9 * 3600}}
    assert helper.evaluate(limits, now)
    limits['seven_day']['resets_at'] -= 3 * 3600
    assert helper.stale_pause(limits, now) is None


@pytest.mark.parametrize('provider', ['anthropic', 'openai-codex'])
def test_expired_selected_bearer_is_repair_blocked(tmp_path, native, provider):
    def expired(url, headers):
        raise urllib.error.HTTPError(url, 401, 'owned fake', Message(), None)
    out = adapter._evaluate_native(request(tmp_path, provider=provider), load=native, get=expired)
    assert out['action'] == 'blocked'
    assert out['reason'] == 'credential_repair'
    again = adapter._evaluate_native(request(tmp_path, provider=provider), load=native,
                                     get=lambda *a: pytest.fail('unthrottled'))
    assert again['action'] == 'blocked'


def test_child_secret_stdin_and_reap_on_cancel(tmp_path, monkeypatch):
    b = binding(config(tmp_path))
    seen = {}
    class Child:
        def __init__(self, argv, **kwargs):
            seen['argv'] = argv; self.dead = False; self.calls = 0
        def communicate(self, payload=None, timeout=None):
            self.calls += 1
            if timeout is None:
                seen['reaped'] = True
                return b'', b''
            seen['payload'] = payload
            b._cancelled = lambda: True
            raise adapter.subprocess.TimeoutExpired('owned', timeout)
        def poll(self):
            return 0 if self.dead else None
        def kill(self):
            self.dead = True; seen['killed'] = True
    monkeypatch.setattr(adapter.subprocess, 'Popen', Child)
    assert b.policy(b.route, {'reset': 123}, 5).reason == 'cancelled'
    assert 'owned-fake-bearer' not in ' '.join(seen['argv'])
    assert b'owned-fake-bearer' in seen['payload']
    assert seen['killed'] and seen['reaped']


def test_source_pin_mismatch_blocks_before_policy_load(tmp_path, native):
    path, _pin = adapter._SOURCES['codex']
    adapter._SOURCES['codex'] = (path, '0' * 64)
    with pytest.raises(ValueError, match='native_source_changed'):
        native('codex', tmp_path / 'state')


def test_deadline_kills_owned_child_and_preserves_hold(tmp_path, monkeypatch):
    b = binding(config(tmp_path))
    seen = {}
    ticks = iter([0, 10])
    from types import SimpleNamespace
    monkeypatch.setattr(adapter, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    class Child:
        def __init__(self, *args, **kwargs):
            self.dead = False
        def poll(self):
            return 0 if self.dead else None
        def kill(self):
            self.dead = True; seen['killed'] = True
        def communicate(self):
            seen['reaped'] = True
    monkeypatch.setattr(adapter.subprocess, 'Popen', Child)
    held = {'reset': 123}
    result = b.policy(b.route, held, 5)
    assert result.reason == 'native_deadline' and result.active_hold is held
    assert seen == {'killed': True, 'reaped': True}


def test_malformed_selected_route_is_visible_blocked(tmp_path):
    b = adapter.bind_request(config(tmp_path), provider=' ', model=' ',
                             credential='owned-fake-bearer', base_url='')
    assert b.policy(b.route, None, 5).action == 'blocked'


def test_async_supervision_yields_and_joins(tmp_path, monkeypatch):
    import threading
    entered, finished = threading.Event(), threading.Event()
    b = binding(config(tmp_path))
    def supervise(self, route, active, budget):
        entered.set()
        while not self._cancelled():
            finished.wait(.01)
        finished.set()
        from agent.quota_pacing import Decision
        return Decision('blocked', 'cancelled')
    monkeypatch.setattr(adapter._Binding, 'policy', supervise)
    async def exercise():
        task = asyncio.create_task(b.async_policy(b.route, None, 5))
        await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    asyncio.run(exercise())


def test_selected_codex_workspace_headers_reach_quota_get(tmp_path, native):
    import base64
    claims = {'https://api.openai.com/auth': {'chatgpt_account_id': 'owned-workspace',
                                           'chatgpt_data_residency': 'owned-region'}}
    token = 'owned.' + base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip('=') + '.fake'
    b = binding(config(tmp_path, credential=token), credential=token)
    payload = b._request(b.route, None, 5)
    seen = []
    def get(url, headers):
        seen.append((url, headers))
        return codex_usage(lead=3)
    assert adapter._evaluate_native(payload, load=native, get=get)['action'] == 'allow'
    assert seen == [('https://chatgpt.com/backend-api/wham/usage', {
        'Authorization': 'Bearer ' + token, 'Accept': 'application/json',
        'User-Agent': 'codex-cli', 'ChatGPT-Account-ID': 'owned-workspace',
        'x-openai-internal-codex-residency': 'owned-region'})]
    assert b.route.account == 'owned-account'  # Claims shape the request, not account identity.


@pytest.mark.parametrize('provider', ['anthropic', 'openai-codex'])
def test_usage_forbidden_retains_native_unavailable_policy(tmp_path, native, provider):
    def forbidden(url, headers):
        raise urllib.error.HTTPError(url, 403, 'owned fake scope refusal', Message(), None)
    result = adapter._evaluate_native(request(tmp_path, provider=provider), load=native, get=forbidden)
    assert result['action'] == 'allow'  # Native initial-unknown policy, not expired authentication.
    assert result['reason'] == 'native_allow'
