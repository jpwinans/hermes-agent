"""Owned stdlib-only driver; not a product entry point or deployment fixture.

SIGALRM bounds even blocked stdin/import/native-lock paths. No descendants.
The policy constant is copied verbatim from the existing pinned fake fixture.
"""
import signal

# Default SIGALRM terminates this exact process, including blocked stdin/locks.
signal.signal(signal.SIGALRM, signal.SIG_DFL)
signal.alarm(30)

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

# Verbatim copied fixture data, never imported from a native installation.
PINNED_CODE = {'codex': 'REFRESH_SEC = 60\n\nPOLL_SEC = 5\n\nHARD_PCT = 98\n\nMAX_AGE_SEC = 90\n\nWEEKLY_LEAD_TRIGGER_HOURS, WEEKLY_LEAD_RESUME_HOURS = 8, 4\n\ndef normalize(result):\n    """Keep quota fields only. Do not assume primary means five hours."""\n    buckets = result.get("rateLimitsByLimitId") or {}\n    single = result.get("rateLimits") or {}\n    if not buckets and single:\n        buckets = {single.get("limitId") or "codex": single}\n    out = {}\n    for key, bucket in buckets.items():\n        windows = []\n        for slot in ("primary", "secondary"):\n            w = bucket.get(slot) or {}\n            values = [w.get(k) for k in ("usedPercent", "windowDurationMins", "resetsAt")]\n            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)\n                       and math.isfinite(v) for v in values):\n                continue\n            used, minutes, reset = values\n            if not 0 <= used <= 100 or minutes <= 0 or reset <= 0:\n                continue\n            windows.append({"used": used, "minutes": minutes, "reset": reset})\n        out[key] = {"name": bucket.get("limitName"), "windows": windows,\n                    "complete": len(windows) == sum(bool(bucket.get(slot)) for slot in ("primary", "secondary"))}\n    return out\n\ndef read_cache(directory):\n    try:\n        data = json.loads((directory / "limits.json").read_text())\n        if isinstance(data.get("buckets"), dict) and isinstance(data.get("checked_at"), (int, float)):\n            return data\n    except (OSError, ValueError, AttributeError):\n        pass\n    return {"buckets": {}, "checked_at": 0}\n\ndef atomic_json(path, data):\n    fd, name = tempfile.mkstemp(prefix=".limits-", dir=path.parent)\n    try:\n        with os.fdopen(fd, "w") as f:\n            json.dump(data, f)\n        os.replace(name, path)\n    finally:\n        if os.path.exists(name):\n            os.unlink(name)\n\ndef snapshot(directory, force=False, refresh_sec=REFRESH_SEC):\n    directory.mkdir(parents=True, exist_ok=True, mode=0o700)\n    # Serialize refreshes across all sessions. Readers cannot see partial JSON.\n    with (directory / "refresh.lock").open("a") as lock:\n        fcntl.flock(lock, fcntl.LOCK_EX)\n        data = read_cache(directory)\n        if not force and time.time() - data["checked_at"] < refresh_sec:\n            return data\n        try:\n            buckets = normalize(rpc())\n            if not buckets or not any(b["windows"] for b in buckets.values()):\n                raise RuntimeError("Codex returned no usable quota windows")\n            data = {"buckets": buckets, "observed_at": time.time(),\n                    "checked_at": time.time(), "error": None}\n        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:\n            # Preserve the last successful reading; throttle failed refreshes too.\n            data.update(checked_at=time.time(), error=str(exc))\n        atomic_json(directory / "limits.json", data)\n        return data\n\ndef ideal(w, now):\n    return max(0, min(100, 100 * (1 - (w["reset"] - now) / (w["minutes"] * 60))))\n\ndef evaluate(windows, now, holds=None):\n    """Keep the bucket latched until every fresh weekly lead is at most +4h.\n\n    A reset timestamp is quota metadata, not the identity of an active hold.\n    Provider corrections must not turn a latched +6h reading into a new call.\n    """\n    holds = {} if holds is None else holds\n    pauses, weekly = [], []\n    for w in windows:\n        used, minutes, reset = w["used"], w["minutes"], w["reset"]\n        if reset <= now:\n            continue\n        label = w.get("label") or ("weekly" if minutes == 10080 else "session" if minutes == 300 else f"{minutes:g}m")\n        if minutes == 10080:\n            lead = (used - ideal(w, now)) / 100 * minutes / 60\n            weekly.append((lead, w))\n        if used > HARD_PCT:\n            pauses.append((reset, f"{label}-hard", f"{label} {used:g}%, until reset"))\n    if weekly:\n        lead, w = max(weekly, key=lambda item: item[0])\n        if lead >= WEEKLY_LEAD_TRIGGER_HOURS - 1e-9 or (\n                holds and lead > WEEKLY_LEAD_RESUME_HOURS + 1e-9):\n            holds.update(reset=w["reset"])\n            resume = w["reset"] - w["minutes"] * 60 * (1 - w["used"] / 100) - WEEKLY_LEAD_RESUME_HOURS * 3600\n            pauses.append((min(w["reset"], resume), "weekly-lead", f"weekly {lead:.1f}h ahead of pace"))\n        else:\n            holds.clear()\n    return max(pauses) if pauses else None\n\ndef weekly_pause(directory, bucket, windows, now, unavailable=False, persist=True, active_hold=None):\n    """Serialize account-wide hysteresis; displays read it without changing it.\n\n    Missing/stale data cannot clear an established hold, even at its old ETA.\n    Only a fresh live weekly reading (including a new week) can release it.\n    """\n    def decide():\n        path = directory / \'hold.json\'\n        try:\n            state = json.loads(path.read_text())\n        except FileNotFoundError:\n            state = {}\n        if not isinstance(state, dict) or not isinstance(state.get(bucket, {}), dict):\n            raise ValueError(\'Invalid pacing hold state\')\n        held = dict(state.get(bucket, {}))\n        before = dict(held)\n        # A waiting hook also remembers its own hold. An older process or another\n        # writer clearing the shared file cannot prematurely release this call.\n        if not held and active_hold:\n            held.update(active_hold)\n        weekly = any(w[\'minutes\'] == 10080 and w[\'reset\'] > now for w in windows)\n        if unavailable or (held and not weekly):\n            pause = stale_pause(windows, now)\n            if held:\n                pending = (now + POLL_SEC, \'weekly-lead-unverified\',\n                           \'weekly hold: awaiting fresh lead <=4h\')\n                pause = max(pause, pending) if pause else pending\n        else:\n            pause = evaluate(windows, now, held)\n        if persist and held != before:\n            if held:\n                state[bucket] = held\n            else:\n                state.pop(bucket, None)\n            atomic_json(path, state)\n        if persist and active_hold is not None:\n            active_hold.clear()\n            active_hold.update(held)\n        return pause\n    if not persist:\n        return decide()\n    directory.mkdir(parents=True, exist_ok=True, mode=0o700)\n    with (directory / \'hold.lock\').open(\'a\') as lock:\n        fcntl.flock(lock, fcntl.LOCK_EX)\n        return decide()\n\ndef stale_pause(windows, now):\n    """Old pacing leads are ignored; only known usage >98% can hold until reset."""\n    high = [w for w in windows if isinstance(w, dict)\n            and isinstance(w.get(\'used\'), (int,float)) and HARD_PCT < w[\'used\'] <= 100\n            and isinstance(w.get(\'reset\'), (int,float)) and w[\'reset\'] > now]\n    return evaluate(high, now)\n\ndef bucket_for(model):\n    # Reserve is a fallback bucket, not a replacement for the ordinary quota.\n    return "codex_bengalfox" if "spark" in model.lower() else "codex"\n'}


def main():
    adapter_path, owned_root, mode = sys.argv[1:]
    root = Path(owned_root).resolve(strict=True)
    assert mode in ('quota', 'hang')
    assert os.environ['HERMES_DISABLE_LAZY_INSTALLS'] == '1'
    assert os.environ['HERMES_TEST_ISOLATION'] == '1'
    plan = json.loads((root / 'plan.json').read_text())
    spec = importlib.util.spec_from_file_location('owned_native_adapter', adapter_path)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    sources = {}
    for kind, code in PINNED_CODE.items():
        path = root / (kind + '.py')
        path.write_text(code)
        sources[kind] = (str(path), hashlib.sha256(code.encode()).hexdigest())
    adapter._SOURCES = sources
    evaluate = adapter._evaluate_native

    def get(url, headers):
        bearer = headers['Authorization'].removeprefix('Bearer ')
        matched = hashlib.sha256(bearer.encode()).hexdigest() == plan['credential_sha256']
        assert matched
        assert url == 'https://chatgpt.com/backend-api/wham/usage'
        (root / 'observed.json').write_text(json.dumps({
            'pid': os.getpid(), 'selected_bearer_matched': matched,
            'isolation': os.environ['HERMES_TEST_ISOLATION'],
            'lazy_installs': os.environ['HERMES_DISABLE_LAZY_INSTALLS']}))
        if mode == 'hang':
            while True:
                signal.pause()
        return plan['usage']

    def owned_evaluate(request):
        # Neither configured native stores nor bearer data can select code here.
        assert request['provider'] == 'openai-codex'
        for key in ('state_dir', 'cache_dir'):
            assert Path(request[key]).resolve().is_relative_to(root.parent)
        return evaluate(request, load=adapter._native, get=get)

    adapter._evaluate_native = owned_evaluate
    adapter._child_main()


if __name__ == '__main__':
    main()
