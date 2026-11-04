"""Monthly API-credit pacing: spend counted from Hermes's own usage rows, paced against the cycle, floor kept."""
import json
import sqlite3
from datetime import datetime

import pytest

from agent import quota_pacing_native as native


def db(tmp_path, rows, name="state.db"):
    path = tmp_path / name
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE session_model_usage (session_id TEXT, model TEXT, billing_provider TEXT, "
              "billing_base_url TEXT, task TEXT, estimated_cost_usd REAL, actual_cost_usd REAL, last_seen REAL)")
    c.executemany("INSERT INTO session_model_usage VALUES (?,?,?,?,?,?,?,?)", rows)
    c.commit()
    c.close()
    return str(path)


def ts(*args):
    return datetime(*args).timestamp()


def row(cost, when, provider="anthropic", url="https://api.anthropic.com", actual=None):
    return ("s", "claude-opus-5-5", provider, url, "", cost, actual, when)


def request(tmp_path, dbs, active_hold=None):
    return {"provider": "anthropic", "model": "claude-opus-5-5", "state_dir": str(tmp_path / "pacing"),
            "label": "Anthropic API credits", "active_hold": active_hold or {},
            "credits": {"monthly_usd": 200, "floor_usd": 25, "renews_day": 23,
                        "trigger_hours": 36, "resume_hours": 18, "state_dbs": dbs}}


@pytest.mark.parametrize("now, start, end", [  # local times, converted inside the test's own clock
    ((2026, 11, 3, 12), (2026, 10, 23), (2026, 11, 23)),
    ((2026, 11, 23, 0, 0, 1), (2026, 11, 23), (2026, 12, 23)),
    ((2027, 1, 5), (2026, 12, 23), (2027, 1, 23)),
])
def test_cycle_runs_from_renewal_to_renewal(now, start, end):
    assert native._cycle(23, ts(*now)) == (ts(*start), ts(*end))


def test_cycle_clamps_a_renewal_day_past_month_end():
    assert native._cycle(31, ts(2027, 3, 1)) == (ts(2027, 2, 28), ts(2027, 3, 31))


def test_under_pace_allows(tmp_path):
    now = ts(2026, 11, 3, 12)  # about a third of the way through the cycle
    out = native._evaluate_credits(request(tmp_path, [db(tmp_path, [row(20.0, now - 3600)])]), now)
    assert out["action"] == "allow"


def test_counts_only_this_cycle_on_the_api_taking_the_larger_cost(tmp_path):
    now = ts(2026, 11, 3, 12)
    rows = [row(100.0, ts(2026, 10, 22)),                                  # previous cycle
            row(100.0, now, provider="openai-codex", url="https://chatgpt.com"),  # another provider
            row(5.0, now, actual=7.0), row(1.0, now),
            row(0.0025, now, actual=0.0)]  # Hermes stores 0.0, not NULL, when only an estimate exists
    assert native._credits_spent([db(tmp_path, rows)], ts(2026, 10, 23)) == pytest.approx(8.0025)


def test_ahead_of_pace_pauses_with_the_message_and_latches(tmp_path):
    now = ts(2026, 11, 3, 12)
    dbs = [db(tmp_path, [row(120.0, now)])]  # well ahead of an even burn of $175
    out = native._evaluate_credits(request(tmp_path, dbs), now)
    assert out["action"] == "hold" and out["reason"] == "native_hold"
    assert out["note"].startswith("Pacing for Anthropic API credits is +")
    assert "hours over the pace. Pausing for about" in out["note"] and "then continuing automatically" in out["note"]
    assert json.loads((tmp_path / "pacing" / "credits-hold.json").read_text())["cycle"] == ts(2026, 10, 23)


def test_latch_holds_until_back_to_resume_point_then_releases(tmp_path):
    start, end = ts(2026, 10, 23), ts(2026, 11, 23)
    budget, window_h = 175.0, (end - start) / 3600
    now = ts(2026, 11, 3, 12)
    ideal = budget * (now - start) / (end - start)
    path = tmp_path / "pacing"
    path.mkdir()
    (path / "credits-hold.json").write_text(json.dumps({"cycle": start}))
    for lead_h, action in ((24, "hold"), (10, "allow")):  # between resume and trigger: still held; below: released
        dbs = [db(tmp_path, [row(ideal + budget * lead_h / window_h, now)], name=f"{lead_h}.db")]
        assert native._evaluate_credits(request(tmp_path, dbs), now)["action"] == action
    assert not (path / "credits-hold.json").exists() or json.loads((path / "credits-hold.json").read_text()) == {}


def test_floor_reached_pauses_until_renewal(tmp_path):
    now = ts(2026, 11, 20)
    out = native._evaluate_credits(request(tmp_path, [db(tmp_path, [row(175.5, now)])]), now)
    assert out["action"] == "hold"
    assert "$175.50 of the $175 budget spent this cycle, keeping $25 in reserve" in out["note"]
    assert "until they renew" in out["note"]


def test_child_routes_credit_rows_to_the_credit_meter(tmp_path):
    now_rows = [row(1.0, datetime.now().timestamp())]
    out = native._evaluate_native(request(tmp_path, [db(tmp_path, now_rows)]))
    assert out["action"] == "allow" and out["reason"] == "native_allow"


def test_binding_passes_credit_settings_to_the_child():
    import hashlib
    section = {"enabled": True, "accounts": [{
        "provider": "anthropic", "account": "anthropic-api-credits",
        "credential_sha256": hashlib.sha256(b"sk-ant-test").hexdigest(),
        "state_dir": "/tmp/p", "cache_dir": "/tmp/p",
        "credits": {"monthly_usd": 200, "floor_usd": 25, "renews_day": 23,
                    "trigger_hours": 36, "resume_hours": 18, "state_dbs": ["/tmp/x.db"]}}]}
    binding = native.bind_request(section, provider="anthropic", model="claude-opus-5-5",
                                  credential="sk-ant-test", base_url="https://api.anthropic.com")
    sent = binding._request(binding.route, None, 5.0)
    assert sent["credits"]["renews_day"] == 23 and sent["label"] == "Anthropic API credits"
