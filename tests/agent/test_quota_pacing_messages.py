"""What James sees: one line for a real pause, nothing for a routine check or a meter fault."""
import json

import pytest

from agent import quota_pacing_native as native
from agent.quota_pacing import Decision, PendingAttempt, Route, Lifetime
from agent.session_activity import quota_wait_message

NOW = 1_000_000.0


@pytest.mark.parametrize("provider, pause, expected", [
    ("openai-codex", (NOW + 45 * 60, "weekly-lead", "weekly 9.2h ahead of pace"),
     "Pacing for OpenAI Codex is +9.2 hours over the pace. Pausing for about 45 minutes, then continuing automatically."),
    ("anthropic", (NOW + 300 * 60, "weekly-lead", "Opus weekly 8.0h ahead of pace"),
     "Pacing for Anthropic Claude is +8.0 hours over the pace. Pausing for about 300 minutes (5.0 hours), then continuing automatically."),
    ("openai-codex", (NOW + 90 * 60, "weekly-hard", "weekly 99%, until reset"),
     "Pacing for OpenAI Codex: weekly usage is 99%, over the limit. Pausing for about 90 minutes until it resets, then continuing automatically."),
    ("openai-codex", (NOW + 5, "weekly-lead-unverified", "weekly hold: awaiting fresh lead <=4h"),
     "Pacing for OpenAI Codex is holding until a fresh usage reading shows it back on pace. Checking again shortly, then continuing automatically."),
])
def test_pause_note(provider, pause, expected):
    assert native._note(provider, pause, NOW) == expected


def test_note_crosses_the_child_boundary_and_is_shown():
    note = native._note("openai-codex", (NOW + 600, "weekly-lead", "weekly 9.0h ahead of pace"), NOW)
    decision = native._Binding._decision(json.dumps(
        {"action": "hold", "reason": "native_hold", "retry_after": 5, "note": note}).encode(), None)
    assert decision.note == note
    assert quota_wait_message(decision) == note
    assert "repair" not in quota_wait_message(decision).lower()


def test_unprintable_note_is_dropped_not_shown():
    route = Route("openai-codex", "m", "a")
    pending = PendingAttempt(route=route, lifetime=Lifetime(), current_route=lambda: route, on_waiting=lambda d: None)
    assert pending._accept(Decision("hold", "native_hold", note="bad\x1b[2Jline")).note == ""


def test_no_message_mentions_repair():
    for decision in (Decision("hold", "native_hold"), Decision("blocked", "credential_repair"),
                     Decision("blocked", "transport_snapshot_required")):
        assert "repair" not in quota_wait_message(decision).lower()
