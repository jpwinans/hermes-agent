"""Admission must accept the client production builds, not only a fake of it."""
import base64
import json

from agent import auxiliary_client as aux
from agent import quota_pacing_aux as pacing


def _jwt():
    def encode(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()
    claims = {"exp": 4102444800, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}}
    return ".".join([encode({"alg": "none"}), encode(claims), "sig"])


def test_production_codex_aux_client_is_not_transport_blocked(monkeypatch):
    token = _jwt()
    monkeypatch.setattr(aux, "_resolve_codex_credential_and_base",
                        lambda *a, **k: (token, "https://chatgpt.com/backend-api/codex"))
    client, model = aux._build_codex_client("gpt-test")

    selected = pacing._snapshot(client, {"model": model, "messages": []}, "openai-codex")

    assert selected.problem == ""
    assert selected.credential == token
