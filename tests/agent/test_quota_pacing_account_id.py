"""A Codex OAuth refresh rotates the bearer but keeps the ChatGPT account."""
import base64
import hashlib
import json

from agent import quota_pacing_native as adapter

ENDPOINT = next(iter(adapter._ENDPOINTS['openai-codex']))


def codex_token(account_id, nonce):
    def encode(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b'=').decode()
    claims = {'nonce': nonce, 'https://api.openai.com/auth': {'chatgpt_account_id': account_id}}
    return '.'.join([encode({'alg': 'none'}), encode(claims), 'sig'])


def section(tmp_path, bound_token):
    row = {'provider': 'openai-codex', 'account': 'owned-account', 'chatgpt_account_id': 'acct-owned',
           'state_dir': str(tmp_path / 'state')}
    return {'enabled': True, 'accounts': [
        {**row, 'credential_sha256': hashlib.sha256(bound_token.encode()).hexdigest()},
        # Operators may repeat the id on every row bound to the same account.
        {**row, 'credential_sha256': '0' * 64},
    ]}


def test_refreshed_token_of_the_bound_account_stays_bound(tmp_path):
    original = codex_token('acct-owned', 'before-refresh')
    refreshed = codex_token('acct-owned', 'after-refresh')

    row, reason = adapter._binding_row(section(tmp_path, original), 'openai-codex', 'gpt-test',
                                       refreshed, ENDPOINT)

    assert (reason, row['account']) == ('', 'owned-account')


def test_token_of_another_account_still_needs_repair(tmp_path):
    original = codex_token('acct-owned', 'before-refresh')
    other = codex_token('acct-other', 'other-login')

    _, reason = adapter._binding_row(section(tmp_path, original), 'openai-codex', 'gpt-test',
                                     other, ENDPOINT)

    assert reason == 'account_repair'
