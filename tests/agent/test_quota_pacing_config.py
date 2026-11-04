"""Owned managed-policy configuration contracts; no live /etc or credentials."""
import json
from contextlib import contextmanager

import pytest

from agent import quota_pacing_host as host
from hermes_cli import managed_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def managed(tmp_path, monkeypatch):
    directory = tmp_path / 'owned-managed'
    directory.mkdir()
    monkeypatch.setenv('HERMES_MANAGED_DIR', str(directory))
    managed_scope.invalidate_managed_cache()
    yield directory
    managed_scope.invalidate_managed_cache()


@contextmanager
def profile(path, section):
    path.mkdir()
    (path / 'config.yaml').write_text(json.dumps({'quota_pacing': section}))
    token = set_hermes_home_override(path)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_managed_policy_wins_in_two_profiles(managed, tmp_path):
    policy = {'enabled': True, 'accounts': [], 'unmetered': []}
    (managed / 'config.yaml').write_text(json.dumps({'quota_pacing': policy}))
    for index in range(2):
        with profile(tmp_path / str(index), {'enabled': False, 'unmetered': [
                {'provider': 'anthropic', 'base_url': 'https://api.anthropic.com'}]}):
            assert host._section() == policy


def test_missing_profile_policy_inherits_managed(managed, tmp_path):
    (managed / 'config.yaml').write_text(json.dumps({'quota_pacing': {'enabled': True}}))
    with profile(tmp_path / 'profile', {}):
        assert host._section()['enabled'] is True


def test_profile_cannot_append_unmetered_escape_to_managed_accounts(managed, tmp_path):
    from hermes_cli.config import load_config
    policy = {'enabled': True, 'accounts': []}
    escape = [{'provider': 'anthropic', 'base_url': 'https://api.anthropic.com'}]
    (managed / 'config.yaml').write_text(json.dumps({'quota_pacing': policy}))
    with profile(tmp_path / 'profile', {'enabled': False, 'unmetered': escape}):
        assert load_config()['quota_pacing']['unmetered'] == escape  # General native overlay is per leaf.
        assert host._section() == policy  # Pacing's operator section is authoritative as a whole.


@pytest.mark.parametrize('raw', ['quota_pacing: [', '["wrong-root-shape"]'])
def test_malformed_managed_policy_never_becomes_unmetered(managed, tmp_path, raw):
    (managed / 'config.yaml').write_text(raw)
    with profile(tmp_path / 'profile', {'enabled': False}):
        section = host._section()
        assert not host._disabled(section), 'unreadable operator policy silently disabled quota pacing'
        assert section.get('invalid') is True


def test_valid_unrelated_managed_settings_preserve_stock_disabled(managed, tmp_path):
    (managed / 'config.yaml').write_text(json.dumps({'display': {'skin': 'default'}}))
    with profile(tmp_path / 'profile', {}):
        assert host._disabled(host._section())


def test_explicit_operator_disable_is_respected(managed, tmp_path):
    (managed / 'config.yaml').write_text(json.dumps({'quota_pacing': {'enabled': False}}))
    with profile(tmp_path / 'profile', {'enabled': True}):
        assert host._disabled(host._section())


@pytest.mark.parametrize('policy', [{}, {'accounts': []}])
def test_managed_section_must_explicitly_state_enabled(managed, tmp_path, policy):
    (managed / 'config.yaml').write_text(json.dumps({'quota_pacing': policy}))
    with profile(tmp_path / 'profile', {'enabled': True}):
        section = host._section()
        assert not host._disabled(section), 'incomplete operator policy silently disabled pacing'
        assert section.get('invalid') is True


@pytest.mark.parametrize('raw', ['', '# unrelated operator comments only\n'])
@pytest.mark.parametrize('enabled', [False, True])
def test_empty_managed_document_preserves_profile_policy(managed, tmp_path, raw, enabled):
    (managed / 'config.yaml').write_text(raw)
    policy = {'enabled': enabled, 'accounts': []}
    with profile(tmp_path / 'profile', policy):
        assert host._section() == policy
