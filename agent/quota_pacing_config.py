"""Pacing-only config read: reuse native managed locations, never fail unmetered."""
from __future__ import annotations

from typing import Any


def section() -> dict[str, Any]:
    from hermes_cli.config import _expand_env_vars, load_config
    from hermes_cli.config_read_errors import FailedConfigRead
    from hermes_cli.managed_scope import get_managed_dir
    from utils import fast_safe_load

    try:
        directory = get_managed_dir()
        if directory is not None:
            try:
                raw = fast_safe_load((directory / "config.yaml").read_text(encoding="utf-8-sig"))
            except FileNotFoundError:
                raw = {}
            if raw is None:  # Empty/comment-only YAML is an absent operator policy.
                raw = {}
            # A malformed/unreadable operator document cannot silently erase an
            # enforced policy. Unlike the general managed layer, pacing fails closed.
            if not isinstance(raw, dict):
                return {"enabled": True, "invalid": True}
            if "quota_pacing" in raw:
                managed = raw["quota_pacing"]
                if not isinstance(managed, dict) or "enabled" not in managed:
                    return {"enabled": True, "invalid": True}
                # The operator's whole pacing section is authoritative. A profile
                # cannot append an unmetered escape beside managed account rows.
                expanded = _expand_env_vars(managed)
                return expanded if isinstance(expanded, dict) else {"enabled": True, "invalid": True}
        config = load_config()
        if isinstance(config, FailedConfigRead):
            return {"enabled": True, "invalid": True}
        value = config.get("quota_pacing", {})
        return value if isinstance(value, dict) else {"enabled": True, "invalid": True}
    except Exception:
        return {"enabled": True, "invalid": True}
