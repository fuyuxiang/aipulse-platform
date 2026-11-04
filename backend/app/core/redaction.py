"""Redact credential values at serialization and audit boundaries."""
from __future__ import annotations
from typing import Any


def redact_credentials(value: Any) -> Any:
    if isinstance(value, list):
        return [redact_credentials(item) for item in value]
    if not isinstance(value, dict):
        return value
    secret_keys = {'api_key', 'apikey', 'secret', 'secret_value', 'password', 'access_token', 'refresh_token', 'authorization', 'webhook_secret', 'bot_token', 'signing_secret', 'app_secret', 'encoding_aes_key'}
    return {key: ('***' if str(key).lower() in secret_keys and item else redact_credentials(item)) for key, item in value.items()}


def restore_redacted_credentials(value: Any, previous: Any) -> Any:
    """Keep server-held credentials when an editor submits the masked read value."""
    if isinstance(value, list):
        old_items = previous if isinstance(previous, list) else []
        return [restore_redacted_credentials(item, old_items[index] if index < len(old_items) else None) for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    old = previous if isinstance(previous, dict) else {}
    restored = {}
    for key, item in value.items():
        if item == '***' and key in old and redact_credentials({key: old[key]}).get(key) == '***':
            restored[key] = old[key]
        else:
            restored[key] = restore_redacted_credentials(item, old.get(key))
    return restored
