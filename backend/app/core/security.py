from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from app.core.config import settings
from app.core.constants import ErrorCode
from app.core.errors import AppError


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        f"{settings.password_salt}:{salt}".encode("utf-8"),
        240_000,
    )
    return f"pbkdf2_sha256${salt}${_b64(digest)}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        method, salt, digest = password_hash.split("$", 2)
    except ValueError:
        return False
    if method != "pbkdf2_sha256":
        return False
    expected = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        f"{settings.password_salt}:{salt}".encode("utf-8"),
        240_000,
    )
    return hmac.compare_digest(_b64(expected), digest)


def hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_token(subject: str, tenant_id: str, token_type: str, expires_in_seconds: int, extra: dict[str, Any] | None = None) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": settings.jwt_issuer,
        "sub": subject,
        "tenant_id": tenant_id,
        "type": token_type,
        "iat": now,
        "exp": now + expires_in_seconds,
        "jti": secrets.token_hex(16),
        **(extra or {}),
    }
    signing_input = f"{_b64(json.dumps(header, separators=(',', ':')).encode())}.{_b64(json.dumps(payload, separators=(',', ':')).encode())}"
    signature = hmac.new(settings.jwt_secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64(signature)}"


def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any]:
    try:
        signing_input, signature = token.rsplit(".", 1)
        expected = hmac.new(settings.jwt_secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64(expected), signature):
            raise ValueError("signature mismatch")
        payload_raw = signing_input.split(".", 1)[1]
        payload = json.loads(_unb64(payload_raw))
    except Exception as exc:
        raise AppError(ErrorCode.UNAUTHORIZED, "invalid token", 401) from exc
    if payload.get("iss") != settings.jwt_issuer or int(payload.get("exp", 0)) < int(time.time()):
        raise AppError(ErrorCode.UNAUTHORIZED, "expired token", 401)
    if expected_type and payload.get("type") != expected_type:
        raise AppError(ErrorCode.UNAUTHORIZED, "unexpected token type", 401)
    return payload

