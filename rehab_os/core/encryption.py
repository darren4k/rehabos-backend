"""
RehabOS - Field-Level PHI Encryption

AES-256-GCM encryption for PHI fields in the database.
Compliant with HIPAA requirements for encryption at rest.

Usage:
    from rehab_os.core.encryption import EncryptedString, HashedString

    # In models:
    first_name: Mapped[str] = mapped_column(EncryptedString(100))
    ssn_hash: Mapped[str | None] = mapped_column(HashedString(64))
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import String, TypeDecorator

logger = logging.getLogger(__name__)

ENCRYPTION_KEY_ENV = "REHAB_ENCRYPTION_KEY"
KEY_DERIVATION_SALT = b"rehab_os_phi_encryption_v1"


def get_encryption_key() -> bytes:
    """Get 32-byte AES-256 key from environment.

    In production, use AWS KMS / Azure Key Vault / HashiCorp Vault.
    Never hardcode keys.
    """
    key_material = os.environ.get(ENCRYPTION_KEY_ENV)

    if key_material:
        return hashlib.pbkdf2_hmac(
            "sha256",
            key_material.encode("utf-8"),
            KEY_DERIVATION_SALT,
            iterations=100000,
            dklen=32,
        )

    # Dev fallback -- DO NOT use in production
    logger.warning(
        "%s not set. Using dev fallback key. "
        "Set %s in production.",
        ENCRYPTION_KEY_ENV,
        ENCRYPTION_KEY_ENV,
    )
    return hashlib.pbkdf2_hmac(
        "sha256",
        b"DEVELOPMENT_ONLY_NOT_FOR_PRODUCTION",
        KEY_DERIVATION_SALT,
        iterations=100000,
        dklen=32,
    )


def encrypt_value(plaintext: str, key: bytes) -> str:
    """Encrypt a string with AES-256-GCM. Returns base64(nonce + ciphertext + tag)."""
    if not plaintext:
        return ""
    nonce = secrets.token_bytes(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_value(encrypted: str, key: bytes) -> str:
    """Decrypt a value produced by encrypt_value."""
    if not encrypted:
        return ""
    try:
        combined = base64.b64decode(encrypted.encode("ascii"))
        nonce = combined[:12]
        ciphertext = combined[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
    except Exception:
        logger.error("Decryption failed for PHI field")
        return ""


class EncryptedString(TypeDecorator):
    """SQLAlchemy type that auto-encrypts on write and decrypts on read.

    The length param is the max plaintext length; storage will be larger.
    Encrypted fields cannot be used in WHERE clauses or indexes.
    """

    impl = String
    cache_ok = True

    def __init__(self, length: Optional[int] = None, **kwargs: Any):
        # base64(nonce 12 + plaintext + tag 16) ~ (length+28)*4/3 + padding
        if length:
            encrypted_length = int((length + 28) * 1.5) + 10
        else:
            encrypted_length = 500
        super().__init__(length=encrypted_length, **kwargs)
        self._plaintext_length = length
        self._key: Optional[bytes] = None

    def _get_key(self) -> bytes:
        if self._key is None:
            self._key = get_encryption_key()
        return self._key

    def process_bind_param(self, value: Optional[str], dialect: Any) -> Optional[str]:
        if value is None:
            return None
        return encrypt_value(value, self._get_key())

    def process_result_value(self, value: Optional[str], dialect: Any) -> Optional[str]:
        if value is None:
            return None
        return decrypt_value(value, self._get_key())


class HashedString(TypeDecorator):
    """One-way SHA-256 hash for searchable PHI fields (e.g., SSN lookup).

    Cannot be decrypted -- use verify() to compare.
    """

    impl = String
    cache_ok = True

    def __init__(self, length: int = 64, **kwargs: Any):
        super().__init__(length=length, **kwargs)

    def process_bind_param(self, value: Optional[str], dialect: Any) -> Optional[str]:
        if value is None:
            return None
        salted = KEY_DERIVATION_SALT + value.encode("utf-8")
        return hashlib.sha256(salted).hexdigest()

    def process_result_value(self, value: Optional[str], dialect: Any) -> Optional[str]:
        return value

    @staticmethod
    def verify(plaintext: str, hashed: str) -> bool:
        """Verify a plaintext value against its stored hash."""
        salted = KEY_DERIVATION_SALT + plaintext.encode("utf-8")
        computed = hashlib.sha256(salted).hexdigest()
        return secrets.compare_digest(computed, hashed)
