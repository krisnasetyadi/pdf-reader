# telegram_crypto.py
"""
Encrypt/decrypt Telegram secrets at rest: the Telethon session string (full
account access — read every chat, send messages as that user) AND the
per-connection api_hash (an application credential, entered per-connection
rather than via server config — see models.TelegramConnectStartRequest).
Neither may ever be written to logs or API responses in plaintext.

Uses Fernet (AES-128-CBC + HMAC, from the `cryptography` package) with a key
from TELEGRAM_SESSION_ENCRYPTION_KEY. Generate one with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from functools import lru_cache
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from config import config


class SessionEncryptionUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    key = config.telegram_session_encryption_key
    if not key:
        raise SessionEncryptionUnavailable(
            "TELEGRAM_SESSION_ENCRYPTION_KEY is not set — cannot encrypt/decrypt Telegram secrets"
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt_secret(encrypted: str) -> Optional[str]:
    try:
        return _fernet().decrypt(encrypted.encode()).decode()
    except InvalidToken:
        return None


# Back-compat aliases (session string is just one kind of secret we encrypt).
encrypt_session = encrypt_secret
decrypt_session = decrypt_secret
