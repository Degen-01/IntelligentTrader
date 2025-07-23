"""Security and encryption utilities."""

import os
import base64
import hashlib
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class SecureConfig:
    def __init__(self, master_key: str = None):
        self.master_key = master_key or os.getenv('MASTER_KEY')
        if not self.master_key:
            raise ValueError("MASTER_KEY environment variable required")
        
        self.cipher_suite = self._create_cipher_suite()

    def _create_cipher_suite(self) -> Fernet:
        """Create cipher suite from master key."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'trading_bot_salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return Fernet(key)

    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret value."""
        encrypted = self.cipher_suite.encrypt(secret.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt a secret value."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_secret.encode())
            decrypted = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            raise

    def get_secure_config(self, key: str, encrypted: bool = True) -> str:
        """Get configuration value, decrypting if needed."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Configuration key {key} not found")
        
        return self.decrypt_secret(value) if encrypted else value

class APISignature:
    @staticmethod
    def create_signature(secret: str, message: str, algorithm: str = 'sha256') -> str:
        """Create HMAC signature for API authentication."""
        return hmac.new(
            secret.encode(),
            message.encode(),
            getattr(hashlib, algorithm)
        ).hexdigest()

    @staticmethod
    def verify_signature(secret: str, message: str, signature: str) -> bool:
        """Verify HMAC signature."""
        expected = APISignature.create_signature(secret, message)
        return hmac.compare_digest(expected, signature)

# Global security instance
secure_config = SecureConfig()
