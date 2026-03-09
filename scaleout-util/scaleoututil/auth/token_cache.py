"""Token cache manager for persistent token storage."""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

from scaleoututil.logging import ScaleoutLogger


class TokenCache:
    """Manages persistent storage of tokens in the user's home directory.

    Tokens are stored in ~/.scaleout/tokens/{cache_id}.json with the following structure:
    {
        "access_token": "...",
        "refresh_token": "...",
        "expires_at": "ISO-8601 timestamp",
        "updated_at": "ISO-8601 timestamp"
    }

    Security Note:
        Tokens are stored UNENCRYPTED in JSON files with 0600 permissions.
        While this restricts access to the file owner, it does not protect against:
        - Root/admin access
        - Backup systems that may not preserve permissions
        - Malware running as the same user
        - Forensic analysis if disk is compromised

        Set SCALEOUT_PERSIST_TOKENS=false to disable token persistence.
    """

    _security_warning_shown = False  # Class variable to show warning once per process

    def __init__(self, cache_id: str, cache_dir: Optional[str] = None):
        """Initialize token cache.

        Args:
            cache_id: Unique identifier for the cache file (e.g., client_id or "api-client")
            cache_dir: Optional custom cache directory (defaults to ~/.scaleout/tokens)

        """
        self.cache_id = cache_id
        self._cache_available = False  # Track if cache directory is usable

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use SCALEOUT_HOME_DIR if set, otherwise fall back to user's home directory
            home_dir = os.environ.get("SCALEOUT_HOME_DIR")
            if home_dir:
                home = Path(home_dir)
            else:
                home = Path.home()
            self.cache_dir = home / ".scaleout" / "tokens"

        # Try to ensure cache directory exists
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Set restrictive permissions on the cache directory (owner read/write/execute only)
            try:
                os.chmod(self.cache_dir, 0o700)
            except Exception as e:
                ScaleoutLogger().warning(f"Could not set restrictive permissions on token cache directory: {e}")

            self._cache_available = True
            self.cache_file = self.cache_dir / f"{cache_id}.json"

        except (PermissionError, OSError) as e:
            ScaleoutLogger().warning(
                f"Token cache directory unavailable ({e}). Token caching disabled. Set SCALEOUT_TOKEN_CACHE_DIR to a writable directory to enable caching."
            )
            self.cache_file = None

    def load(self) -> Optional[Dict[str, Any]]:
        """Load tokens from cache file.

        Returns:
            Dictionary with token data or None if cache doesn't exist or is invalid

        """
        if not self._cache_available or not self.cache_file:
            return None

        if not self.cache_file.exists():
            ScaleoutLogger().debug(f"No token cache found at {self.cache_file}")
            return None

        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)

            ScaleoutLogger().debug(f"Loaded tokens from cache: {self.cache_file}")
            return data
        except Exception as e:
            ScaleoutLogger().warning(f"Failed to load token cache: {e}")
            return None

    def save(self, access_token: str, refresh_token: str, expires_at: Optional[datetime] = None) -> None:
        """Save tokens to cache file.

        Args:
            access_token: Access token to store
            refresh_token: Refresh token to store
            expires_at: Optional expiration datetime (UTC)

        """
        # Check if cache directory is available
        if not self._cache_available or not self.cache_file:
            ScaleoutLogger().debug("Token cache unavailable, skipping save")
            return

        # Check if token persistence is disabled
        if os.environ.get("SCALEOUT_PERSIST_TOKENS", "true").lower() in ("false", "0", "no"):
            ScaleoutLogger().debug("Token persistence disabled via SCALEOUT_PERSIST_TOKENS environment variable")
            return

        # Show security warning once per process
        if not TokenCache._security_warning_shown:
            ScaleoutLogger().warning(
                f"Storing tokens unencrypted in {self.cache_file}. Set SCALEOUT_PERSIST_TOKENS=false to disable. See documentation for security considerations."
            )
            TokenCache._security_warning_shown = True

        data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Create secure temporary file with 0600 permissions atomically
            # mkstemp() creates the file with 0600 before any data is written
            fd, temp_path = tempfile.mkstemp(dir=self.cache_dir, prefix=f".{self.cache_id}_", suffix=".tmp")

            try:
                # Write token data to the secure temp file
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data is written to disk

                # Atomically replace old file with new file
                os.replace(temp_path, self.cache_file)

                ScaleoutLogger().debug(f"Saved tokens to cache: {self.cache_file}")
            except Exception:
                # Clean up temp file if something went wrong
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                raise

        except Exception as e:
            ScaleoutLogger().error(f"Failed to save token cache: {e}")
            raise

    def clear(self) -> None:
        """Remove cached tokens."""
        if not self._cache_available or not self.cache_file:
            return

        if self.cache_file.exists():
            try:
                self.cache_file.unlink()
                ScaleoutLogger().info(f"Cleared token cache: {self.cache_file}")
            except Exception as e:
                ScaleoutLogger().warning(f"Failed to clear token cache: {e}")

    def get_access_token(self) -> Optional[str]:
        """Get access token from cache.

        Returns:
            Access token or None if not found

        """
        data = self.load()
        return data.get("access_token") if data else None

    def get_refresh_token(self) -> Optional[str]:
        """Get refresh token from cache.

        Returns:
            Refresh token or None if not found

        """
        data = self.load()
        return data.get("refresh_token") if data else None

    def is_access_token_valid(self) -> bool:
        """Check if cached access token is still valid (not expired).

        Returns:
            True if access token exists and is not expired, False otherwise

        """
        data = self.load()
        if not data or not data.get("access_token"):
            return False

        expires_at_str = data.get("expires_at")
        if not expires_at_str:
            return False

        try:
            expires_at = datetime.fromisoformat(expires_at_str)
            # Add a small buffer (10 seconds) to account for clock skew
            return datetime.now(timezone.utc) < (expires_at - timedelta(seconds=10))
        except Exception as e:
            ScaleoutLogger().warning(f"Failed to parse token expiration: {e}")
            return False

    def get_token_data(self) -> Optional[Dict[str, Any]]:
        """Get full token data including metadata.

        Returns:
            Dictionary with all token data including timestamps, or None if not found

        """
        return self.load()

    def exists(self) -> bool:
        """Check if token cache exists.

        Returns:
            True if cache file exists, False otherwise

        """
        if not self._cache_available or not self.cache_file:
            return False
        return self.cache_file.exists()
