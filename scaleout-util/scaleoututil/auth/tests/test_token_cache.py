"""Unit tests for TokenCache."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from scaleoututil.auth.token_cache import TokenCache


class TestTokenCache(unittest.TestCase):
    """Test cases for TokenCache class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.cache_id = "test-client"
        self.access_token = "test-access-token"
        self.refresh_token = "test-refresh-token"
        self.expires_at = datetime.now(timezone.utc)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init_creates_cache_directory(self):
        """Test TokenCache initialization creates cache directory."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        self.assertTrue(cache.cache_dir.exists())
        self.assertTrue(cache.cache_dir.is_dir())
        self.assertEqual(cache.cache_id, self.cache_id)
        self.assertEqual(cache.cache_file, Path(self.temp_dir) / f"{self.cache_id}.json")

    def test_init_with_default_cache_dir(self):
        """Test TokenCache initialization with default cache directory."""
        cache = TokenCache(cache_id=self.cache_id)

        expected_dir = Path.home() / ".scaleout" / "tokens"
        self.assertEqual(cache.cache_dir, expected_dir)

    def test_init_sets_directory_permissions(self):
        """Test TokenCache sets restrictive permissions on cache directory."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Check directory permissions (should be 0700 on Unix-like systems)
        if os.name != 'nt':  # Skip on Windows
            dir_stat = os.stat(cache.cache_dir)
            # Get the last 3 octal digits (permissions)
            permissions = oct(dir_stat.st_mode)[-3:]
            self.assertEqual(permissions, '700')

    def test_save_creates_cache_file(self):
        """Test save() creates cache file with correct content."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        cache.save(self.access_token, self.refresh_token, self.expires_at)

        self.assertTrue(cache.cache_file.exists())

        # Verify file content
        with open(cache.cache_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(data["access_token"], self.access_token)
        self.assertEqual(data["refresh_token"], self.refresh_token)
        self.assertEqual(data["expires_at"], self.expires_at.isoformat())
        self.assertIn("updated_at", data)

    def test_save_sets_file_permissions(self):
        """Test save() sets restrictive permissions on cache file."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        cache.save(self.access_token, self.refresh_token, self.expires_at)

        # Check file permissions (should be 0600 on Unix-like systems)
        if os.name != 'nt':  # Skip on Windows
            file_stat = os.stat(cache.cache_file)
            permissions = oct(file_stat.st_mode)[-3:]
            self.assertEqual(permissions, '600')

    def test_save_without_expires_at(self):
        """Test save() works without expires_at parameter."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        cache.save(self.access_token, self.refresh_token)

        with open(cache.cache_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(data["access_token"], self.access_token)
        self.assertEqual(data["refresh_token"], self.refresh_token)
        self.assertIsNone(data["expires_at"])

    def test_save_overwrites_existing_cache(self):
        """Test save() overwrites existing cache file."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Save initial tokens
        cache.save("old-access", "old-refresh")

        # Save new tokens
        cache.save(self.access_token, self.refresh_token)

        # Verify new tokens are saved
        with open(cache.cache_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(data["access_token"], self.access_token)
        self.assertEqual(data["refresh_token"], self.refresh_token)

    def test_load_returns_none_when_cache_not_exists(self):
        """Test load() returns None when cache file doesn't exist."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        result = cache.load()

        self.assertIsNone(result)

    def test_load_returns_cached_data(self):
        """Test load() returns cached data."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Save data first
        cache.save(self.access_token, self.refresh_token, self.expires_at)

        # Load data
        result = cache.load()

        self.assertIsNotNone(result)
        self.assertEqual(result["access_token"], self.access_token)
        self.assertEqual(result["refresh_token"], self.refresh_token)
        self.assertEqual(result["expires_at"], self.expires_at.isoformat())

    def test_load_handles_corrupted_json(self):
        """Test load() handles corrupted JSON gracefully."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Write corrupted JSON to cache file
        cache.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache.cache_file, 'w') as f:
            f.write("{ invalid json")

        result = cache.load()

        self.assertIsNone(result)

    def test_clear_removes_cache_file(self):
        """Test clear() removes cache file."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Create cache file
        cache.save(self.access_token, self.refresh_token)
        self.assertTrue(cache.cache_file.exists())

        # Clear cache
        cache.clear()

        self.assertFalse(cache.cache_file.exists())

    def test_clear_when_cache_not_exists(self):
        """Test clear() doesn't raise error when cache file doesn't exist."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Should not raise error
        cache.clear()

        self.assertFalse(cache.cache_file.exists())

    def test_get_access_token(self):
        """Test get_access_token() retrieves access token."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        cache.save(self.access_token, self.refresh_token)

        result = cache.get_access_token()

        self.assertEqual(result, self.access_token)

    def test_get_access_token_returns_none_when_no_cache(self):
        """Test get_access_token() returns None when cache doesn't exist."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        result = cache.get_access_token()

        self.assertIsNone(result)

    def test_get_refresh_token(self):
        """Test get_refresh_token() retrieves refresh token."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        cache.save(self.access_token, self.refresh_token)

        result = cache.get_refresh_token()

        self.assertEqual(result, self.refresh_token)

    def test_get_refresh_token_returns_none_when_no_cache(self):
        """Test get_refresh_token() returns None when cache doesn't exist."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        result = cache.get_refresh_token()

        self.assertIsNone(result)

    def test_exists_returns_true_when_cache_exists(self):
        """Test exists() returns True when cache file exists."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        cache.save(self.access_token, self.refresh_token)

        self.assertTrue(cache.exists())

    def test_exists_returns_false_when_cache_not_exists(self):
        """Test exists() returns False when cache file doesn't exist."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        self.assertFalse(cache.exists())

    def test_atomic_write(self):
        """Test save() uses atomic write (temp file + rename)."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        cache.save(self.access_token, self.refresh_token)

        # Verify temp file doesn't exist after save
        temp_file = cache.cache_file.with_suffix(".tmp")
        self.assertFalse(temp_file.exists())

        # Verify final file exists
        self.assertTrue(cache.cache_file.exists())

    def test_cache_id_parameter(self):
        """Test TokenCache requires cache_id parameter."""
        cache = TokenCache(cache_id="test-id", cache_dir=self.temp_dir)

        self.assertEqual(cache.cache_id, "test-id")
        self.assertEqual(cache.cache_file.name, "test-id.json")

    def test_save_raises_on_write_error(self):
        """Test save() raises exception on write error."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Make directory read-only to cause write error
        if os.name != 'nt':  # Skip on Windows
            os.chmod(cache.cache_dir, 0o500)

            with self.assertRaises(Exception):
                cache.save(self.access_token, self.refresh_token)

            # Restore permissions for cleanup
            os.chmod(cache.cache_dir, 0o700)

    def test_multiple_cache_instances_with_same_id(self):
        """Test multiple TokenCache instances with same cache_id share cache."""
        cache1 = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)
        cache2 = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Save with cache1
        cache1.save(self.access_token, self.refresh_token)

        # Load with cache2
        result = cache2.get_access_token()

        self.assertEqual(result, self.access_token)

    def test_different_cache_ids_use_separate_files(self):
        """Test different cache_ids use separate cache files."""
        cache1 = TokenCache(cache_id="client-1", cache_dir=self.temp_dir)
        cache2 = TokenCache(cache_id="client-2", cache_dir=self.temp_dir)

        cache1.save("token-1", "refresh-1")
        cache2.save("token-2", "refresh-2")

        result1 = cache1.get_access_token()
        result2 = cache2.get_access_token()

        self.assertEqual(result1, "token-1")
        self.assertEqual(result2, "token-2")

    @patch('scaleoututil.auth.token_cache.ScaleoutLogger')
    def test_logging_on_permission_error(self, mock_logger):
        """Test TokenCache logs warning when permission setting fails."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        with patch('os.chmod', side_effect=PermissionError("Permission denied")):
            cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Verify warning was logged
        mock_logger_instance.warning.assert_called()

    def test_cache_file_path_format(self):
        """Test cache file path has correct format."""
        cache = TokenCache(cache_id="my-test-client", cache_dir=self.temp_dir)

        expected_path = Path(self.temp_dir) / "my-test-client.json"
        self.assertEqual(cache.cache_file, expected_path)

    def test_save_and_load_round_trip(self):
        """Test saving and loading tokens maintains data integrity."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Save tokens
        cache.save(self.access_token, self.refresh_token, self.expires_at)

        # Load tokens
        data = cache.load()

        # Verify all fields
        self.assertEqual(data["access_token"], self.access_token)
        self.assertEqual(data["refresh_token"], self.refresh_token)
        self.assertEqual(data["expires_at"], self.expires_at.isoformat())
        self.assertIn("updated_at", data)

        # Verify updated_at is a valid ISO timestamp
        updated_at = datetime.fromisoformat(data["updated_at"])
        self.assertIsInstance(updated_at, datetime)

    def test_save_disabled_via_env_var_false(self):
        """Test token save is disabled when SCALEOUT_PERSIST_TOKENS=false."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        with patch.dict(os.environ, {"SCALEOUT_PERSIST_TOKENS": "false"}):
            cache.save(self.access_token, self.refresh_token, self.expires_at)

        # Verify no cache file was created
        self.assertFalse(cache.cache_file.exists())

    def test_save_disabled_via_env_var_zero(self):
        """Test token save is disabled when SCALEOUT_PERSIST_TOKENS=0."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        with patch.dict(os.environ, {"SCALEOUT_PERSIST_TOKENS": "0"}):
            cache.save(self.access_token, self.refresh_token, self.expires_at)

        # Verify no cache file was created
        self.assertFalse(cache.cache_file.exists())

    def test_save_disabled_via_env_var_no(self):
        """Test token save is disabled when SCALEOUT_PERSIST_TOKENS=no."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        with patch.dict(os.environ, {"SCALEOUT_PERSIST_TOKENS": "no"}):
            cache.save(self.access_token, self.refresh_token, self.expires_at)

        # Verify no cache file was created
        self.assertFalse(cache.cache_file.exists())

    def test_save_disabled_case_insensitive(self):
        """Test SCALEOUT_PERSIST_TOKENS is case-insensitive."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Test uppercase FALSE
        with patch.dict(os.environ, {"SCALEOUT_PERSIST_TOKENS": "FALSE"}):
            cache.save(self.access_token, self.refresh_token, self.expires_at)

        self.assertFalse(cache.cache_file.exists())

        # Test mixed case No
        with patch.dict(os.environ, {"SCALEOUT_PERSIST_TOKENS": "No"}):
            cache.save(self.access_token, self.refresh_token, self.expires_at)

        self.assertFalse(cache.cache_file.exists())

    def test_save_enabled_by_default(self):
        """Test token save is enabled by default (no env var set)."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SCALEOUT_PERSIST_TOKENS", None)
            cache.save(self.access_token, self.refresh_token, self.expires_at)

        # Verify cache file was created
        self.assertTrue(cache.cache_file.exists())

    def test_save_enabled_via_env_var_true(self):
        """Test token save is enabled when SCALEOUT_PERSIST_TOKENS=true."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        with patch.dict(os.environ, {"SCALEOUT_PERSIST_TOKENS": "true"}):
            cache.save(self.access_token, self.refresh_token, self.expires_at)

        # Verify cache file was created
        self.assertTrue(cache.cache_file.exists())

    def test_security_warning_shown_once(self):
        """Test security warning is shown only once per process."""
        # Reset the class variable before test
        TokenCache._security_warning_shown = False

        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        with patch("scaleoututil.auth.token_cache.ScaleoutLogger") as mock_logger_class:
            mock_logger = MagicMock()
            mock_logger_class.return_value = mock_logger

            # First save should show warning
            cache.save(self.access_token, self.refresh_token, self.expires_at)
            warning_calls_1 = [call for call in mock_logger.warning.call_args_list
                              if "unencrypted" in str(call)]
            self.assertEqual(len(warning_calls_1), 1)

            # Second save should NOT show warning (already shown)
            cache.save(self.access_token + "2", self.refresh_token + "2", self.expires_at)
            warning_calls_2 = [call for call in mock_logger.warning.call_args_list
                              if "unencrypted" in str(call)]
            self.assertEqual(len(warning_calls_2), 1)  # Still only 1 total

    def test_security_warning_mentions_env_var(self):
        """Test security warning mentions the SCALEOUT_PERSIST_TOKENS env var."""
        # Reset the class variable before test
        TokenCache._security_warning_shown = False

        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)

        with patch("scaleoututil.auth.token_cache.ScaleoutLogger") as mock_logger_class:
            mock_logger = MagicMock()
            mock_logger_class.return_value = mock_logger

            cache.save(self.access_token, self.refresh_token, self.expires_at)

            # Verify warning was called
            mock_logger.warning.assert_called()

            # Check warning message content
            warning_msg = str(mock_logger.warning.call_args[0][0])
            self.assertIn("unencrypted", warning_msg)
            self.assertIn("SCALEOUT_PERSIST_TOKENS=false", warning_msg)

    def test_is_access_token_valid_with_valid_token(self):
        """Test is_access_token_valid returns True for valid token."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)
        
        # Save token that expires in the future
        future_expiry = datetime.now(timezone.utc) + timedelta(minutes=5)
        cache.save(self.access_token, self.refresh_token, future_expiry)
        
        self.assertTrue(cache.is_access_token_valid())

    def test_is_access_token_valid_with_expired_token(self):
        """Test is_access_token_valid returns False for expired token."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)
        
        # Save token that expired in the past
        past_expiry = datetime.now(timezone.utc) - timedelta(minutes=5)
        cache.save(self.access_token, self.refresh_token, past_expiry)
        
        self.assertFalse(cache.is_access_token_valid())

    def test_is_access_token_valid_with_expiring_soon_token(self):
        """Test is_access_token_valid returns False for token expiring within buffer."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)
        
        # Save token that expires in 5 seconds (within 10 second buffer)
        soon_expiry = datetime.now(timezone.utc) + timedelta(seconds=5)
        cache.save(self.access_token, self.refresh_token, soon_expiry)
        
        self.assertFalse(cache.is_access_token_valid())

    def test_is_access_token_valid_with_no_cache(self):
        """Test is_access_token_valid returns False when no cache exists."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)
        
        self.assertFalse(cache.is_access_token_valid())

    def test_is_access_token_valid_with_no_expiry(self):
        """Test is_access_token_valid returns False when no expiry in cache."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)
        cache.save(self.access_token, self.refresh_token, None)
        
        self.assertFalse(cache.is_access_token_valid())

    def test_is_access_token_valid_with_invalid_expiry_format(self):
        """Test is_access_token_valid returns False with invalid expiry format."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)
        
        # Manually write invalid expiry format
        cache_file = cache.cache_dir / f"{self.cache_id}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "expires_at": "invalid-date-format",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }, f)
        
        self.assertFalse(cache.is_access_token_valid())

    def test_get_token_data_returns_full_data(self):
        """Test get_token_data returns complete token data."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)
        
        future_expiry = datetime.now(timezone.utc) + timedelta(minutes=5)
        cache.save(self.access_token, self.refresh_token, future_expiry)
        
        data = cache.get_token_data()
        
        self.assertIsNotNone(data)
        self.assertEqual(data["access_token"], self.access_token)
        self.assertEqual(data["refresh_token"], self.refresh_token)
        self.assertEqual(data["expires_at"], future_expiry.isoformat())
        self.assertIn("updated_at", data)

    def test_get_token_data_with_no_cache(self):
        """Test get_token_data returns None when no cache exists."""
        cache = TokenCache(cache_id=self.cache_id, cache_dir=self.temp_dir)
        
        data = cache.get_token_data()
        
        self.assertIsNone(data)


if __name__ == '__main__':
    unittest.main()
