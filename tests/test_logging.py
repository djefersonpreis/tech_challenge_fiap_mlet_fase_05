"""Tests for logging configuration."""

from pathlib import Path

from src.utils.logging_config import LOG_DIR, logger


class TestLoggingConfig:
    """Tests for logging configuration."""

    def test_log_dir_exists(self):
        """Test that the log directory exists."""
        assert LOG_DIR.exists()

    def test_log_dir_is_directory(self):
        """Test that LOG_DIR is a directory."""
        assert LOG_DIR.is_dir()

    def test_logger_is_configured(self):
        """Test that logger is importable and functional."""
        # Should not raise
        logger.info("Test log message from test_logging")

    def test_logger_debug_level(self):
        """Test that logger can log debug messages."""
        logger.debug("Debug test message")
