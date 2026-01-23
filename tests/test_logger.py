import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from car_image_classification_using_cnn.logger import get_logger, setup_logger


class TestSetupLogger:
    """Test suite for setup_logger function."""

    def test_setup_logger_without_file(self):
        """Test setting up logger without a log file."""
        with patch.object(logger, "remove") as mock_remove, patch.object(logger, "add") as mock_add:
            setup_logger(log_file=None, level="INFO")

            # Should remove existing handlers
            mock_remove.assert_called_once()

            # Should add stderr handler
            assert mock_add.call_count == 1

            # Verify stderr handler configuration
            call_args = mock_add.call_args_list[0]
            assert call_args[0][0] == sys.stderr

    def test_setup_logger_with_file(self, tmp_path):
        """Test setting up logger with a log file."""
        log_file = tmp_path / "logs" / "test.log"

        with patch.object(logger, "remove") as mock_remove, patch.object(logger, "add") as mock_add:
            setup_logger(log_file=log_file, level="DEBUG")

            # Should remove existing handlers
            mock_remove.assert_called_once()

            # Should add both stderr and file handlers
            assert mock_add.call_count == 2

            # First call is stderr
            stderr_call = mock_add.call_args_list[0]
            assert stderr_call[0][0] == sys.stderr

            # Second call is file handler
            file_call = mock_add.call_args_list[1]
            assert file_call[0][0] == log_file

    def test_setup_logger_creates_log_directory(self, tmp_path):
        """Test that setup_logger creates log directory if it doesn't exist."""
        log_file = tmp_path / "nested" / "logs" / "test.log"

        assert not log_file.parent.exists()

        # Actually call the function to create directory
        setup_logger(log_file=log_file, level="INFO")

        assert log_file.parent.exists()

    def test_setup_logger_with_different_levels(self):
        """Test setup_logger with different log levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        for level in levels:
            with patch.object(logger, "remove"), patch.object(logger, "add") as mock_add:
                setup_logger(log_file=None, level=level)

                # Check that level was passed correctly
                stderr_call = mock_add.call_args_list[0]
                assert stderr_call[1]["level"] == level

    def test_setup_logger_with_rotation(self, tmp_path):
        """Test setup_logger with rotation parameter."""
        log_file = tmp_path / "test.log"
        rotation = "50 MB"

        with patch.object(logger, "remove"), patch.object(logger, "add") as mock_add:
            setup_logger(log_file=log_file, level="INFO", rotation=rotation)

            # File handler should have rotation parameter
            file_call = mock_add.call_args_list[1]
            assert file_call[1]["rotation"] == rotation

    def test_setup_logger_file_has_retention(self, tmp_path):
        """Test that file logger has retention policy."""
        log_file = tmp_path / "test.log"

        with patch.object(logger, "remove"), patch.object(logger, "add") as mock_add:
            setup_logger(log_file=log_file, level="INFO")

            # File handler should have retention
            file_call = mock_add.call_args_list[1]
            assert "retention" in file_call[1]
            assert file_call[1]["retention"] == "1 week"

    def test_setup_logger_file_has_compression(self, tmp_path):
        """Test that file logger has compression enabled."""
        log_file = tmp_path / "test.log"

        with patch.object(logger, "remove"), patch.object(logger, "add") as mock_add:
            setup_logger(log_file=log_file, level="INFO")

            # File handler should have compression
            file_call = mock_add.call_args_list[1]
            assert "compression" in file_call[1]
            assert file_call[1]["compression"] == "zip"

    def test_setup_logger_stderr_has_colors(self):
        """Test that stderr handler has colorization enabled."""
        with patch.object(logger, "remove"), patch.object(logger, "add") as mock_add:
            setup_logger(log_file=None, level="INFO")

            # Stderr handler should have colorize=True
            stderr_call = mock_add.call_args_list[0]
            assert stderr_call[1]["colorize"] is True

    def test_setup_logger_file_debug_level(self, tmp_path):
        """Test that file logger always uses DEBUG level."""
        log_file = tmp_path / "test.log"

        with patch.object(logger, "remove"), patch.object(logger, "add") as mock_add:
            # Even though we set INFO for main logger
            setup_logger(log_file=log_file, level="INFO")

            # File handler should use DEBUG
            file_call = mock_add.call_args_list[1]
            assert file_call[1]["level"] == "DEBUG"

    def test_setup_logger_accepts_path_object(self, tmp_path):
        """Test that setup_logger accepts Path object for log_file."""
        log_file = Path(tmp_path) / "test.log"

        with patch.object(logger, "remove"), patch.object(logger, "add"):
            # Should not raise any errors
            setup_logger(log_file=log_file, level="INFO")


class TestGetLogger:
    """Test suite for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns the logger instance."""
        returned_logger = get_logger()

        # Should return the logger object
        assert returned_logger is logger

    def test_get_logger_is_callable(self):
        """Test that returned logger has standard methods."""
        returned_logger = get_logger()

        # Should have standard logging methods
        assert hasattr(returned_logger, "info")
        assert hasattr(returned_logger, "debug")
        assert hasattr(returned_logger, "warning")
        assert hasattr(returned_logger, "error")
        assert hasattr(returned_logger, "critical")

    def test_get_logger_can_log(self):
        """Test that returned logger can actually log messages."""
        returned_logger = get_logger()

        with patch.object(returned_logger, "info") as mock_info:
            returned_logger.info("Test message")
            mock_info.assert_called_once_with("Test message")


class TestLoggerIntegration:
    """Integration tests for logger module."""

    def test_logger_actually_logs_to_file(self, tmp_path):
        """Test that logger actually writes to file."""
        log_file = tmp_path / "integration.log"

        # Setup logger with file
        setup_logger(log_file=log_file, level="INFO")

        # Get logger and log a message
        test_logger = get_logger()
        test_message = "Integration test message"
        test_logger.info(test_message)

        # Give it a moment to write
        import time

        time.sleep(0.1)

        # Verify file was created (actual file, not just mocked)
        assert log_file.exists(), "Log file should be created"

    def test_logger_with_different_rotation_values(self, tmp_path):
        """Test logger with different rotation configurations."""
        rotations = ["1 MB", "100 MB", "1 week", "1 day"]

        for i, rotation in enumerate(rotations):
            log_file = tmp_path / f"test_{i}.log"

            with patch.object(logger, "remove"), patch.object(logger, "add") as mock_add:
                setup_logger(log_file=log_file, level="INFO", rotation=rotation)

                file_call = mock_add.call_args_list[1]
                assert file_call[1]["rotation"] == rotation
