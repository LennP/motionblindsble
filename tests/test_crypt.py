"""Tests for the crypt.py module."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from motionblindsble.crypt import MotionCrypt, TimezoneNotSetException


class TestCrypt:
    """Test the crypt.py module."""

    def test_encrypt_decrypt(self) -> None:
        """Test encryption and decryption."""

        MotionCrypt.set_timezone("Europe/Amsterdam")

        expected_encrypted = "244e1d963ebdc5453f43e896465b5bcf"
        expected_decrypted = "070404020e0059b4"

        decrypted = MotionCrypt.decrypt(expected_encrypted)
        encrypted = MotionCrypt.encrypt(decrypted)

        assert expected_decrypted == decrypted
        assert expected_encrypted == encrypted

    @patch("motionblindsble.crypt.datetime")
    def test_get_time(self, mock_datetime: Mock) -> None:
        """Test getting the time string."""

        MotionCrypt.set_timezone("Europe/Amsterdam")

        assert isinstance(MotionCrypt.get_time(), str)

        mock_datetime.datetime.now.return_value = datetime(
            year=2015,
            month=3,
            day=4,
            hour=5,
            minute=6,
            second=7,
            microsecond=999999,
            tzinfo=timezone.utc,
        )

        assert MotionCrypt.get_time() == "0f030405060703e7"

    def test_get_time_timezone_not_set(self) -> None:
        """Test getting the time string when the timezone is not set."""

        MotionCrypt._timezone = None

        with pytest.raises(TimezoneNotSetException):
            MotionCrypt.get_time()

        MotionCrypt.get_time(timezone=timezone.utc)
