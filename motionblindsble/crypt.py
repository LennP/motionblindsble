"""Encryption for Motionblinds BLE."""

from __future__ import annotations

import datetime
from datetime import tzinfo
from zoneinfo import ZoneInfo

from Crypto.Cipher import AES
from Crypto.Cipher._mode_ecb import EcbMode
from Crypto.Util.Padding import pad, unpad


class MotionCrypt:
    """Used for the encryption & decryption of bluetooth messages."""

    _timezone: tzinfo | None = None

    _encryption_key: bytes = b"a3q8r8c135sqbn66"
    _cipher: EcbMode = AES.new(_encryption_key, AES.MODE_ECB)

    @staticmethod
    def set_timezone(timezone: str | None) -> None:
        """Set the timezone for encryption, such as 'Europe/Amsterdam'."""
        MotionCrypt._timezone = (
            ZoneInfo(timezone) if timezone is not None else None
        )

    @staticmethod
    def encrypt(plaintext_hex: str) -> str:
        """Encrypt a hex string."""
        plaintext_bytes = bytes.fromhex(plaintext_hex)
        ciphertext_bytes = MotionCrypt._cipher.encrypt(
            pad(plaintext_bytes, AES.block_size)
        )
        ciphertext_hex = ciphertext_bytes.hex()
        return ciphertext_hex

    @staticmethod
    def decrypt(cipheredtext_hex: str) -> str:
        """Decrypt a hex string."""
        ciphertext_bytes = bytes.fromhex(cipheredtext_hex)
        plaintext_bytes = unpad(
            MotionCrypt._cipher.decrypt(ciphertext_bytes), AES.block_size
        )
        plaintext_hex = plaintext_bytes.hex()
        return plaintext_hex

    @staticmethod
    def _format_hex(number: int, number_of_chars: int = 2) -> str:
        """Format a number as a hex string with set number of characters."""
        return hex(number & 2 ** (number_of_chars * 4) - 1)[2:].zfill(
            number_of_chars
        )

    @staticmethod
    def get_time(timezone: tzinfo | None = None) -> str:
        """Get the current time string."""
        if not MotionCrypt._timezone and not timezone:
            raise TimezoneNotSetException(
                "Motion encryption requires a valid timezone."
            )
        now = datetime.datetime.now(
            timezone if timezone is not None else MotionCrypt._timezone
        )

        year = now.year % 100
        month = now.month
        day = now.day
        hour = now.hour
        minute = now.minute
        second = now.second
        microsecond = now.microsecond // 1000

        year_hex = MotionCrypt._format_hex(year)
        month_hex = MotionCrypt._format_hex(month)
        day_hex = MotionCrypt._format_hex(day)
        hour_hex = MotionCrypt._format_hex(hour)
        minute_hex = MotionCrypt._format_hex(minute)
        second_hex = MotionCrypt._format_hex(second)
        microsecond_hex = MotionCrypt._format_hex(
            microsecond, number_of_chars=4
        )

        return (
            year_hex
            + month_hex
            + day_hex
            + hour_hex
            + minute_hex
            + second_hex
            + microsecond_hex
        )


class TimezoneNotSetException(Exception):
    """Error to indicate the timezone was not set."""
