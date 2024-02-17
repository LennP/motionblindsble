"""Tests for the util.py module."""

from unittest.mock import patch, AsyncMock, Mock
from bleak.backends.device import BLEDevice

from motionblindsble.util import discover, discover_device


class TestUtil:
    """Test util functions in util.py."""

    async def test_discover(self) -> None:
        """Test the discover Motionblinds function."""
        # Mock BLE devices and advertisement data
        mock_devices_advertisements = {
            "00:11:22:33:44:55": (
                BLEDevice("00:11:22:33:44:55", "MOTION_1234", {}, rssi=-60),
                Mock(),
            ),
            "11:22:33:44:55:66": (
                BLEDevice("11:22:33:44:55:66", "NOT_MOTION", {}, rssi=-50),
                Mock(),
            ),
            "22:33:44:55:66:77": (
                BLEDevice("22:33:44:55:66:77", "MOTION_ABCD", {}, rssi=-70),
                Mock(),
            ),
        }

        with patch(
            "motionblindsble.util.BleakScanner.discover",
            new_callable=AsyncMock,
        ) as mock_discover:
            mock_discover.return_value = mock_devices_advertisements

            discovered = await discover()

            # Verify only devices with name matching 'MOTION_****' returned
            expected_devices = [
                mock_devices_advertisements["00:11:22:33:44:55"],
                mock_devices_advertisements["22:33:44:55:66:77"],
            ]
            assert discovered == expected_devices

    @patch("motionblindsble.util.discover")
    async def test_discover_device(self, mock_discover) -> None:
        """Test the get_device function."""
        # Mock BLE devices and advertisement data
        mock_devices_advertisements = {
            "00:11:22:33:44:55": (
                BLEDevice("00:11:22:33:44:55", "MOTION_4455", {}, rssi=-60),
                Mock(),
            ),
            "00:11:22:33:AB:CD": (
                BLEDevice("00:11:22:33:AB:CD", "MOTION_ABCD", {}, rssi=-60),
                Mock(),
            ),
        }
        mock_discover.return_value = mock_devices_advertisements.values()

        assert (
            await discover_device("4455")
            == mock_devices_advertisements["00:11:22:33:44:55"]
        )
        assert (
            await discover_device("001122334455")
            == mock_devices_advertisements["00:11:22:33:44:55"]
        )
        assert (
            await discover_device("00:11:22:33:44:55")
            == mock_devices_advertisements["00:11:22:33:44:55"]
        )
        assert (
            await discover_device("abcd")
            == mock_devices_advertisements["00:11:22:33:AB:CD"]
        )
        assert await discover_device("aaaa") is None
