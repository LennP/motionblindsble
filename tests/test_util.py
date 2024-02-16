"""Tests for the util.py module."""

from unittest.mock import patch, AsyncMock, Mock
from bleak.backends.device import BLEDevice

from motionblindsble.util import discover


class TestUtil:

    async def test_discover_motionblinds(self):
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

            # Verify that only devices with names matching 'MOTION_****' are returned
            expected_devices = [
                mock_devices_advertisements["00:11:22:33:44:55"],
                mock_devices_advertisements["22:33:44:55:66:77"],
            ]
            assert discovered == expected_devices
