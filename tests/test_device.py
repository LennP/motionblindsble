"""Tests for the device.py module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from motionblindsble.device import (SETTING_MAX_COMMAND_ATTEMPTS, BleakError,
                                    ConnectionQueue, MotionDevice,
                                    MotionPositionInfo, MotionSpeedLevel,
                                    NoEndPositionsException,
                                    NoFavoritePositionException,
                                    requires_connection,
                                    requires_end_positions,
                                    requires_favorite_position)


class TestDeviceDecorators:
    """Test the decorators in device.py module."""

    async def test_requires_end_positions(self) -> None:
        """Test the @requires_end_positions decorator."""
        device = MotionDevice("00:11:22:33:44:55")
        device.end_position_info = MotionPositionInfo(
            0xFF, 0xFFFF
        )  # Simulating end positions being set
        device.register_running_callback(lambda _: None)

        @requires_end_positions
        async def mock_method(self, ignore_end_positions_not_set=False):
            return True

        result = await mock_method(device)
        assert result

        device.end_position_info.update_end_positions(
            0x00
        )  # # Simulating end positions not being
        with pytest.raises(NoEndPositionsException):
            result = await mock_method(device)

    async def test_requires_favorite_position(self) -> None:
        """Test the @requires_favorite_position decorator."""
        device = MotionDevice("00:11:22:33:44:55")
        device.end_position_info = MotionPositionInfo(
            0xFF, 0xFFFF
        )  # Simulating end positions being set
        device.register_running_callback(lambda _: None)

        @requires_favorite_position
        async def mock_method(self, ignore_end_positions_not_set=False):
            return True

        result = await mock_method(device)
        assert result

        device.end_position_info = MotionPositionInfo(
            0x00, 0x0000
        )  # Simulating end positions being set
        with pytest.raises(NoFavoritePositionException):
            await mock_method(device)

    async def test_requires_connection(self) -> None:
        """Test the @requires_connection decorator."""
        device = MotionDevice("00:11:22:33:44:55")

        @requires_connection
        async def mock_method(self, ignore_end_positions_not_set=False):
            return True

        with patch(
            "motionblindsble.device.MotionDevice.connect",
            AsyncMock(return_value=True),
        ):
            result = await mock_method(device)
            assert result

        with patch(
            "motionblindsble.device.MotionDevice.connect",
            AsyncMock(return_value=False),
        ):
            result = await mock_method(device)
            assert not result


class TestDeviceConnectionQueue:
    """Test the ConnectionQueue in device.py module."""

    def test_create_connection_task(self) -> None:
        """Test the creation of a connection task."""
        connection_queue = ConnectionQueue()
        device = MotionDevice("00:11:22:33:44:55")

        # Test creation of connect task
        with patch(
            "motionblindsble.device.MotionDevice.establish_connection",
            AsyncMock(return_value=True),
        ):
            connection_queue._create_connection_task(device)

        # Test creation of connect task with Home Assistant
        mock_ha_create_task = Mock()
        connection_queue.set_ha_create_task(mock_ha_create_task)
        assert connection_queue._ha_create_task is not None
        with patch(
            "motionblindsble.device.MotionDevice.establish_connection",
            AsyncMock(return_value=True),
        ):
            connection_queue._create_connection_task(device)
        mock_ha_create_task.assert_called()

    # async def test_wait_for_connection(self) -> None:
    #     """Tests waiting for a connection."""
    #     pass
    #     connection_queue = ConnectionQueue()
    #     device = MotionDevice("00:11:22:33:44:55")

    #     Test creation of connection
    #     with patch(
    #         "motionblindsble.device.ConnectionQueue._create_connection_task",
    #         AsyncMock(return_value=True),
    #     ):
    #         can_execute_command = await connection_queue \
    #             .wait_for_connection(device)
    #         assert can_execute_command
    #         assert connection_queue._connection_task is None

    async def test_wait_for_connection(self) -> None:
        """Test cancelling a connection task."""
        connection_queue = ConnectionQueue()

        # Test cancelling with no connection task
        assert not connection_queue.cancel()

        # Test cancelling with connection task
        connection_task = Mock()
        connection_queue._connection_task = connection_task
        assert connection_queue.cancel()
        connection_task.cancel.assert_called_once()
        assert connection_queue._connection_task is None


class TestDevice:
    """Test the Device commands in device.py module."""

    async def test_send_command(self) -> None:
        """Test sending a command."""
        device = MotionDevice("00:11:22:33:44:55")

        # Test sending command success
        with patch(
            "motionblindsble.device.MotionCrypt", Mock()
        ) as MotionCryptMock:
            MotionCryptMock.encrypt.return_value = "AA"
            MotionCryptMock.decrypt.return_value = "AA"
            MotionCryptMock.get_time.return_value = "AA"

            device._current_bleak_client = AsyncMock(return_value=True)
            assert await device._send_command("0011223344556677889900")
            device._current_bleak_client.write_gatt_char.assert_called_once()

        # Test sending success after SETTING_MAX_COMMAND_ATTEMPTS tries
        with patch(
            "motionblindsble.device.MotionCrypt", Mock()
        ) as MotionCryptMock:
            MotionCryptMock.encrypt.return_value = "AA"
            MotionCryptMock.decrypt.return_value = "AA"
            MotionCryptMock.get_time.return_value = "AA"

            device._current_bleak_client = AsyncMock()
            device._current_bleak_client.write_gatt_char.side_effect = [
                BleakError
            ] * (SETTING_MAX_COMMAND_ATTEMPTS - 1) + [True]
            assert await device._send_command("0011223344556677889900")
            device._current_bleak_client.write_gatt_char.assert_called()
            assert (
                device._current_bleak_client.write_gatt_char.call_count
                == SETTING_MAX_COMMAND_ATTEMPTS
            )

        # Test sending fail after SETTING_MAX_COMMAND_ATTEMPTS tries
        with patch(
            "motionblindsble.device.MotionCrypt", Mock()
        ) as MotionCryptMock:
            MotionCryptMock.encrypt.return_value = "AA"
            MotionCryptMock.decrypt.return_value = "AA"
            MotionCryptMock.get_time.return_value = "AA"

            device._current_bleak_client = AsyncMock()
            device._current_bleak_client.write_gatt_char.side_effect = (
                BleakError
            )
            assert not await device._send_command("0011223344556677889900")
            device._current_bleak_client.write_gatt_char.assert_called()
            assert (
                device._current_bleak_client.write_gatt_char.call_count
                == SETTING_MAX_COMMAND_ATTEMPTS
            )

    @patch(
        "motionblindsble.device.MotionDevice.connect",
        AsyncMock(return_value=True),
    )
    @patch(
        "motionblindsble.device.MotionDevice._send_command",
        AsyncMock(return_value=True),
    )
    async def test_commands(self) -> None:
        """Test sending different commands."""
        device = MotionDevice("00:11:22:33:44:55")
        device.end_position_info = MotionPositionInfo(0xFF, 0xFFFF)

        call_counter = 0
        test_commands = [
            device.user_query,
            device.set_key,
            device.status_query,
            device.point_set_query,
            (device.speed, [MotionSpeedLevel.MEDIUM]),
            (device.percentage, [10]),
            device.open,
            device.close,
            device.stop,
            device.favorite,
            (device.percentage_tilt, [10]),
            device.open_tilt,
            device.close_tilt,
        ]
        for test_command in test_commands:
            if isinstance(test_command, tuple):
                assert await test_command[0](*test_command[1])
            else:
                assert await test_command()
            call_counter += 1
            device._send_command: Mock
            assert device._send_command.call_count == call_counter

    async def test_callbacks(self) -> None:
        """Test the device callbacks."""
        device = MotionDevice("00:11:22:33:44:55")

        position_callback = Mock()
        device.register_position_callback(position_callback)
        device._position_callback()
        device._position_callback.assert_called_once()

        running_callback = Mock()
        device.register_running_callback(running_callback)
        device.running_callback()
        device.running_callback.assert_called_once()

        connection_callback = Mock()
        device.register_connection_callback(connection_callback)
        device._connection_callback()
        device._connection_callback.assert_called_once()

        status_callback = Mock()
        device.register_status_callback(status_callback)
        device._status_callback()
        device._status_callback.assert_called_once()
