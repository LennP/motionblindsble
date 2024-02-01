from motionblindsble.device import (
    MotionPositionInfo,
    requires_end_positions,
    requires_favorite_position,
    requires_connection,
    MotionDevice,
    NoEndPositionsException,
    NoFavoritePositionException,
    ConnectionQueue,
)

import pytest
from unittest.mock import patch, AsyncMock, Mock


class TestDeviceDecorators:
    """Test the decorators in device.py module."""

    async def test_requires_end_positions_success(self) -> None:
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

    async def test_requires_favorite_position_success(self) -> None:
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

    async def test_requires_connection_success(self) -> None:
        device = MotionDevice("00:11:22:33:44:55")

        @requires_connection
        async def mock_method(self, ignore_end_positions_not_set=False):
            return True

        with patch(
            "motionblindsble.device.MotionDevice.connect", AsyncMock(return_value=True)
        ):
            result = await mock_method(device)
            assert result

        with patch(
            "motionblindsble.device.MotionDevice.connect", AsyncMock(return_value=False)
        ):
            result = await mock_method(device)
            assert not result


class TestDeviceConnectionQueue:
    """Test the ConnectionQueue in device.py module."""

    def test_connection_create_task(self) -> None:
        connection_queue = ConnectionQueue()
        device = MotionDevice("00:11:22:33:44:55")

        # Test creation of connect task
        with patch(
            "motionblindsble.device.establish_connection", AsyncMock(return_value=True)
        ):
            connection_queue._create_connection_task(device)

        # Test creation of connect task with Home Assistant
        mock_ha_create_task = Mock()
        connection_queue.set_ha_create_task(mock_ha_create_task)
        assert connection_queue._ha_create_task is not None
        with patch(
            "motionblindsble.device.establish_connection", AsyncMock(return_value=True)
        ):
            connection_queue._create_connection_task(device)
        mock_ha_create_task.assert_called()
