from motionblindsble.device import (
    MotionPositionInfo,
    requires_end_positions,
    requires_favorite_position,
    requires_connection,
    MotionDevice,
    NoEndPositionsException,
    NoFavoritePositionException,
)

import pytest
from unittest.mock import patch, AsyncMock


async def test_requires_end_positions_success():
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


async def test_requires_favorite_position_success():
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


async def test_requires_connection_success():
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
