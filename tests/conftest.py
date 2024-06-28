"""Main configuration file for pytest."""
# pylint: disable=redefined-outer-name
import pytest

from motionblindsble.device import (
    ConnectionManager,
    MotionDevice,
    MotionPositionInfo,
)


@pytest.fixture
def device():
    """Returns a base device."""
    return MotionDevice("00:11:22:33:44:55")


@pytest.fixture
def device_all_positions(device):
    """Returns a device with all (up, down, favorite) positions set."""
    end_position_info = MotionPositionInfo(0x0E, 0xFFFF)
    device.update_end_position_info(end_position_info)
    return device


@pytest.fixture
def device_end_positions(device):
    """Returns a device with end positions set."""
    end_position_info = MotionPositionInfo(0x0E, 0x0000)
    device.update_end_position_info(end_position_info)
    return device


@pytest.fixture
def device_no_end_positions(device):
    """Returns a device with no positions set."""
    end_position_info = MotionPositionInfo(0x02, 0x0000)
    device.update_end_position_info(end_position_info)
    return device


@pytest.fixture(
    params=[
        MotionPositionInfo(0x0E, 0xFFFF),
        MotionPositionInfo(0x0E, 0x0000),
        MotionPositionInfo(0x02, 0x0000),
    ],
    scope="function",
)
def device_any_positions(request, device):
    """Returns a device with any positions set."""
    end_position_info = request.param
    device.update_end_position_info(end_position_info)
    return device


@pytest.fixture
def connection_manager():
    """Returns a connection manager."""
    return ConnectionManager()
