"""Tests for the device.py module."""

import asyncio
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from motionblindsble.device import (
    SETTING_DISCONNECT_TIME,
    SETTING_MAX_COMMAND_ATTEMPTS,
    SETTING_NOTIFICATION_DELAY,
    BleakError,
    BleakNotFoundError,
    BleakOutOfConnectionSlotsError,
    BLEDevice,
    ConnectionQueue,
    MotionConnectionType,
    MotionDevice,
    MotionNotificationType,
    MotionPositionInfo,
    MotionSpeedLevel,
    NoEndPositionsException,
    NoFavoritePositionException,
    requires_connection,
    requires_end_positions,
    requires_favorite_position,
)


class TestDeviceDecorators:
    """Test the decorators in device.py module."""

    @patch(
        "motionblindsble.device.MotionDevice.refresh_disconnect_timer",
        Mock(),
    )
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

    @patch(
        "motionblindsble.device.MotionDevice.refresh_disconnect_timer",
        Mock(),
    )
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

    @patch("motionblindsble.device.MotionDevice.connect")
    async def test_requires_connection(self, mock_connect) -> None:
        """Test the @requires_connection decorator."""
        device = MotionDevice("00:11:22:33:44:55")

        @requires_connection
        async def mock_method(self, ignore_end_positions_not_set=False):
            return True

        mock_connect.return_value = True
        assert await mock_method(device)

        mock_connect.return_value = False
        assert not await mock_method(device)


class TestDeviceConnection:
    """Test the ConnectionQueue & establish_connection in device.py module."""

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

    @patch("motionblindsble.device.ConnectionQueue._create_connection_task")
    @patch("motionblindsble.device.wait")
    async def test_wait_for_connection(
        self, mock_wait, mock_create_connection_task
    ) -> None:
        """Tests waiting for a connection."""
        connection_queue = ConnectionQueue()
        device = MotionDevice("00:11:22:33:44:55")

        connection_task = Mock()
        mock_create_connection_task.return_value = connection_task

        async def mock_wait_return_connection_task(tasks, return_when):
            return ([tasks[0]], None)

        async def mock_wait_return_cancel(tasks, return_when):
            return ([tasks[1]], None)

        # Test creation of connection
        connection_task.result.return_value = True
        mock_wait.side_effect = mock_wait_return_connection_task
        can_execute_command = await connection_queue.wait_for_connection(
            device
        )
        assert can_execute_command
        assert connection_queue._connection_task is None

        # Test creation of connection fails
        connection_task.result.return_value = False
        mock_wait.side_effect = mock_wait_return_connection_task
        can_execute_command = await connection_queue.wait_for_connection(
            device
        )
        assert not can_execute_command
        assert connection_queue._connection_task is None

        # Test creation of connection cancelled
        connection_task.result.return_value = True
        mock_wait.side_effect = mock_wait_return_cancel
        can_execute_command = await connection_queue.wait_for_connection(
            device
        )
        assert not can_execute_command
        assert connection_queue._connection_task is not None

        # Test creation of connection with exception
        exceptions = [BleakOutOfConnectionSlotsError, BleakNotFoundError]
        for exception in exceptions:
            connection_task.result.side_effect = exception
            mock_wait.side_effect = mock_wait_return_connection_task
            with pytest.raises(exception):
                can_execute_command = (
                    await connection_queue.wait_for_connection(device)
                )
            assert not can_execute_command
            assert connection_queue._connection_task is None
            assert device._connection_type is MotionConnectionType.DISCONNECTED

    async def test_cancel(self) -> None:
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

    async def test_cancel_disconnect_timer(self) -> None:
        """Test cancelling a disconnect timer."""
        device = MotionDevice("00:11:22:33:44:55")

        # Test with normal disconnect timer
        device._disconnect_timer = Mock()
        device.cancel_disconnect_timer()
        device._disconnect_timer.assert_called_once()

    @patch(
        "motionblindsble.device.MotionDevice.refresh_disconnect_timer",
        Mock(),
    )
    @patch(
        "motionblindsble.device.MotionDevice._send_command",
        AsyncMock(return_value=True),
    )
    @patch(
        "motionblindsble.device.establish_connection",
        AsyncMock(return_value=AsyncMock()),
    )
    @patch("motionblindsble.device.sleep")
    @patch("motionblindsble.device.MotionDevice.set_connection")
    async def test_establish_connection(
        self, mock_set_connection, mock_sleep
    ) -> None:
        """Test the establish_connection function."""
        device = MotionDevice("00:11:22:33:44:55")

        # Test establish normal connection
        await device.establish_connection()
        device._current_bleak_client.set_disconnected_callback \
            .assert_called_once()
        device.refresh_disconnect_timer: Mock
        device.refresh_disconnect_timer.assert_called()
        mock_set_connection.assert_has_calls(
            [
                call(MotionConnectionType.CONNECTING),
                call(MotionConnectionType.CONNECTED),
            ]
        )

        # Test establish connection if already connecting
        device._connection_type = MotionConnectionType.CONNECTING
        assert not await device.establish_connection()
        device._connection_type = None

        # Test establish connection with notification delay
        await device.establish_connection(use_notification_delay=True)
        mock_sleep.assert_called_once_with(SETTING_NOTIFICATION_DELAY)

    @patch("motionblindsble.device.MotionDevice.is_connected")
    @patch("motionblindsble.device.ConnectionQueue.wait_for_connection")
    @patch("motionblindsble.device.MotionDevice.refresh_disconnect_timer")
    async def test_connect(
        self,
        mock_refresh_disconnect_timer,
        mock_wait_for_connection,
        mock_is_connected,
    ) -> None:
        """Test the connect function."""
        device = MotionDevice("00:11:22:33:44:55")

        # Test connect success
        mock_is_connected.return_value = False
        mock_wait_for_connection.return_value = True
        assert await device.connect()

        # Test connect failure
        mock_is_connected.return_value = False
        mock_wait_for_connection.return_value = False
        assert not await device.connect()

        # Test connect when connected
        mock_is_connected.return_value = True
        mock_wait_for_connection.return_value = True
        assert await device.connect()

        assert mock_refresh_disconnect_timer.call_count == 1
        assert mock_wait_for_connection.call_count == 2

    @patch("motionblindsble.device.MotionDevice.set_connection")
    async def test_disconnect(self, mock_set_connection) -> None:
        """Test the disconnect function."""
        device = MotionDevice("00:11:22:33:44:55")

        # Test disconnecting when connected
        current_bleak_client = AsyncMock()
        device._current_bleak_client = current_bleak_client

        def side_effect_disconnect(*args, **kwargs):
            device._disconnect_callback(
                device._current_bleak_client, *args, **kwargs
            )

        device._current_bleak_client.disconnect.side_effect = (
            side_effect_disconnect
        )
        await device.disconnect()
        mock_set_connection.assert_has_calls(
            [
                call(MotionConnectionType.DISCONNECTING),
                call(MotionConnectionType.DISCONNECTED),
            ]
        )
        current_bleak_client.disconnect.assert_called_once()

        # Test disconnecting when BleakClient is None
        device._current_bleak_client = None
        assert not await device.disconnect()

        # Test disconnecting when connecting
        device._connection_queue = Mock()
        assert not await device.disconnect()
        device._connection_queue.cancel.assert_called_once()

    @patch("motionblindsble.device.MotionDevice.disconnect")
    @patch("motionblindsble.device.time_ns")
    async def test_refresh_disconnect_timer(
        self, mock_time_ns, mock_disconnect
    ) -> None:
        """Test the refresh_disconnect_timer function."""
        device = MotionDevice("00:11:22:33:44:55")
        mock_time_ns.return_value = 0

        assert device._disconnect_time is None
        assert device._disconnect_timer is None

        # Test creating a disconnect timer
        device.refresh_disconnect_timer()
        assert device._disconnect_time == SETTING_DISCONNECT_TIME * 1e3
        assert device._disconnect_timer is not None

        # Test creating a disconnect timer with custom disconnect time
        CUSTOM_DISCONNECT_TIME = 999
        device.refresh_disconnect_timer(timeout=CUSTOM_DISCONNECT_TIME)
        assert device._disconnect_time == CUSTOM_DISCONNECT_TIME * 1e3

        # Test creating a disconnect timer, with and without force
        device._disconnect_time = float("inf")
        device.refresh_disconnect_timer()
        assert device._disconnect_time == float("inf")
        device.refresh_disconnect_timer(force=True)
        assert device._disconnect_time == SETTING_DISCONNECT_TIME * 1e3

        # Test with _ha_call_later and _disconnect_later:
        def call_later(delay: int, action):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(action())
            else:
                asyncio.run(action())

        device._ha_call_later = Mock(side_effect=call_later)
        device.refresh_disconnect_timer()
        device._ha_call_later.assert_called_once()

        await asyncio.sleep(
            0
        )  # Wait for _disconnect_later to finish in event loop
        mock_disconnect.assert_called_once()


class TestDevice:
    """Test the Device in device.py module."""

    def test_init(self) -> None:
        """Test initializing a MotionDevice."""
        ble_device1 = BLEDevice(
            "00:11:22:33:44:55", "00:11:22:33:44:55", {}, rssi=0
        )
        ble_device2 = BLEDevice(
            "00:11:22:33:44:55", "00:11:22:33:44:55", {}, rssi=0
        )
        MotionDevice("00:11:22:33:44:55")
        device = MotionDevice("00:11:22:33:44:55", ble_device1)
        device.set_ble_device(ble_device2)
        assert device._ble_device == ble_device2

    def test_setters(self) -> None:
        """Test initializing a MotionDevice."""
        device = MotionDevice("00:11:22:33:44:55")
        mock = Mock()
        mock2 = Mock()
        device.set_ha_create_task(mock)
        device.set_ha_call_later(mock2)
        assert device._connection_queue._ha_create_task is mock
        assert device._ha_call_later is mock2

    @patch("motionblindsble.device.MotionCrypt.decrypt", lambda x: x)
    async def test_notification_callback(self) -> None:
        """Test the notification callback."""
        device = MotionDevice("00:11:22:33:44:55")
        device._position_callback = Mock()
        device._status_callback = Mock()

        # Test POSITION notification with end positions
        device._notification_callback(
            None,
            bytearray.fromhex(
                MotionNotificationType.POSITION.value
                + "02"
                + "00"
                + "64"
                + "B4"
            ),
        )
        device._position_callback.assert_called_once_with(
            100, 100, device.end_position_info
        )
        assert (
            not device.end_position_info.up
            and not device.end_position_info.down
            and device.end_position_info.favorite is None
        )

        # Test POSITION notification without end positions
        device._notification_callback(
            None,
            bytearray.fromhex(
                MotionNotificationType.POSITION.value
                + "0c"
                + "00"
                + "00"
                + "00"
            ),
        )
        device._position_callback.assert_called_with(
            0, 0, device.end_position_info
        )
        assert device._position_callback.call_count == 2
        assert (
            device.end_position_info.up
            and device.end_position_info.down
            and device.end_position_info.favorite is None
        )

        # Test STATUS notification with end positions
        device.end_position_info = None
        device._notification_callback(
            None,
            bytearray.fromhex(
                MotionNotificationType.STATUS.value
                + "0C"
                + "00"
                + "64"
                + "B4"
                + "00000000"
                + "02"
                + "00"
                + "FFFF"
                + "00"
                + "60"
            ),
        )
        device._status_callback.assert_called_once_with(
            100, 100, 96, MotionSpeedLevel.MEDIUM, device.end_position_info
        )
        assert device.end_position_info is not None
        assert (
            device.end_position_info.up
            and device.end_position_info.down
            and device.end_position_info.favorite
        )

        # Test STATUS notification without end positions
        device.end_position_info = None
        device._notification_callback(
            None,
            bytearray.fromhex(
                MotionNotificationType.STATUS.value
                + "02"
                + "00"
                + "00"
                + "00"
                + "00000000"
                + "03"
                + "00"
                + "0000"
                + "00"
                + "0A"
            ),
        )
        device._status_callback.assert_called_with(
            0, 0, 10, MotionSpeedLevel.HIGH, device.end_position_info
        )
        assert device.end_position_info is not None
        assert device._position_callback.call_count == 2

        # Test with invalid arguments
        device._notification_callback(
            None,
            bytearray.fromhex(
                MotionNotificationType.STATUS.value
                + "02"
                + "00"
                + "00"
                + "00"
                + "00000000"
                + "04"
                + "00"
                + "0000"
                + "00"
                + "0A"
            ),
        )
        device._status_callback.assert_called_with(
            0, 0, 10, None, device.end_position_info
        )

    @patch("motionblindsble.device.MotionCrypt.encrypt", return_value="AA")
    @patch("motionblindsble.device.MotionCrypt.decrypt", return_value="AA")
    @patch("motionblindsble.device.MotionCrypt.get_time", return_value="AA")
    async def test_send_command(
        self, mock_get_time, mock_decrypt, mock_encrypt
    ) -> None:
        """Test sending a command."""
        device = MotionDevice("00:11:22:33:44:55")

        # Test sending command without BleakClient
        device._current_bleak_client = None
        assert not await device._send_command("0011223344556677889900")

        # Test sending command success
        device._current_bleak_client = AsyncMock(return_value=True)
        assert await device._send_command("0011223344556677889900")
        device._current_bleak_client.write_gatt_char.assert_called_once()

        # Test sending success after SETTING_MAX_COMMAND_ATTEMPTS tries
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
        device._current_bleak_client = AsyncMock()
        device._current_bleak_client.write_gatt_char.side_effect = BleakError
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
        device.set_connection(MotionConnectionType.CONNECTED)
        assert device._connection_type is MotionConnectionType.CONNECTED

        status_callback = Mock()
        device.register_status_callback(status_callback)
        device._status_callback()
        device._status_callback.assert_called_once()
