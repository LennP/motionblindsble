"""Tests for the device.py module."""

import asyncio
from asyncio import Event
import fnmatch
from unittest.mock import AsyncMock, Mock, call, patch
from typing import Callable

import pytest
from motionblindsble.device import (
    SETTING_DISCONNECT_TIME,
    SETTING_MAX_COMMAND_ATTEMPTS,
    SETTING_NOTIFICATION_DELAY,
    SETTING_CALIBRATION_DISCONNECT_TIME,
    BleakError,
    BleakNotFoundError,
    BleakOutOfConnectionSlotsError,
    BLEDevice,
    ConnectionQueue,
    MotionConnectionType,
    MotionDevice,
    MotionEndPositions,
    MotionNotificationType,
    MotionRunningType,
    MotionCalibrationType,
    MotionCallback,
    MotionPositionInfo,
    MotionBlindType,
    MotionSpeedLevel,
    NoEndPositionsException,
    NoFavoritePositionException,
    NotCalibratedException,
    requires_connection,
    requires_end_positions,
    requires_favorite_position,
)


class TestDeviceDecorators:
    """Test the decorators in device.py module."""

    @patch("motionblindsble.device.MotionDevice.refresh_disconnect_timer")
    @patch("motionblindsble.device.MotionDevice.update_calibration")
    async def test_requires_end_positions(
        self,
        mock_update_calibration,
        mock_refresh_disconnect_timer,
    ) -> None:
        """Test the @requires_end_positions decorator."""
        device = MotionDevice("00:11:22:33:44:55")
        device.register_running_callback(lambda _: None)

        @requires_end_positions
        async def mock_method(self):
            return True

        @requires_end_positions(can_calibrate_curtain=True)
        async def mock_method_calibration(self):
            return True

        with patch.object(
            device, "_received_end_position_info_event"
        ) as end_position_event_mock:

            def set_end_positions():
                # Set end positions after Event.wait() is called
                device.update_end_position_info(
                    MotionPositionInfo(
                        0x0E, 0xFFFF
                    )  # Simulating end positions being set
                )

            end_position_event_mock.wait = AsyncMock(
                side_effect=set_end_positions
            )

            result = await mock_method(device)
            assert result

        device._end_position_info.update_end_positions(
            0x00
        )  # Simulating end positions not being
        device._calibration_type = MotionCalibrationType.UNCALIBRATED

        for blind_type in [
            MotionBlindType.ROLLER,
            MotionBlindType.DOUBLE_ROLLER,
            MotionBlindType.HONEYCOMB,
            MotionBlindType.ROMAN,
            MotionBlindType.VENETIAN,
            MotionBlindType.VENETIAN_TILT_ONLY,
        ]:
            device.blind_type = blind_type
            with pytest.raises(NoEndPositionsException):
                result = await mock_method(device)

        device.blind_type = MotionBlindType.VERTICAL
        with pytest.raises(NotCalibratedException):
            result = await mock_method(device)

        device.blind_type = MotionBlindType.CURTAIN
        mock_refresh_disconnect_timer.reset_mock()
        mock_update_calibration.reset_mock()
        result = await mock_method(device)
        assert result

        result = await mock_method_calibration(device)
        assert result
        mock_update_calibration.assert_called_once_with(
            MotionCalibrationType.CALIBRATING
        )
        mock_refresh_disconnect_timer.assert_called_once_with(
            SETTING_CALIBRATION_DISCONNECT_TIME
        )

    @patch(
        "motionblindsble.device.MotionDevice.refresh_disconnect_timer",
        Mock(),
    )
    async def test_requires_favorite_position(self) -> None:
        """Test the @requires_favorite_position decorator."""
        device = MotionDevice("00:11:22:33:44:55")
        device.update_end_position_info(
            MotionPositionInfo(
                0x0E, 0xFFFF
            )  # Simulating end positions being set
        )
        device.register_running_callback(lambda _: None)

        @requires_favorite_position
        async def mock_method(self):
            return True

        result = await mock_method(device)
        assert result

        device.update_end_position_info(
            MotionPositionInfo(
                0x00, 0x0000
            )  # Simulating end positions not being set
        )
        with pytest.raises(NoFavoritePositionException):
            await mock_method(device)

    @patch("motionblindsble.device.MotionDevice.connect")
    async def test_requires_connection(self, mock_connect) -> None:
        """Test the @requires_connection decorator."""
        device = MotionDevice("00:11:22:33:44:55")

        @requires_connection
        async def mock_method(self):
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
        mock_create_task = Mock()
        device.set_create_task_factory(mock_create_task)
        assert device._create_task is not None
        with patch(
            "motionblindsble.device.MotionDevice.establish_connection",
            AsyncMock(return_value=True),
        ):
            connection_queue._create_connection_task(device)
        mock_create_task.assert_called()

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
            assert device.connection_type is MotionConnectionType.DISCONNECTED

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
    @patch("motionblindsble.device.MotionDevice.update_connection")
    async def test_establish_connection(
        self, mock_update_connection, mock_sleep
    ) -> None:
        """Test the establish_connection function."""
        device = MotionDevice("00:11:22:33:44:55")

        # Test establish normal connection
        await device.establish_connection()
        # pylint: disable=no-member
        device.refresh_disconnect_timer.assert_called()
        mock_update_connection.assert_has_calls(
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
        for blind_type in [MotionBlindType.CURTAIN, MotionBlindType.VERTICAL]:
            mock_sleep.reset_mock()
            device.blind_type = blind_type
            await device.establish_connection()
            mock_sleep.assert_called_once_with(SETTING_NOTIFICATION_DELAY)

        for blind_type in [
            MotionBlindType.ROLLER,
            MotionBlindType.DOUBLE_ROLLER,
            MotionBlindType.HONEYCOMB,
            MotionBlindType.ROMAN,
            MotionBlindType.VENETIAN,
            MotionBlindType.VENETIAN_TILT_ONLY,
        ]:
            mock_sleep.reset_mock()
            device.blind_type = blind_type
            await device.establish_connection()
            assert mock_sleep.call_count == 0

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

        # Test exception in connection which should update connection
        # and then raise the exception again
        mock_is_connected.return_value = False
        mock_wait_for_connection.side_effect = Exception()
        with pytest.raises(Exception):
            assert await device.connect()
            assert device._connection_type == MotionConnectionType.DISCONNECTED

        # Test connect when connected
        mock_is_connected.return_value = True
        mock_wait_for_connection.return_value = True
        assert await device.connect()

        assert mock_refresh_disconnect_timer.call_count == 1
        assert mock_wait_for_connection.call_count == 3

    @patch("motionblindsble.device.MotionDevice.update_connection")
    async def test_disconnect(self, mock_update_connection) -> None:
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
        mock_update_connection.assert_has_calls(
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

    @patch(
        "motionblindsble.device.establish_connection",
        AsyncMock(),
    )
    @patch(
        "motionblindsble.crypt.MotionCrypt.get_time",
        Mock(return_value=""),
    )
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

        # Test custom disconnect time
        device._disconnect_time = 0
        device.set_custom_disconnect_time(9999)
        device.refresh_disconnect_timer()
        assert device._disconnect_time == 9999000

        # Test with _call_later and _disconnect_later:
        def call_later(delay: int, action):
            # Run immediately instead of after delay
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(action())
            else:
                asyncio.run(action())

        device._call_later = Mock(side_effect=call_later)
        device.refresh_disconnect_timer()
        device._call_later.assert_called_once()

        await asyncio.sleep(
            0
        )  # Wait for _disconnect_later to finish in event loop
        mock_disconnect.assert_called_once()

        # Test permanent connection, no disconnect timer
        await device.set_permanent_connection(True)
        mock_time_ns.reset_mock()
        device.refresh_disconnect_timer()
        assert mock_time_ns.call_count == 0

        # Test permanent connection, disconnect timer cancels
        permanent_connection_enabled = Event()

        async def call_once_condition_is_met(action: Callable) -> None:
            print("Waiting for condition")
            await permanent_connection_enabled.wait()
            await action()

        def call_later_condition(delay: int, action: Callable) -> None:
            # Run immediately instead of after delay
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(call_once_condition_is_met(action))
            else:
                asyncio.run(call_once_condition_is_met(action))

        mock_disconnect.reset_mock()
        device._call_later = Mock(side_effect=call_later_condition)
        # First set it to False and pass first line of refresh_disconnect_timer
        # and make sure that a _disconnect_later is created
        await device.set_permanent_connection(False)
        device.refresh_disconnect_timer()
        # Set _permanent_connection to True and after that
        # _disconnect_later can be called
        await device.set_permanent_connection(True)
        await asyncio.sleep(
            0
        )  # Wait for _disconnect_later to finish in event loop
        permanent_connection_enabled.set()
        assert mock_disconnect.call_count == 1

    @patch("motionblindsble.device.MotionDevice.connect")
    async def test_permanent_connection(self, mock_connect) -> None:
        """Test the permanent connection function."""
        device = MotionDevice("00:11:22:33:44:55")
        device._disconnect_callback(Mock())
        assert mock_connect.call_count == 0

        # Test normal permanent connection
        await device.set_permanent_connection(True)
        assert mock_connect.call_count == 1
        device._disconnect_callback(Mock())
        assert mock_connect.call_count == 2

        # Test permanent connection with Home Assistant
        mock_create_task = Mock()
        device.set_create_task_factory(mock_create_task)
        await device.set_permanent_connection(True)
        device._disconnect_callback(Mock())
        assert mock_create_task.call_count == 1


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
        device = MotionDevice(ble_device1)
        device.set_ble_device(ble_device2)
        assert device.ble_device == ble_device2

        device2 = MotionDevice(ble_device2.address)
        assert device2.ble_device.address == ble_device2.address

    @patch("motionblindsble.device.discover_device")
    async def test_init_discover(self, mock_discover_device) -> None:
        """Test initializing a MotionDevice with discover function."""
        # Test no device found
        mock_discover_device.return_value = None
        device = await MotionDevice.discover("00:11:22:33:44:55")
        assert device is None

        # Test device found
        ble_device = Mock()
        ble_device.__class__ = BLEDevice
        ble_device.address = "00:11:22:33:44:55"
        advertisement_data = Mock()
        mock_discover_device.return_value = (ble_device, advertisement_data)
        device = await MotionDevice.discover("00:11:22:33:44:55")
        assert device is not None
        assert device.ble_device is ble_device

    def test_setters(self) -> None:
        """Test initializing a MotionDevice."""
        device = MotionDevice("00:11:22:33:44:55")
        mock = Mock()
        mock2 = Mock()
        device.set_create_task_factory(mock)
        device.set_call_later_factory(mock2)
        assert device._create_task is mock
        assert device._call_later is mock2

    @patch("motionblindsble.device.MotionCrypt.decrypt", lambda x: x)
    async def test_notification_callback(self) -> None:
        """Test the notification callback."""
        device = MotionDevice("00:11:22:33:44:55")
        device.register_feedback_callback(Mock())
        device.register_status_callback(Mock())

        # Test FEEDBACK notification with end positions
        device._notification_callback(
            None,
            bytearray.fromhex(
                MotionNotificationType.FEEDBACK.value
                + "02"
                + "00"
                + "64"
                + "B4"
            ),
        )
        for feedback_callback in device._feedback_callbacks:
            feedback_callback.assert_called_once_with(
                100, 180, device._end_position_info
            )
        assert (
            device._end_position_info is not MotionEndPositions.NONE
            and device._end_position_info.favorite_position is None
        )

        # Test FEEDBACK notification without end positions
        for feedback_callback in device._feedback_callbacks:
            feedback_callback.reset_mock()
        device._notification_callback(
            None,
            bytearray.fromhex(
                MotionNotificationType.FEEDBACK.value
                + "0e"
                + "00"
                + "00"
                + "00"
            ),
        )
        for feedback_callback in device._feedback_callbacks:
            feedback_callback.assert_called_with(
                0, 0, device._end_position_info
            )
        assert (
            device._end_position_info.end_positions is MotionEndPositions.BOTH
            and device._end_position_info.favorite_position is None
        )

        # Test STATUS notification with end positions
        for status_callback in device._status_callbacks:
            status_callback.reset_mock()
        device._end_position_info = None
        device._notification_callback(
            None,
            bytearray.fromhex(
                MotionNotificationType.STATUS.value
                + "0E"
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
        for status_callback in device._status_callbacks:
            status_callback.assert_called_once_with(
                100,
                180,
                96,
                MotionSpeedLevel.MEDIUM,
                device._end_position_info,
            )
        assert device._end_position_info is not None
        assert (
            device._end_position_info.end_positions is MotionEndPositions.BOTH
            and device._end_position_info.favorite_position
        )

        # Test STATUS notification without end positions
        for status_callback in device._status_callbacks:
            status_callback.reset_mock()
        device._end_position_info = None
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
        for status_callback in device._status_callbacks:
            status_callback.assert_called_with(
                0, 0, 10, MotionSpeedLevel.HIGH, device._end_position_info
            )
        assert device._end_position_info is not None

        # Test with invalid arguments
        for status_callback in device._status_callbacks:
            status_callback.reset_mock()
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
        for status_callback in device._status_callbacks:
            status_callback.assert_called_with(
                0, 0, 10, None, device._end_position_info
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
        device.update_end_position_info(MotionPositionInfo(0x0E, 0xFFFF))

        call_counter = 0
        test_commands = [
            device.user_query,
            device.set_key,
            device.status_query,
            device.point_set_query,
            (device.speed, [MotionSpeedLevel.MEDIUM]),
            (device.position, [10]),
            device.open,
            device.close,
            device.stop,
            device.favorite,
            (device.tilt, [10]),
            device.open_tilt,
            device.close_tilt,
        ]
        for test_command in test_commands:
            if isinstance(test_command, tuple):
                assert await test_command[0](*test_command[1])
            else:
                assert await test_command()
            call_counter += 1
            # pylint: disable=no-member
            assert device._send_command.call_count == call_counter

    async def test_register_callbacks(self) -> None:
        """Test registering device callbacks."""
        device = MotionDevice("00:11:22:33:44:55")

        register_callbacks = []
        for attr_name in dir(device):
            if fnmatch.fnmatch(attr_name, "register_*_callback"):
                method = getattr(device, attr_name)
                if callable(method):
                    register_callbacks.append(method)

        for register_callback in register_callbacks:
            callback = Mock()
            register_callback(callback)
            callback()
            callback.assert_called_once()

    @patch("motionblindsble.device.time_ns", Mock(return_value=0))
    async def test_disabled_connection_callbacks(self) -> None:
        """Test the device callbacks that are disabled on connection."""

        device = MotionDevice("00:11:22:33:44:55")
        device._connect_status_query_time = 0

        inputs = [
            (
                device.register_status_callback,
                device.remove_status_callback,
                MotionCallback.STATUS,
                device.update_status,
                [
                    5,
                    5,
                    10,
                    MotionSpeedLevel.HIGH,
                    MotionPositionInfo(0x0E, 0xFFFF),
                ],
                [
                    "_position",
                    "_tilt",
                    "_battery",
                    "_speed",
                    "_end_position_info",
                ],
            ),
            (
                device.register_feedback_callback,
                device.remove_feedback_callback,
                MotionCallback.FEEDBACK,
                device.update_feedback,
                [
                    5,
                    5,
                    MotionPositionInfo(0x0E, 0xFFFF),
                ],
                [
                    "_position",
                    "_tilt",
                    "_end_position_info",
                ],
            ),
            (
                device.register_end_position_callback,
                device.remove_end_position_callback,
                MotionCallback.END_POSITION_INFO,
                device.update_end_position_info,
                [MotionPositionInfo(0x0E, 0xFFFF)],
                ["_end_position_info"],
            ),
            (
                device.register_connection_callback,
                device.remove_connection_callback,
                MotionCallback.CONNECTION,
                device.update_connection,
                [MotionConnectionType.CONNECTED],
                ["_connection_type"],
            ),
            (
                device.register_calibration_callback,
                device.remove_calibration_callback,
                MotionCallback.CALIBRATION,
                device.update_calibration,
                [MotionCalibrationType.CALIBRATED],
                ["_calibration_type"],
            ),
            (
                device.register_running_callback,
                device.remove_running_callback,
                MotionCallback.RUNNING,
                device.update_running,
                [MotionRunningType.OPENING],
                ["_running_type"],
            ),
            (
                device.register_battery_callback,
                device.remove_battery_callback,
                MotionCallback.BATTERY,
                device.update_battery,
                [5],
                ["_battery"],
            ),
            (
                device.register_speed_callback,
                device.remove_speed_callback,
                MotionCallback.SPEED,
                device.update_speed,
                [MotionSpeedLevel.HIGH],
                ["_speed"],
            ),
            (
                device.register_position_callback,
                device.remove_position_callback,
                MotionCallback.POSITION,
                device.update_position,
                [5, 5],
                ["_position", "_tilt"],
            ),
            (
                device.register_signal_strength_callback,
                device.remove_signal_strength_callback,
                MotionCallback.SIGNAL_STRENGTH,
                device.update_signal_strength,
                [-15],
                ["rssi"],
            ),
        ]

        callback = Mock()

        for inp in inputs:
            register_callback = inp[0]
            remove_callback = inp[1]
            disable_callback = inp[2]
            update = inp[3]
            args = inp[4]
            attribute_names = inp[5]

            callback.reset_mock()
            register_callback(callback)
            update(*args)
            if len(args) == len(attribute_names):
                for arg, attribute_name in zip(args, attribute_names):
                    assert (
                        getattr(device, attribute_name, "Attribute not found")
                        == arg
                    )
            callback.assert_called_once_with(*args)

            # Test callback not calling when disabled
            device._disabled_connection_callbacks = [disable_callback]
            update(*args)
            callback.assert_called_once_with(*args)

            # Test deleting callback
            remove_callback(callback)
            device._disabled_connection_callbacks = []
            update(*args)
            callback.assert_called_once_with(*args)

            # Test removing by name
            callback.__name__ = "test_name"
            register_callback(callback)
            remove_callback(callback.__name__)
            update(*args)
            callback.assert_called_once_with(*args)

        # Test removing callback that is not a function or a string
        with pytest.raises(ValueError):
            device._generic_remove_callback(5, lambda _: _)
