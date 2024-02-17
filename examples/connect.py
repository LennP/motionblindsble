"""Example showing how to connect and open a blind."""

import asyncio
import logging

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
from motionblindsble.crypt import MotionCrypt
from motionblindsble.device import _LOGGER, MotionDevice
from motionblindsble.util import discover_device

# Configure logging to output to console at DEBUG level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)

_LOGGER.addHandler(console_handler)
_LOGGER.setLevel(logging.DEBUG)


async def main():
    """Main function that connects and opens the blind."""

    # Timezone is needed for encryption
    MotionCrypt.set_timezone("Europe/Amsterdam")

    # Create MotionDevice using MAC code (2DFB)
    # or address (A4:C1:38:00:2D:FB or A4C138002DFB)
    device = await MotionDevice.discover("A4:C1:38:00:2D:FB")

    # Create a MotionDevice using BLEDevice
    # ble_device, _ = await discover_device("2DFB")
    # device = MotionDevice(ble_device)

    # Scan for new BLEDevice for motor
    def detection_callback(
        ble_device: BLEDevice, advertisement_data: AdvertisementData
    ):
        if device.ble_device.address == ble_device.address:
            _LOGGER.info("New BLEDevice found: %s", ble_device.address)

    scanner = BleakScanner(detection_callback=detection_callback)
    await scanner.start()

    # Connect and open blind
    await device.connect()
    await device.open()

    # Wait for blind to automatically disconnect after timeout
    await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(main())
