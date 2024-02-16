"""Util for MotionBlinds BLE."""

import re

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData


async def discover(
    *args, **kwargs
) -> list[tuple[BLEDevice, AdvertisementData]]:
    """
    Asynchronously scan for BLE devices with names matching 'MOTION_****'
    and print their addresses and names.
    """
    motionblinds_device_name = re.compile(r"MOTION_[A-Z0-9]{4}$")
    devices_advertisements = await BleakScanner.discover(
        return_adv=True, *args, **kwargs
    )
    motion_devices_advertisements = []
    for device, advertisement in devices_advertisements.values():
        if device.name and motionblinds_device_name.match(device.name):
            motion_devices_advertisements.append((device, advertisement))
    return motion_devices_advertisements
