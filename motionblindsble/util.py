"""Util for Motionblinds BLE."""

import re

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData


async def discover(
    *args, **kwargs
) -> list[tuple[BLEDevice, AdvertisementData]]:
    """
    Asynchronously scan for BLE devices with names matching 'MOTION_****'
    and return the BLEDevice and AdvertisementData.
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


async def discover_device(
    mac: str, *args, **kwargs
) -> tuple[BLEDevice, AdvertisementData] | None:
    """
    Asynchronously scan for a BLE device matching the mac address or mac code,
    such as A4:C1:38:00:2D:FB, A4C138002DFB or 2DFB.
    Return the BLEDevice and AdvertisementData.
    """
    motion_devices_advertisements = await discover(*args, **kwargs)
    for device, advertisement in motion_devices_advertisements:
        if (
            device.address.replace(":", "")
            .upper()
            .endswith(mac.replace(":", "").upper())
        ):
            return (device, advertisement)

    return None
