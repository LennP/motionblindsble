"""Constants for MotionBlinds BLE."""

from enum import Enum, IntEnum

SETTING_MAX_CONNECT_ATTEMPTS = 5
SETTING_MAX_COMMAND_ATTEMPTS = 5
SETTING_DISCONNECT_TIME = 15  # Seconds
SETTING_CALIBRATION_DISCONNECT_TIME = 45  # Seconds
SETTING_NOTIFICATION_DELAY = 0.5  # Seconds

EXCEPTION_NO_END_POSITIONS = "{device_name}'s end positions need to be set"
" before usage of this command."
EXCEPTION_NO_FAVORITE_POSITION = "{device_name}'s favorite position needs"
" to be set before usage of this command."
EXCEPTION_NOT_CALIBRATED = "{device_name} needs to be calibrated using the"
" MotionBlinds BLE app before usage."


class MotionService(Enum):
    """Enum for the BLE service."""

    CONTROL = "d973f2e0-b19e-11e2-9e96-0800200c9a66"


class MotionCharacteristic(Enum):
    """Enum for the BLE characteristic."""

    COMMAND = "d973f2e2-b19e-11e2-9e96-0800200c9a66"
    NOTIFICATION = "d973f2e1-b19e-11e2-9e96-0800200c9a66"


class MotionCommandType(Enum):
    """Enum for the command type."""

    OPEN = "03020301"
    CLOSE = "03020302"
    STOP = "03020303"
    OPEN_TILT = "03020309"
    CLOSE_TILT = "0302030a"
    FAVORITE = "03020306"
    PERCENT = "05020440"
    ANGLE = "05020420"
    SPEED = "0403010a"
    SET_KEY = "02c001"
    STATUS_QUERY = "03050f02"
    USER_QUERY = "02c005"
    POINT_SET_QUERY = "03050120"


class MotionNotificationType(Enum):
    """Enum for the notification type."""

    POSITION = "07040402"
    STATUS = "12040f02"


class MotionConnectionType(Enum):
    """Enum for the connection type."""

    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    DISCONNECTING = "disconnecting"


class MotionRunningType(Enum):
    """Enum for the blind running type."""

    OPENING = "opening"
    CLOSING = "closing"
    STILL = "still"
    UNKNOWN = "unknown"


class MotionSpeedLevel(IntEnum):
    """Enum for the speed level."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class MotionCalibrationType(Enum):
    """Enum for the calibration type."""

    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"
    CALIBRATING = "calibrating"


class MotionBlindType(Enum):
    """Enum for the blind type."""

    ROLLER = "Roller blind"
    HONEYCOMB = "Honeycomb blind"
    ROMAN = "Roman blind"
    VENETIAN = "Venetian blind"
    VENETIAN_TILT_ONLY = "Venetian blind (tilt-only)"
    DOUBLE_ROLLER = "Double Roller blind"
    CURTAIN = "Curtain blind"
    VERTICAL = "Vertical blind"
