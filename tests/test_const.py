"""Tests for the const.py module."""


class TestConst:
    """Test the const.py module."""

    def test_import_const(self):
        """Test importing the constants module."""
        # Attempt to import the module
        try:
            import motionblindsble.const  # noqa # pylint: disable=unused-import, import-outside-toplevel

            success = True
        except ImportError:
            success = False

        assert success, "Failed to import motion_blinds_ble module"
