class TestConst:
    """Test the const.py module."""

    def test_import_const(self):
        """Test importing the constants module."""
        # Attempt to import the module
        try:
            import motionblindsble.const  # noqa

            success = True
        except ImportError:
            success = False

        assert success, "Failed to import motion_blinds_ble module"
