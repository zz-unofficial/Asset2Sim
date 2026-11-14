"""Viewer controls must not become global keyboard monitoring or Viewer shortcuts."""

from unittest.mock import Mock, call

from asset2sim.simulation.grasp.keyboard_x11 import ViewerKeys


def test_grab_only_control_keys_on_target_window():
    keys = ViewerKeys.__new__(ViewerKeys)
    keys.x = Mock()
    keys.display = 123
    keys.codes = {"w": 25, "2": 11, "invalid": 0, "duplicate": 25}
    keys.grabbed_window = None
    keys._grab_control_keys(456)
    keys.x.XGrabKey.assert_has_calls(
        [call(123, 25, 1 << 15, 456, 0, 1, 1), call(123, 11, 1 << 15, 456, 0, 1, 1)],
        any_order=True,
    )
    keys._grab_control_keys(456)
    assert keys.x.XGrabKey.call_count == 2


def test_unfocused_viewer_does_not_read_keys():
    keys = ViewerKeys.__new__(ViewerKeys)
    keys.x = Mock()
    keys.focused = Mock(return_value=False)
    assert keys.read() is None
    keys.x.XQueryKeymap.assert_not_called()
