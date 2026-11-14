"""Read held keys only while this process's MuJoCo X11 window has focus."""

from __future__ import annotations

import ctypes as c
import ctypes.util
import os


class ViewerKeys:
    """Focused X11 key state, including releases and simultaneous keys."""

    def __init__(self):
        library = ctypes.util.find_library("X11")
        if not library or not os.environ.get("DISPLAY"):
            raise RuntimeError("Keyboard control requires an X11 DISPLAY and libX11")
        self.x = c.CDLL(library)
        self.x.XOpenDisplay.argtypes = [c.c_char_p]
        self.x.XOpenDisplay.restype = c.c_void_p
        self.x.XCloseDisplay.argtypes = [c.c_void_p]
        self.x.XGetInputFocus.argtypes = [c.c_void_p, c.POINTER(c.c_ulong), c.POINTER(c.c_int)]
        self.x.XFetchName.argtypes = [c.c_void_p, c.c_ulong, c.POINTER(c.c_void_p)]
        self.x.XFree.argtypes = [c.c_void_p]
        self.x.XInternAtom.argtypes = [c.c_void_p, c.c_char_p, c.c_int]
        self.x.XInternAtom.restype = c.c_ulong
        self.x.XGetWindowProperty.argtypes = [
            c.c_void_p,
            c.c_ulong,
            c.c_ulong,
            c.c_long,
            c.c_long,
            c.c_int,
            c.c_ulong,
            c.POINTER(c.c_ulong),
            c.POINTER(c.c_int),
            c.POINTER(c.c_ulong),
            c.POINTER(c.c_ulong),
            c.POINTER(c.c_void_p),
        ]
        self.x.XQueryKeymap.argtypes = [c.c_void_p, c.c_char_p]
        self.x.XGrabKey.argtypes = [
            c.c_void_p,
            c.c_int,
            c.c_uint,
            c.c_ulong,
            c.c_int,
            c.c_int,
            c.c_int,
        ]
        self.x.XFlush.argtypes = [c.c_void_p]
        self.x.XPending.argtypes = [c.c_void_p]
        self.x.XNextEvent.argtypes = [c.c_void_p, c.c_void_p]
        self.x.XStringToKeysym.argtypes = [c.c_char_p]
        self.x.XStringToKeysym.restype = c.c_ulong
        self.x.XKeysymToKeycode.argtypes = [c.c_void_p, c.c_ulong]
        self.x.XKeysymToKeycode.restype = c.c_ubyte
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("Cannot connect to X11 display")
        self.pid_atom = self.x.XInternAtom(self.display, b"_NET_WM_PID", 0)
        self.grabbed_window = None
        self.codes = {
            key: self.x.XKeysymToKeycode(self.display, self.x.XStringToKeysym(key.encode()))
            for key in "wasdrfjkuoiph lq0123".replace(" ", "")
        }

    def focused(self) -> bool:
        window, revert = c.c_ulong(), c.c_int()
        self.x.XGetInputFocus(self.display, c.byref(window), c.byref(revert))
        if window.value <= 1:
            return False
        name = c.c_void_p()
        if not self.x.XFetchName(self.display, window.value, c.byref(name)) or not name.value:
            return False
        try:
            if not c.string_at(name).decode(errors="replace").startswith("MuJoCo"):
                return False
        finally:
            self.x.XFree(name)
        actual_type, fmt, count, remaining, value = (
            c.c_ulong(),
            c.c_int(),
            c.c_ulong(),
            c.c_ulong(),
            c.c_void_p(),
        )
        status = self.x.XGetWindowProperty(
            self.display,
            window.value,
            self.pid_atom,
            0,
            1,
            0,
            0,
            c.byref(actual_type),
            c.byref(fmt),
            c.byref(count),
            c.byref(remaining),
            c.byref(value),
        )
        try:
            owned = bool(
                status == 0
                and value.value
                and fmt.value == 32
                and count.value == 1
                and c.cast(value, c.POINTER(c.c_ulong))[0] == os.getpid()
            )
        finally:
            if value.value:
                self.x.XFree(value)
        if owned:
            self._grab_control_keys(window.value)
        return owned

    def _grab_control_keys(self, window: int) -> None:
        """Reserve control keys on the verified Viewer window only."""
        if self.grabbed_window == window:
            return
        # owner_events=False keeps these keys from triggering Viewer shortcuts.
        for code in set(self.codes.values()) - {0}:
            self.x.XGrabKey(self.display, code, 1 << 15, window, 0, 1, 1)
        self.x.XFlush(self.display)
        self.grabbed_window = window

    def read(self) -> set[str] | None:
        if not self.focused():
            return None
        # Passive grabs deliver events to this X connection; held-key state below
        # is authoritative, so drain those events rather than letting them pile up.
        event = c.create_string_buffer(192)
        while self.x.XPending(self.display):
            self.x.XNextEvent(self.display, event)
        keys = c.create_string_buffer(32)
        self.x.XQueryKeymap(self.display, keys)
        raw = keys.raw
        return {
            key for key, code in self.codes.items() if code and raw[code // 8] & (1 << (code % 8))
        }

    def close(self):
        if self.display:
            self.x.XCloseDisplay(self.display)
            self.display = None
