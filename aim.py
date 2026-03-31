from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass


@dataclass
class AimController:
    """Steuert die Maus relativ, um ein Ziel in die Bildschirmmitte zu ziehen."""

    smoothing: float = 0.35
    deadzone_px: int = 3
    max_step_px: int = 120

    def __post_init__(self):
        self._is_windows = platform.system().lower() == "windows"

    def aim_to_screen_center(
        self,
        target_center: tuple[float, float],
        screen_center: tuple[float, float],
    ) -> bool:
        """
        Bewegt die Maus relativ in Richtung Zielmittelpunkt.

        Rückgabe:
        - True: Eingabe wurde gesendet.
        - False: keine Aktion (z. B. Deadzone / nicht unterstützt).
        """
        if not self._is_windows:
            return False

        delta_x = target_center[0] - screen_center[0]
        delta_y = target_center[1] - screen_center[1]

        if abs(delta_x) <= self.deadzone_px and abs(delta_y) <= self.deadzone_px:
            return False

        move_x = self._clip_step(int(round(delta_x * self.smoothing)))
        move_y = self._clip_step(int(round(delta_y * self.smoothing)))

        if move_x == 0 and move_y == 0:
            return False

        self._send_relative_mouse_move(move_x, move_y)
        return True

    def _clip_step(self, value: int) -> int:
        if value > self.max_step_px:
            return self.max_step_px
        if value < -self.max_step_px:
            return -self.max_step_px
        return value

    @staticmethod
    def _send_relative_mouse_move(dx: int, dy: int):
        # WinAPI SendInput für relative Mausbewegung.
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_ulong),
                ("mi", MOUSEINPUT),
            ]

        input_struct = INPUT(
            type=0,  # INPUT_MOUSE
            mi=MOUSEINPUT(
                dx=dx,
                dy=dy,
                mouseData=0,
                dwFlags=0x0001,  # MOUSEEVENTF_MOVE
                time=0,
                dwExtraInfo=None,
            ),
        )

        ctypes.windll.user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))
