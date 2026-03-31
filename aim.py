from __future__ import annotations

import ctypes
import platform
import time
from dataclasses import dataclass, field


@dataclass
class AimState:
    last_time: float = 0.0
    last_error_x: float = 0.0
    last_error_y: float = 0.0
    command_vx: float = 0.0
    command_vy: float = 0.0


@dataclass
class AimController:
    """Steuert die Maus relativ, um ein Ziel in die Bildschirmmitte zu ziehen."""

    kp_fresh: float = 0.038
    kd_fresh: float = 0.020
    kp_predicted: float = 0.024
    kd_predicted: float = 0.010
    deadzone_px: float = 2.5
    stop_stale_ms: float = 300.0
    caution_stale_ms: float = 110.0
    near_damping_radius_px: float = 120.0
    max_speed_px_s: float = 1400.0
    max_accel_px_s2: float = 5200.0
    max_step_px: int = 55
    micro_step_threshold: float = 0.35
    derivative_clip: float = 1800.0

    _state: AimState = field(default_factory=AimState, init=False)

    def __post_init__(self):
        self._is_windows = platform.system().lower() == "windows"

    def aim_target(self, target: dict, screen_center: tuple[float, float], now: float | None = None) -> bool:
        if not self._is_windows or target is None:
            self._reset_if_idle(now)
            return False

        now_ts = time.perf_counter() if now is None else now
        stale_ms = float(target.get("stale_ms", 0.0))
        if stale_ms >= self.stop_stale_ms:
            self._reset_if_idle(now_ts)
            return False

        error_x = float(target["center_x"]) - screen_center[0]
        error_y = float(target["center_y"]) - screen_center[1]
        error_mag = (error_x * error_x + error_y * error_y) ** 0.5

        if error_mag <= self.deadzone_px:
            self._reset_if_idle(now_ts)
            return False

        dt = self._compute_dt(now_ts)
        deriv_x = self._clamp((error_x - self._state.last_error_x) / dt, -self.derivative_clip, self.derivative_clip)
        deriv_y = self._clamp((error_y - self._state.last_error_y) / dt, -self.derivative_clip, self.derivative_clip)

        predicted = bool(target.get("predicted", False))
        if predicted:
            kp = self.kp_predicted
            kd = self.kd_predicted
        else:
            kp = self.kp_fresh
            kd = self.kd_fresh

        damping = min(1.0, error_mag / self.near_damping_radius_px)
        stale_factor = 1.0
        if stale_ms > self.caution_stale_ms:
            stale_factor = max(0.35, 1.0 - ((stale_ms - self.caution_stale_ms) / self.stop_stale_ms))

        cmd_vx = ((kp * error_x) + (kd * deriv_x)) * self.max_speed_px_s * damping * stale_factor
        cmd_vy = ((kp * error_y) + (kd * deriv_y)) * self.max_speed_px_s * damping * stale_factor

        cmd_vx = self._rate_limit(self._state.command_vx, cmd_vx, dt)
        cmd_vy = self._rate_limit(self._state.command_vy, cmd_vy, dt)

        step_x = self._clamp(cmd_vx * dt, -self.max_step_px, self.max_step_px)
        step_y = self._clamp(cmd_vy * dt, -self.max_step_px, self.max_step_px)

        if abs(step_x) < self.micro_step_threshold and abs(step_y) < self.micro_step_threshold:
            self._remember(now_ts, error_x, error_y, cmd_vx, cmd_vy)
            return False

        move_x = int(round(step_x))
        move_y = int(round(step_y))
        if move_x == 0 and move_y == 0:
            self._remember(now_ts, error_x, error_y, cmd_vx, cmd_vy)
            return False

        self._send_relative_mouse_move(move_x, move_y)
        self._remember(now_ts, error_x, error_y, cmd_vx, cmd_vy)
        return True

    def _compute_dt(self, now_ts: float) -> float:
        if self._state.last_time <= 0.0:
            return 1.0 / 60.0
        return self._clamp(now_ts - self._state.last_time, 1.0 / 240.0, 0.08)

    def _rate_limit(self, previous: float, target: float, dt: float) -> float:
        delta = target - previous
        max_delta = self.max_accel_px_s2 * dt
        if delta > max_delta:
            return previous + max_delta
        if delta < -max_delta:
            return previous - max_delta
        return target

    def _remember(self, now_ts: float, error_x: float, error_y: float, cmd_vx: float, cmd_vy: float):
        self._state.last_time = now_ts
        self._state.last_error_x = error_x
        self._state.last_error_y = error_y
        self._state.command_vx = cmd_vx
        self._state.command_vy = cmd_vy

    def _reset_if_idle(self, now_ts: float | None = None):
        self._state.last_time = time.perf_counter() if now_ts is None else now_ts
        self._state.last_error_x = 0.0
        self._state.last_error_y = 0.0
        self._state.command_vx = 0.0
        self._state.command_vy = 0.0

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _send_relative_mouse_move(dx: int, dy: int):
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
            type=0,
            mi=MOUSEINPUT(
                dx=dx,
                dy=dy,
                mouseData=0,
                dwFlags=0x0001,
                time=0,
                dwExtraInfo=None,
            ),
        )

        ctypes.windll.user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))
