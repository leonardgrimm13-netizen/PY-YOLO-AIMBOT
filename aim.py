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
    """Steuert die Maus relativ, um ein Ziel weich in die Bildschirmmitte zu führen."""

    kp_fresh: float = 0.032
    kd_fresh: float = 0.013
    kp_predicted: float = 0.020
    kd_predicted: float = 0.008
    velocity_lead_gain: float = 0.018
    deadzone_px: float = 2.0
    soft_zone_px: float = 7.0
    near_damping_radius_px: float = 115.0
    caution_stale_ms: float = 115.0
    stop_stale_ms: float = 320.0
    max_speed_px_s_fresh: float = 1280.0
    max_speed_px_s_predicted: float = 820.0
    max_accel_px_s2: float = 4200.0
    max_step_px: int = 42
    derivative_clip: float = 1400.0
    micro_step_threshold: float = 0.34

    _state: AimState = field(default_factory=AimState, init=False)

    def __post_init__(self):
        self._is_windows = platform.system().lower() == "windows"

    def aim_target(self, target: dict, screen_center: tuple[float, float], now: float | None = None) -> bool:
        if not self._is_windows or target is None:
            self._reset(now)
            return False

        now_ts = time.perf_counter() if now is None else now
        stale_ms = float(target.get("stale_ms", 0.0))
        if stale_ms >= self.stop_stale_ms:
            self._reset(now_ts)
            return False

        predicted = bool(target.get("predicted", False))
        max_speed = self.max_speed_px_s_predicted if predicted else self.max_speed_px_s_fresh
        kp = self.kp_predicted if predicted else self.kp_fresh
        kd = self.kd_predicted if predicted else self.kd_fresh

        aim_x, aim_y = self._compute_aim_point(target=target, predicted=predicted)
        error_x = aim_x - screen_center[0]
        error_y = aim_y - screen_center[1]
        error_mag = (error_x * error_x + error_y * error_y) ** 0.5

        if error_mag <= self.deadzone_px:
            self._reset(now_ts)
            return False

        dt = self._compute_dt(now_ts)
        deriv_x = self._derivative(error_x, self._state.last_error_x, dt)
        deriv_y = self._derivative(error_y, self._state.last_error_y, dt)

        near_scale = min(1.0, error_mag / self.near_damping_radius_px)
        if error_mag < self.soft_zone_px:
            near_scale *= max(0.28, error_mag / max(self.deadzone_px, self.soft_zone_px))

        stale_factor = 1.0
        if stale_ms > self.caution_stale_ms:
            stale_window = max(1.0, self.stop_stale_ms - self.caution_stale_ms)
            stale_factor = max(0.08, 1.0 - ((stale_ms - self.caution_stale_ms) / stale_window))

        target_vx = ((kp * error_x) + (kd * deriv_x)) * max_speed * near_scale * stale_factor
        target_vy = ((kp * error_y) + (kd * deriv_y)) * max_speed * near_scale * stale_factor

        cmd_vx = self._rate_limit(self._state.command_vx, target_vx, dt)
        cmd_vy = self._rate_limit(self._state.command_vy, target_vy, dt)

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

    def _compute_aim_point(self, target: dict, predicted: bool) -> tuple[float, float]:
        center_x = float(target.get("center_x", 0.0))
        center_y = float(target.get("center_y", 0.0))
        velocity_x = float(target.get("velocity_x", 0.0))
        velocity_y = float(target.get("velocity_y", 0.0))
        lead = self.velocity_lead_gain * (0.45 if predicted else 1.0)
        return center_x + (velocity_x * lead), center_y + (velocity_y * lead)

    def _compute_dt(self, now_ts: float) -> float:
        if self._state.last_time <= 0.0:
            return 1.0 / 60.0
        return self._clamp(now_ts - self._state.last_time, 1.0 / 300.0, 0.085)

    def _derivative(self, error: float, previous: float, dt: float) -> float:
        return self._clamp((error - previous) / dt, -self.derivative_clip, self.derivative_clip)

    def _rate_limit(self, previous: float, target: float, dt: float) -> float:
        max_delta = self.max_accel_px_s2 * dt
        delta = target - previous
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

    def _reset(self, now_ts: float | None = None):
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
