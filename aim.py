from __future__ import annotations

import ctypes
import platform
import time
from dataclasses import dataclass, field


@dataclass
class AimState:
    last_time: float = 0.0
    last_target_time: float = 0.0
    last_track_id: int = -1
    last_error_x: float = 0.0
    last_error_y: float = 0.0
    command_vx: float = 0.0
    command_vy: float = 0.0
    residual_x: float = 0.0
    residual_y: float = 0.0
    integral_x: float = 0.0
    integral_y: float = 0.0


@dataclass
class AimController:
    """Steuert die Maus relativ, um ein Ziel weich und präzise in die Bildschirmmitte zu führen."""

    kp_fresh: float = 0.034
    kd_fresh: float = 0.012
    kp_predicted: float = 0.024
    kd_predicted: float = 0.009
    ki_close: float = 0.020
    velocity_lead_weight: float = 0.24
    acceleration_lead_weight: float = 0.09
    deadzone_px: float = 0.45
    soft_zone_px: float = 5.5
    near_damping_radius_px: float = 120.0
    close_integral_radius_px: float = 28.0
    caution_stale_ms: float = 120.0
    stop_stale_ms: float = 360.0
    max_speed_px_s_fresh: float = 1450.0
    max_speed_px_s_predicted: float = 880.0
    max_accel_px_s2: float = 4600.0
    max_step_px: float = 40.0
    derivative_clip: float = 1450.0
    close_nudge_px: float = 1.0
    integral_clip: float = 14.0

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

        self._ensure_track_continuity(target=target, now_ts=now_ts)

        predicted = bool(target.get("predicted", False))
        max_speed = self.max_speed_px_s_predicted if predicted else self.max_speed_px_s_fresh
        kp = self.kp_predicted if predicted else self.kp_fresh
        kd = self.kd_predicted if predicted else self.kd_fresh

        aim_x, aim_y = self._compute_latency_compensated_aim(target=target, now_ts=now_ts)
        error_x = aim_x - screen_center[0]
        error_y = aim_y - screen_center[1]
        error_mag = (error_x * error_x + error_y * error_y) ** 0.5

        dt = self._compute_dt(now_ts)
        deriv_x = self._derivative(error_x, self._state.last_error_x, dt)
        deriv_y = self._derivative(error_y, self._state.last_error_y, dt)

        near_scale = self._compute_near_scale(error_mag)
        stale_factor = self._compute_stale_factor(stale_ms)
        predictive_factor = 0.72 if predicted else 1.0

        self._update_integral(error_x=error_x, error_y=error_y, error_mag=error_mag, dt=dt)
        integral_boost_x = self.ki_close * self._state.integral_x
        integral_boost_y = self.ki_close * self._state.integral_y

        target_vx = ((kp * error_x) + (kd * deriv_x) + integral_boost_x) * max_speed
        target_vy = ((kp * error_y) + (kd * deriv_y) + integral_boost_y) * max_speed
        target_vx *= near_scale * stale_factor * predictive_factor
        target_vy *= near_scale * stale_factor * predictive_factor

        cmd_vx = self._rate_limit(self._state.command_vx, target_vx, dt)
        cmd_vy = self._rate_limit(self._state.command_vy, target_vy, dt)

        step_x = self._clamp(cmd_vx * dt, -self.max_step_px, self.max_step_px)
        step_y = self._clamp(cmd_vy * dt, -self.max_step_px, self.max_step_px)

        move_x, move_y = self._quantize_step(step_x, step_y, error_x=error_x, error_y=error_y, predicted=predicted)
        if move_x == 0 and move_y == 0:
            self._remember(now_ts, error_x, error_y, cmd_vx, cmd_vy)
            return False

        self._send_relative_mouse_move(move_x, move_y)
        self._remember(now_ts, error_x, error_y, cmd_vx, cmd_vy)
        return True

    def _compute_latency_compensated_aim(self, target: dict, now_ts: float) -> tuple[float, float]:
        aim_x = float(target.get("aim_x", target.get("predicted_x", 0.0)))
        aim_y = float(target.get("aim_y", target.get("predicted_y", 0.0)))

        vx = float(target.get("velocity_x", 0.0))
        vy = float(target.get("velocity_y", 0.0))
        ax = float(target.get("accel_x", 0.0))
        ay = float(target.get("accel_y", 0.0))

        capture_ts = float(target.get("capture_timestamp", now_ts))
        predict_dt = max(0.0, min(0.30, now_ts - capture_ts))

        lead_x = (vx * predict_dt * self.velocity_lead_weight) + (0.5 * ax * predict_dt * predict_dt * self.acceleration_lead_weight)
        lead_y = (vy * predict_dt * self.velocity_lead_weight) + (0.5 * ay * predict_dt * predict_dt * self.acceleration_lead_weight)
        return aim_x + lead_x, aim_y + lead_y

    def _compute_near_scale(self, error_mag: float) -> float:
        if error_mag <= self.deadzone_px:
            return 0.0
        scale = min(1.0, error_mag / self.near_damping_radius_px)
        if error_mag < self.soft_zone_px:
            soft = max(0.15, (error_mag - self.deadzone_px) / max(1e-3, self.soft_zone_px - self.deadzone_px))
            scale *= soft
        return scale

    def _compute_stale_factor(self, stale_ms: float) -> float:
        if stale_ms <= self.caution_stale_ms:
            return 1.0
        fade_window = max(1.0, self.stop_stale_ms - self.caution_stale_ms)
        return max(0.10, 1.0 - ((stale_ms - self.caution_stale_ms) / fade_window))

    def _update_integral(self, error_x: float, error_y: float, error_mag: float, dt: float):
        if error_mag > self.close_integral_radius_px:
            self._state.integral_x *= 0.70
            self._state.integral_y *= 0.70
            return

        self._state.integral_x = self._clamp(self._state.integral_x + (error_x * dt), -self.integral_clip, self.integral_clip)
        self._state.integral_y = self._clamp(self._state.integral_y + (error_y * dt), -self.integral_clip, self.integral_clip)

    def _ensure_track_continuity(self, target: dict, now_ts: float):
        track_id = int(target.get("track_id", -1))
        if track_id != self._state.last_track_id:
            self._state.last_track_id = track_id
            self._state.command_vx = 0.0
            self._state.command_vy = 0.0
            self._state.last_error_x = 0.0
            self._state.last_error_y = 0.0
            self._state.integral_x = 0.0
            self._state.integral_y = 0.0
            self._state.residual_x = 0.0
            self._state.residual_y = 0.0
            self._state.last_time = now_ts
        self._state.last_target_time = now_ts

    def _compute_dt(self, now_ts: float) -> float:
        if self._state.last_time <= 0.0:
            return 1.0 / 60.0
        return self._clamp(now_ts - self._state.last_time, 1.0 / 300.0, 0.080)

    def _derivative(self, error: float, previous: float, dt: float) -> float:
        return self._clamp((error - previous) / dt, -self.derivative_clip, self.derivative_clip)

    def _quantize_step(self, step_x: float, step_y: float, error_x: float, error_y: float, predicted: bool) -> tuple[int, int]:
        self._state.residual_x += step_x
        self._state.residual_y += step_y

        move_x = int(self._state.residual_x)
        move_y = int(self._state.residual_y)
        self._state.residual_x -= move_x
        self._state.residual_y -= move_y

        if predicted:
            return move_x, move_y

        if move_x == 0 and abs(error_x) > self.deadzone_px and abs(error_x) < self.soft_zone_px:
            move_x = int(self.close_nudge_px if error_x > 0 else -self.close_nudge_px)
            self._state.residual_x = 0.0
        if move_y == 0 and abs(error_y) > self.deadzone_px and abs(error_y) < self.soft_zone_px:
            move_y = int(self.close_nudge_px if error_y > 0 else -self.close_nudge_px)
            self._state.residual_y = 0.0

        return move_x, move_y

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
        self._state.last_target_time = self._state.last_time
        self._state.last_error_x = 0.0
        self._state.last_error_y = 0.0
        self._state.command_vx = 0.0
        self._state.command_vy = 0.0
        self._state.residual_x = 0.0
        self._state.residual_y = 0.0
        self._state.integral_x = 0.0
        self._state.integral_y = 0.0

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
