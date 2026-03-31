from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DeviceInfo:
    display_name: str
    name: str
    device: str
    kind: str
    detected: bool
    tested: bool
    recommended_rank: int
    note: str = ""

    def to_ui_data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "kind": self.kind,
            "detected": self.detected,
            "tested": self.tested,
            "recommended_rank": self.recommended_rank,
            "note": self.note,
        }


def safe_import_torch():
    try:
        import torch  # type: ignore

        return torch, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _probe_device(torch_mod, device: str) -> tuple[bool, str]:
    try:
        _ = torch_mod.empty(1, device=device)
        return True, "OK"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def list_available_devices() -> tuple[list[DeviceInfo], list[str]]:
    devices: list[DeviceInfo] = [
        DeviceInfo(
            display_name="Auto (empfohlen)",
            name="Auto (empfohlen)",
            device="auto",
            kind="auto",
            detected=True,
            tested=True,
            recommended_rank=0,
            note="Automatische Auswahl mit Fallback",
        ),
        DeviceInfo(
            display_name="CPU",
            name="CPU",
            device="cpu",
            kind="torch",
            detected=True,
            tested=True,
            recommended_rank=100,
            note="Immer verfügbar",
        ),
    ]
    log_lines: list[str] = []

    torch_mod, torch_err = safe_import_torch()
    if torch_mod is None:
        log_lines.append(f"PyTorch nicht verfügbar: {torch_err}")
        return devices, log_lines

    # NVIDIA CUDA
    try:
        if torch_mod.cuda.is_available():
            for index in range(torch_mod.cuda.device_count()):
                name = torch_mod.cuda.get_device_name(index)
                usable, message = _probe_device(torch_mod, f"cuda:{index}")
                note = "Getestet" if usable else f"Erkannt, aber nicht testbar: {message}"
                devices.append(
                    DeviceInfo(
                        display_name=f"NVIDIA CUDA {index}: {name}",
                        name=f"NVIDIA CUDA {index}: {name}",
                        device=f"cuda:{index}",
                        kind="torch",
                        detected=True,
                        tested=usable,
                        recommended_rank=10 + index,
                        note=note,
                    )
                )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"CUDA-Prüfung fehlgeschlagen: {exc}")

    # Intel XPU
    try:
        if hasattr(torch_mod, "xpu") and torch_mod.xpu.is_available():
            for index in range(torch_mod.xpu.device_count()):
                name = torch_mod.xpu.get_device_name(index)
                usable, message = _probe_device(torch_mod, f"xpu:{index}")
                note = "Getestet" if usable else f"Erkannt, aber nicht testbar: {message}"
                devices.append(
                    DeviceInfo(
                        display_name=f"Intel XPU {index}: {name}",
                        name=f"Intel XPU {index}: {name}",
                        device=f"xpu:{index}",
                        kind="torch",
                        detected=True,
                        tested=usable,
                        recommended_rank=30 + index,
                        note=note,
                    )
                )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"XPU-Prüfung fehlgeschlagen: {exc}")

    # Apple MPS
    try:
        has_mps = hasattr(torch_mod, "backends") and hasattr(torch_mod.backends, "mps")
        if has_mps and torch_mod.backends.mps.is_available():
            usable, message = _probe_device(torch_mod, "mps")
            note = "Getestet" if usable else f"Erkannt, aber nicht testbar: {message}"
            devices.append(
                DeviceInfo(
                    display_name="Apple MPS",
                    name="Apple MPS",
                    device="mps",
                    kind="torch",
                    detected=True,
                    tested=usable,
                    recommended_rank=50,
                    note=note,
                )
            )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"MPS-Prüfung fehlgeschlagen: {exc}")

    return devices, log_lines


def resolve_auto_device(device_infos: list[DeviceInfo]) -> DeviceInfo:
    real = [d for d in device_infos if d.kind != "auto"]
    tested_real = sorted(
        (d for d in real if d.tested),
        key=lambda item: item.recommended_rank,
    )
    if tested_real:
        return tested_real[0]
    # Sicherer letzter Ausweg
    for d in real:
        if d.device == "cpu":
            return d
    return DeviceInfo("CPU", "CPU", "cpu", "torch", True, True, 100, "Fallback")


def should_use_half(device_string: str) -> bool:
    return device_string.startswith("cuda:")
