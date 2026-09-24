from dataclasses import dataclass

from core.runtime_monitor import (
    RuntimeMonitor,
)


@dataclass
class FakeMemory:
    total_bytes: int
    used_bytes: int
    available_bytes: int
    swap_total_bytes: int
    swap_used_bytes: int


@dataclass
class FakeGPU:
    vendor: str
    model: str
    vram_total_bytes: int
    vram_used_bytes: int
    vram_free_bytes: int
    temperature_c: float | None
    utilization_percent: float | None
    power_watts: float | None
    power_limit_watts: float | None


@dataclass
class FakeHardware:
    memory: FakeMemory
    gpus: list


class FakeDetector:
    def detect(self):
        return FakeHardware(
            memory=FakeMemory(
                total_bytes=48 * 1024**3,
                used_bytes=20 * 1024**3,
                available_bytes=28 * 1024**3,
                swap_total_bytes=8 * 1024**3,
                swap_used_bytes=1 * 1024**3,
            ),
            gpus=[
                FakeGPU(
                    vendor="NVIDIA",
                    model="RTX 3060",
                    vram_total_bytes=12 * 1024**3,
                    vram_used_bytes=8 * 1024**3,
                    vram_free_bytes=4 * 1024**3,
                    temperature_c=55.0,
                    utilization_percent=72.0,
                    power_watts=135.0,
                    power_limit_watts=170.0,
                )
            ],
        )


def test_runtime_snapshot():
    monitor = RuntimeMonitor(
        detector=FakeDetector()
    )

    monitor._cpu_percent = staticmethod(
        lambda: 35.0
    )

    snapshot = monitor.snapshot()

    assert snapshot.cpu_percent == 35.0
    assert snapshot.ram_total_bytes == 48 * 1024**3
    assert snapshot.ram_used_bytes == 20 * 1024**3
    assert snapshot.ram_available_bytes == 28 * 1024**3
    assert snapshot.ram_used_percent == (
        (20 / 48) * 100
    )

    assert len(snapshot.gpus) == 1

    gpu = snapshot.gpus[0]

    assert gpu.vendor == "NVIDIA"
    assert gpu.model == "RTX 3060"
    assert gpu.vram_used_percent == (
        (8 / 12) * 100
    )
    assert gpu.temperature_c == 55.0
    assert gpu.utilization_percent == 72.0
    assert gpu.power_watts == 135.0


def test_swap_percentage():
    monitor = RuntimeMonitor(
        detector=FakeDetector()
    )

    snapshot = monitor.snapshot()

    assert snapshot.swap_used_percent == (
        (1 / 8) * 100
    )
