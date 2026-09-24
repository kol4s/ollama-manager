from dataclasses import dataclass, field

from hardware_detector import HardwareDetector


@dataclass
class RuntimeGPU:
    vendor: str
    model: str
    vram_total_bytes: int
    vram_used_bytes: int
    vram_free_bytes: int
    temperature_c: float | None
    utilization_percent: float | None
    power_watts: float | None
    power_limit_watts: float | None

    @property
    def vram_used_percent(self) -> float:
        if self.vram_total_bytes <= 0:
            return 0.0

        return (
            self.vram_used_bytes
            / self.vram_total_bytes
        ) * 100.0


@dataclass
class RuntimeSnapshot:
    cpu_percent: float
    ram_total_bytes: int
    ram_used_bytes: int
    ram_available_bytes: int
    swap_total_bytes: int
    swap_used_bytes: int
    gpus: list[RuntimeGPU] = field(
        default_factory=list
    )

    @property
    def ram_used_percent(self) -> float:
        if self.ram_total_bytes <= 0:
            return 0.0

        return (
            self.ram_used_bytes
            / self.ram_total_bytes
        ) * 100.0

    @property
    def swap_used_percent(self) -> float:
        if self.swap_total_bytes <= 0:
            return 0.0

        return (
            self.swap_used_bytes
            / self.swap_total_bytes
        ) * 100.0


class RuntimeMonitor:
    """Obtiene una instantánea del estado de ejecución del sistema."""

    def __init__(self, detector=None):
        self.detector = detector or HardwareDetector()

    def snapshot(self) -> RuntimeSnapshot:
        hardware = self.detector.detect()

        cpu_percent = self._cpu_percent()

        gpus = [
            RuntimeGPU(
                vendor=gpu.vendor or "",
                model=gpu.model or "",
                vram_total_bytes=gpu.vram_total_bytes,
                vram_used_bytes=gpu.vram_used_bytes,
                vram_free_bytes=gpu.vram_free_bytes,
                temperature_c=gpu.temperature_c,
                utilization_percent=gpu.utilization_percent,
                power_watts=gpu.power_watts,
                power_limit_watts=gpu.power_limit_watts,
            )
            for gpu in hardware.gpus
        ]

        return RuntimeSnapshot(
            cpu_percent=cpu_percent,
            ram_total_bytes=hardware.memory.total_bytes,
            ram_used_bytes=hardware.memory.used_bytes,
            ram_available_bytes=hardware.memory.available_bytes,
            swap_total_bytes=hardware.memory.swap_total_bytes,
            swap_used_bytes=hardware.memory.swap_used_bytes,
            gpus=gpus,
        )

    @staticmethod
    def _cpu_percent() -> float:
        import psutil

        return float(
            psutil.cpu_percent(
                interval=None
            )
        )
