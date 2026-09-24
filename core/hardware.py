from dataclasses import dataclass, field


@dataclass
class GPU:
    vendor: str
    model: str
    memory_type: str
    dedicated: bool

    vram_total_bytes: int = 0
    vram_used_bytes: int = 0
    vram_free_bytes: int = 0

    pci_vendor_id: str | None = None
    pci_device_id: str | None = None

    drm_card: str | None = None
    drm_render_node: str | None = None

    driver: str | None = None
    compute_api: list[str] = field(default_factory=list)

    temperature_c: float | None = None
    utilization_percent: float | None = None
    power_watts: float | None = None
    power_limit_watts: float | None = None


@dataclass
class CPU:
    vendor: str
    model: str
    physical_cores: int
    logical_cores: int

    min_frequency_mhz: float | None = None
    max_frequency_mhz: float | None = None


@dataclass
class Memory:
    total_bytes: int
    available_bytes: int
    used_bytes: int

    swap_total_bytes: int = 0
    swap_used_bytes: int = 0


@dataclass
class System:
    os: str
    kernel: str
    architecture: str


@dataclass
class Hardware:
    system: System
    cpu: CPU
    memory: Memory
    gpus: list[GPU] = field(default_factory=list)


def from_detector_info(info) -> Hardware:
    """
    Convierte el resultado de hardware_detector.HardwareInfo
    al modelo normalizado utilizado por el resto de la aplicación.
    """

    system = System(
        os=info.system.os or "",
        kernel=info.system.kernel or "",
        architecture=info.system.architecture or "",
    )

    cpu = CPU(
        vendor=info.cpu.vendor or "",
        model=info.cpu.model or "",
        physical_cores=info.cpu.physical_cores or 0,
        logical_cores=info.cpu.logical_cores or 0,
        min_frequency_mhz=info.cpu.min_frequency_mhz,
        max_frequency_mhz=info.cpu.max_frequency_mhz,
    )

    memory = Memory(
        total_bytes=info.memory.total_bytes,
        available_bytes=info.memory.available_bytes,
        used_bytes=info.memory.used_bytes,
        swap_total_bytes=info.memory.swap_total_bytes,
        swap_used_bytes=info.memory.swap_used_bytes,
    )

    gpus = []

    for gpu_info in info.gpus:
        gpus.append(
            GPU(
                vendor=gpu_info.vendor or "",
                model=gpu_info.model or "",
                memory_type=gpu_info.memory_type or "",
                dedicated=gpu_info.dedicated,
                vram_total_bytes=gpu_info.vram_total_bytes,
                vram_used_bytes=gpu_info.vram_used_bytes,
                vram_free_bytes=gpu_info.vram_free_bytes,
                pci_vendor_id=gpu_info.pci_vendor_id,
                pci_device_id=gpu_info.pci_device_id,
                drm_card=gpu_info.drm_card,
                drm_render_node=gpu_info.drm_render_node,
                driver=gpu_info.driver,
                compute_api=list(gpu_info.compute_api),
                temperature_c=gpu_info.temperature_c,
                utilization_percent=gpu_info.utilization_percent,
                power_watts=gpu_info.power_watts,
                power_limit_watts=gpu_info.power_limit_watts,
            )
        )

    return Hardware(
        system=system,
        cpu=cpu,
        memory=memory,
        gpus=gpus,
    )
