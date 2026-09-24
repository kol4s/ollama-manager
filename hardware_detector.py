import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil
import pynvml


# =============================================================
# INFORMACIÓN DE CPU
# =============================================================

@dataclass
class CPUInfo:
    vendor: Optional[str] = None
    model: Optional[str] = None
    physical_cores: Optional[int] = None
    logical_cores: Optional[int] = None
    min_frequency_mhz: Optional[float] = None
    max_frequency_mhz: Optional[float] = None


# =============================================================
# INFORMACIÓN DE MEMORIA
# =============================================================

@dataclass
class MemoryInfo:
    total_bytes: int = 0
    available_bytes: int = 0
    used_bytes: int = 0
    swap_total_bytes: int = 0
    swap_used_bytes: int = 0


# =============================================================
# INFORMACIÓN DE GPU
# =============================================================

@dataclass
class GPUInfo:
    vendor: Optional[str] = None
    model: Optional[str] = None

    # ---------------------------------------------------------
    # TIPO DE MEMORIA
    #
    # dedicated = VRAM dedicada
    # shared    = memoria compartida con RAM del sistema
    # ---------------------------------------------------------

    memory_type: Optional[str] = None

    dedicated: bool = False

    # ---------------------------------------------------------
    # VRAM
    #
    # En una iGPU normalmente permanecerá en 0 porque no
    # existe VRAM dedicada.
    # ---------------------------------------------------------

    vram_total_bytes: int = 0
    vram_used_bytes: int = 0
    vram_free_bytes: int = 0

    # ---------------------------------------------------------
    # IDENTIFICACIÓN PCI
    # ---------------------------------------------------------

    pci_vendor_id: Optional[str] = None
    pci_device_id: Optional[str] = None

    # ---------------------------------------------------------
    # LINUX DRM
    # ---------------------------------------------------------

    drm_card: Optional[str] = None
    drm_render_node: Optional[str] = None

    # ---------------------------------------------------------
    # DRIVER
    # ---------------------------------------------------------

    driver: Optional[str] = None

    # ---------------------------------------------------------
    # APIs / CAPACIDADES DE CÓMPUTO
    # ---------------------------------------------------------

    compute_api: list[str] = field(default_factory=list)

    # ---------------------------------------------------------
    # TELEMETRÍA
    # ---------------------------------------------------------

    temperature_c: Optional[float] = None
    utilization_percent: Optional[float] = None
    power_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None


# =============================================================
# INFORMACIÓN DEL SISTEMA
# =============================================================

@dataclass
class SystemInfo:
    os: Optional[str] = None
    kernel: Optional[str] = None
    architecture: Optional[str] = None


# =============================================================
# INFORMACIÓN COMPLETA DEL HARDWARE
# =============================================================

@dataclass
class HardwareInfo:
    system: SystemInfo = field(default_factory=SystemInfo)
    cpu: CPUInfo = field(default_factory=CPUInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    gpus: list[GPUInfo] = field(default_factory=list)


# =============================================================
# DETECTOR PRINCIPAL
# =============================================================

class HardwareDetector:

    def detect(self) -> HardwareInfo:
        hardware = HardwareInfo()

        self._detect_system(hardware)
        self._detect_cpu(hardware)
        self._detect_memory(hardware)

        # NVIDIA mediante NVML
        self._detect_nvidia(hardware)
        self._detect_intel(hardware)

        return hardware

    def detect_normalized(self):
        """
        Detecta el hardware y devuelve el modelo normalizado
        utilizado por el resto de la aplicación.
        """
        from core.hardware import from_detector_info

        detected = self.detect()

        return from_detector_info(detected)

    # =========================================================
    # SISTEMA OPERATIVO
    # =========================================================

    def _detect_system(self, hardware: HardwareInfo) -> None:

        hardware.system.os = platform.platform()
        hardware.system.kernel = platform.release()
        hardware.system.architecture = platform.machine()

    # =========================================================
    # CPU
    # =========================================================

    def _detect_cpu(self, hardware: HardwareInfo) -> None:

        hardware.cpu.physical_cores = psutil.cpu_count(
            logical=False
        )

        hardware.cpu.logical_cores = psutil.cpu_count(
            logical=True
        )

        frequency = psutil.cpu_freq()

        if frequency:

            hardware.cpu.min_frequency_mhz = frequency.min
            hardware.cpu.max_frequency_mhz = frequency.max

        # -----------------------------------------------------
        # Linux: /proc/cpuinfo
        # -----------------------------------------------------

        if platform.system() == "Linux":

            try:

                with open(
                    "/proc/cpuinfo",
                    "r",
                    encoding="utf-8",
                ) as file:

                    cpuinfo = file.read()

                for line in cpuinfo.splitlines():

                    if line.lower().startswith("vendor_id"):

                        try:

                            hardware.cpu.vendor = (
                                line.split(":", 1)[1].strip()
                            )

                        except IndexError:
                            pass

                    elif line.lower().startswith("model name"):

                        try:

                            hardware.cpu.model = (
                                line.split(":", 1)[1].strip()
                            )

                        except IndexError:
                            pass

                    if (
                        hardware.cpu.vendor
                        and hardware.cpu.model
                    ):
                        break

            except OSError:
                pass

        # -----------------------------------------------------
        # Fallback
        # -----------------------------------------------------

        if not hardware.cpu.model:

            processor = platform.processor()

            if processor:
                hardware.cpu.model = processor

    # =========================================================
    # MEMORIA RAM Y SWAP
    # =========================================================

    def _detect_memory(self, hardware: HardwareInfo) -> None:

        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        hardware.memory.total_bytes = memory.total
        hardware.memory.available_bytes = memory.available
        hardware.memory.used_bytes = memory.used

        hardware.memory.swap_total_bytes = swap.total
        hardware.memory.swap_used_bytes = swap.used

    def _get_vulkan_devices(self) -> list[dict[str, str]]:
        """
        Obtiene información de las GPUs físicas expuestas por Vulkan.

        Ignora dispositivos de tipo CPU, como llvmpipe.
        """
        devices: list[dict[str, str]] = []

        try:
            result = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return devices

        if result.returncode != 0:
            return devices

        current: Optional[dict[str, str]] = None

        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()

            if line.startswith("GPU") and line.endswith(":"):
                if current and current.get("deviceType") != "PHYSICAL_DEVICE_TYPE_CPU":
                    devices.append(current)

                current = {}
                continue

            if current is None or "=" not in line:
                continue

            key, value = line.split("=", 1)
            current[key.strip()] = value.strip()

        if current and current.get("deviceType") != "PHYSICAL_DEVICE_TYPE_CPU":
            devices.append(current)

        return devices

    # =========================================================
    # INTEL
    # =========================================================

    def _detect_intel(self, hardware: HardwareInfo) -> None:
        """
        Detecta GPUs Intel mediante Linux DRM/sysfs.

        No presupone que Intel sea card0 o card1.
        La identificación se realiza mediante el PCI vendor ID.
        """

        drm_path = Path("/sys/class/drm")

        if not drm_path.exists():
            return

        existing_pci_ids = {
            (
                gpu.pci_vendor_id,
                gpu.pci_device_id,
            )
            for gpu in hardware.gpus
        }

        for card_path in sorted(drm_path.glob("card[0-9]*")):

            if "-" in card_path.name:
                continue

            device_path = card_path / "device"

            if not device_path.exists():
                continue

            try:
                vendor_id = (
                    (device_path / "vendor")
                    .read_text(encoding="utf-8")
                    .strip()
                    .lower()
                )
            except OSError:
                continue

            # Intel PCI vendor ID
            if vendor_id != "0x8086":
                continue

            try:
                device_id = (
                    (device_path / "device")
                    .read_text(encoding="utf-8")
                    .strip()
                    .lower()
                )
            except OSError:
                device_id = None

            pci_key = (vendor_id, device_id)

            # Evitar duplicados
            if pci_key in existing_pci_ids:
                continue

            # -------------------------------------------------
            # DRIVER
            # -------------------------------------------------

            driver = None

            try:
                driver_link = device_path / "driver"

                if driver_link.exists():
                    driver = driver_link.resolve().name

            except OSError:
                pass

            # -------------------------------------------------
            # MODELO / DRIVER MEDIANTE VULKAN
            # -------------------------------------------------

            model = "Intel Graphics"

            vulkan_devices = self._get_vulkan_devices()

            for vulkan_device in vulkan_devices:

                vulkan_vendor = (
                    vulkan_device.get("vendorID", "")
                    .strip()
                    .lower()
                )

                vulkan_device_id = (
                    vulkan_device.get("deviceID", "")
                    .strip()
                    .lower()
                )

                if (
                    vulkan_vendor == vendor_id
                    and vulkan_device_id == device_id
                ):

                    vulkan_model = (
                        vulkan_device.get("deviceName")
                    )

                    if vulkan_model:
                        model = vulkan_model.strip()

                    vulkan_driver = (
                        vulkan_device.get("driverName")
                    )

                    vulkan_driver_info = (
                        vulkan_device.get("driverInfo")
                    )

                    if vulkan_driver:
                        driver = vulkan_driver.strip()

                        if vulkan_driver_info:
                            driver = (
                                f"{driver} "
                                f"({vulkan_driver_info.strip()})"
                            )

                    break

            # -------------------------------------------------
            # RENDER NODE
            # -------------------------------------------------

            render_node = None

            try:

                for render_path in sorted(
                    drm_path.glob("renderD*")
                ):

                    render_device = render_path / "device"

                    if not render_device.exists():
                        continue

                    try:

                        resolved = render_device.resolve()
                        resolved_device = device_path.resolve()

                        if resolved == resolved_device:
                            render_node = render_path.name
                            break

                    except OSError:
                        continue

            except OSError:
                pass

            # -------------------------------------------------
            # GPU INFO
            # -------------------------------------------------

            gpu = GPUInfo(
                vendor="Intel",
                model=model,
                memory_type="shared",
                dedicated=False,
                vram_total_bytes=0,
                vram_used_bytes=0,
                vram_free_bytes=0,
                pci_vendor_id=vendor_id,
                pci_device_id=device_id,
                drm_card=card_path.name,
                drm_render_node=render_node,
                driver=driver,
                compute_api=["Vulkan"],
            )

            hardware.gpus.append(gpu)
            existing_pci_ids.add(pci_key)


    # =========================================================
    # NVIDIA
    # =========================================================

    def _detect_nvidia(self, hardware: HardwareInfo) -> None:

        try:

            pynvml.nvmlInit()

        except Exception:

            return

        try:

            gpu_count = pynvml.nvmlDeviceGetCount()

            for index in range(gpu_count):

                handle = (
                    pynvml.nvmlDeviceGetHandleByIndex(index)
                )

                gpu = GPUInfo(
                    vendor="NVIDIA",
                    dedicated=True,
                    memory_type="dedicated",
                    compute_api=["CUDA"],
                )

                # -------------------------------------------------
                # IDENTIFICACIÓN PCI
                # -------------------------------------------------

                try:

                    pci_info = (
                        pynvml.nvmlDeviceGetPciInfo(handle)
                    )

                    pci_bus_id = pci_info.busId

                    if isinstance(pci_bus_id, bytes):
                        pci_bus_id = pci_bus_id.decode(
                            "utf-8",
                            errors="replace",
                        )

                    pci_bus_id = pci_bus_id.strip()

                    # NVML puede devolver el dominio PCI con 8 dígitos
                    # (ej. 00000000:02:00.0), mientras que Linux sysfs
                    # utiliza normalmente 4 dígitos (0000:02:00.0).
                    if (
                        pci_bus_id.count(":") == 2
                        and len(pci_bus_id.split(":", 1)[0]) == 8
                    ):
                        pci_bus_id = pci_bus_id[4:]

                    pci_device_path = (
                        Path("/sys/bus/pci/devices")
                        / pci_bus_id
                    )

                    if pci_device_path.exists():

                        try:
                            gpu.pci_vendor_id = (
                                (pci_device_path / "vendor")
                                .read_text(
                                    encoding="utf-8"
                                )
                                .strip()
                                .lower()
                            )
                        except OSError:
                            pass

                        try:
                            gpu.pci_device_id = (
                                (pci_device_path / "device")
                                .read_text(
                                    encoding="utf-8"
                                )
                                .strip()
                                .lower()
                            )
                        except OSError:
                            pass

                except Exception:
                    pass

                # -------------------------------------------------
                # MODELO
                # -------------------------------------------------

                try:

                    gpu.model = (
                        pynvml.nvmlDeviceGetName(handle)
                    )

                    if isinstance(gpu.model, bytes):

                        gpu.model = gpu.model.decode(
                            "utf-8",
                            errors="replace",
                        )

                except Exception:
                    pass

                # -------------------------------------------------
                # VRAM
                # -------------------------------------------------

                try:

                    memory = (
                        pynvml.nvmlDeviceGetMemoryInfo(
                            handle
                        )
                    )

                    gpu.vram_total_bytes = memory.total
                    gpu.vram_used_bytes = memory.used
                    gpu.vram_free_bytes = memory.free

                except Exception:
                    pass

                # -------------------------------------------------
                # DRIVER
                # -------------------------------------------------

                try:

                    gpu.driver = (
                        pynvml.nvmlSystemGetDriverVersion()
                    )

                    if isinstance(gpu.driver, bytes):

                        gpu.driver = gpu.driver.decode(
                            "utf-8",
                            errors="replace",
                        )

                except Exception:
                    pass

                # -------------------------------------------------
                # TEMPERATURA
                # -------------------------------------------------

                try:

                    gpu.temperature_c = (
                        pynvml.nvmlDeviceGetTemperature(
                            handle,
                            pynvml.NVML_TEMPERATURE_GPU,
                        )
                    )

                except Exception:
                    pass

                # -------------------------------------------------
                # UTILIZACIÓN
                # -------------------------------------------------

                try:

                    utilization = (
                        pynvml.nvmlDeviceGetUtilizationRates(
                            handle
                        )
                    )

                    gpu.utilization_percent = (
                        utilization.gpu
                    )

                except Exception:
                    pass

                # -------------------------------------------------
                # CONSUMO
                # -------------------------------------------------

                try:

                    gpu.power_watts = (
                        pynvml.nvmlDeviceGetPowerUsage(
                            handle
                        )
                        / 1000.0
                    )

                except Exception:
                    pass

                # -------------------------------------------------
                # LÍMITE DE POTENCIA
                # -------------------------------------------------

                try:

                    gpu.power_limit_watts = (
                        pynvml.nvmlDeviceGetPowerManagementLimit(
                            handle
                        )
                        / 1000.0
                    )

                except Exception:
                    pass

                # -------------------------------------------------
                # LINUX DRM
                # -------------------------------------------------

                # -------------------------------------------------
                # RESOLVER DRM MEDIANTE PCI BUS ID
                # -------------------------------------------------

                try:

                    pci_info = (
                        pynvml.nvmlDeviceGetPciInfo(handle)
                    )

                    pci_bus_id = pci_info.busId

                    if isinstance(pci_bus_id, bytes):
                        pci_bus_id = pci_bus_id.decode(
                            "utf-8",
                            errors="replace",
                        )

                    pci_bus_id = pci_bus_id.strip()

                    if (
                        pci_bus_id.count(":") == 2
                        and len(pci_bus_id.split(":", 1)[0]) == 8
                    ):
                        pci_bus_id = pci_bus_id[4:]

                    pci_device_path = (
                        Path("/sys/bus/pci/devices")
                        / pci_bus_id
                    )

                    resolved_pci = pci_device_path.resolve()

                    drm_path = Path("/sys/class/drm")

                    for card_path in sorted(
                        drm_path.glob("card[0-9]*")
                    ):

                        if "-" in card_path.name:
                            continue

                        device_path = card_path / "device"

                        if not device_path.exists():
                            continue

                        try:
                            if device_path.resolve() == resolved_pci:
                                gpu.drm_card = card_path.name
                                break
                        except OSError:
                            continue

                    for render_path in sorted(
                        drm_path.glob("renderD*")
                    ):

                        device_path = render_path / "device"

                        if not device_path.exists():
                            continue

                        try:
                            if device_path.resolve() == resolved_pci:
                                gpu.drm_render_node = render_path.name
                                break
                        except OSError:
                            continue

                except Exception:
                    pass

                # -------------------------------------------------
                # Añadir GPU
                # -------------------------------------------------

                hardware.gpus.append(gpu)

        finally:

            try:

                pynvml.nvmlShutdown()

            except Exception:
                pass


# =============================================================
# PRUEBA DEL DETECTOR
# =============================================================

if __name__ == "__main__":

    detector = HardwareDetector()

    hardware = detector.detect()

    print(hardware)
