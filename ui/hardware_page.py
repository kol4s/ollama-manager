from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.hardware import Hardware


class HardwarePage(QWidget):
    """Página de información del hardware."""

    def __init__(self, hardware: Hardware, recommendations=None):
        super().__init__()

        layout = QVBoxLayout(self)

        self.recommendations = recommendations or []

        title = QLabel("Hardware")
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold;"
        )

        cpu = QLabel(
            f"CPU: {hardware.cpu.model} "
            f"({hardware.cpu.logical_cores} hilos)"
        )

        memory_gb = hardware.memory.total_bytes / (1024 ** 3)

        memory = QLabel(
            f"RAM: {memory_gb:.1f} GiB"
        )

        gpu_title = QLabel("GPUs:")
        gpu_title.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )

        layout.addWidget(title)
        layout.addWidget(cpu)
        layout.addWidget(memory)
        layout.addWidget(gpu_title)

        for index, gpu in enumerate(hardware.gpus, start=1):
            if gpu.vram_total_bytes:
                vram_gb = gpu.vram_total_bytes / (1024 ** 3)
                vram_text = f"{vram_gb:.1f} GiB VRAM"
            else:
                vram_text = "memoria compartida"

            gpu_label = QLabel(
                f"GPU {index}: {gpu.model} — "
                f"{vram_text} — "
                f"API: {', '.join(gpu.compute_api) or 'N/D'}"
            )

            layout.addWidget(gpu_label)

        recommendations_title = QLabel("Recomendaciones")
        recommendations_title.setStyleSheet(
            "font-size: 18px; font-weight: bold; margin-top: 15px;"
        )

        layout.addWidget(recommendations_title)

        if self.recommendations:
            for recommendation in self.recommendations:
                recommendation_label = QLabel(
                    f"• {recommendation.title}: "
                    f"{recommendation.explanation}"
                )
                recommendation_label.setWordWrap(True)
                layout.addWidget(recommendation_label)
        else:
            no_recommendations = QLabel(
                "No hay recomendaciones disponibles."
            )
            layout.addWidget(no_recommendations)

        layout.addStretch()
