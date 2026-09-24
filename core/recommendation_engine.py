from dataclasses import dataclass, field
from typing import Any

from core.hardware import Hardware
from core.knowledge_base import KnowledgeBase
from core.model_info import ModelInfo
from core.model_sizer import ModelMemoryEstimate, ModelSizer
from core.ollama_client import OllamaRunningModel


@dataclass
class Recommendation:
    title: str
    explanation: str
    severity: str = "info"
    data: dict[str, Any] = field(default_factory=dict)


class RecommendationEngine:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.model_sizer = ModelSizer(knowledge_base)

    def analyze(
        self,
        hardware: Hardware,
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []

        self._check_gpu(
            hardware,
            recommendations,
        )

        self._check_memory(
            hardware,
            recommendations,
        )

        self._check_multi_gpu(
            hardware,
            recommendations,
        )

        return recommendations

    def analyze_model(
        self,
        hardware: Hardware,
        model: ModelInfo,
        context_length: int | None = None,
    ) -> list[Recommendation]:
        """
        Analiza un modelo concreto sobre el hardware disponible.

        El contexto puede especificarse explícitamente. Si no se
        especifica, se utiliza el contexto declarado por el modelo.
        """

        recommendations: list[Recommendation] = []

        estimate = self.model_sizer.estimate(
            model,
            context_length,
        )

        dedicated_gpus = [
            gpu
            for gpu in hardware.gpus
            if gpu.dedicated
            and gpu.vram_total_bytes > 0
        ]

        if not dedicated_gpus:
            recommendations.append(
                Recommendation(
                    title="Ejecución sin GPU dedicada",
                    explanation=(
                        "No se ha detectado una GPU con VRAM dedicada "
                        "suficiente para evaluar una ejecución GPU."
                    ),
                    severity="warning",
                    data={
                        "strategy": "cpu",
                        "estimated_memory_bytes": estimate.total_bytes,
                    },
                )
            )

            return recommendations

        best_gpu = max(
            dedicated_gpus,
            key=lambda gpu: gpu.vram_total_bytes,
        )

        available_vram_bytes = (
            best_gpu.vram_free_bytes
            if best_gpu.vram_free_bytes > 0
            else best_gpu.vram_total_bytes
        )

        status = self.model_sizer.capacity_status(
            estimate,
            available_vram_bytes,
        )

        if status == "recommended":
            recommendations.append(
                Recommendation(
                    title="Ejecución recomendada en GPU",
                    explanation=(
                        f"El modelo {model.name} puede ejecutarse "
                        f"completamente en {best_gpu.model} con el "
                        f"contexto solicitado."
                    ),
                    severity="info",
                    data={
                        "strategy": "gpu",
                        "model": model.name,
                        "context_length": (
                            context_length
                            if context_length is not None
                            else model.context_length
                        ),
                        "gpu": best_gpu.model,
                        "vram_bytes": best_gpu.vram_total_bytes,
                        "vram_used_bytes": best_gpu.vram_used_bytes,
                        "vram_free_bytes": best_gpu.vram_free_bytes,
                        "available_vram_bytes": available_vram_bytes,
                        "estimated_memory_bytes": estimate.total_bytes,
                        "status": status,
                    },
                )
            )

        elif status == "tight":
            recommendations.append(
                Recommendation(
                    title="Ejecución GPU ajustada",
                    explanation=(
                        f"El modelo {model.name} puede entrar en la "
                        f"VRAM disponible, pero con poco margen de "
                        f"seguridad."
                    ),
                    severity="warning",
                    data={
                        "strategy": "gpu_tight",
                        "model": model.name,
                        "context_length": (
                            context_length
                            if context_length is not None
                            else model.context_length
                        ),
                        "gpu": best_gpu.model,
                        "vram_bytes": best_gpu.vram_total_bytes,
                        "vram_used_bytes": best_gpu.vram_used_bytes,
                        "vram_free_bytes": best_gpu.vram_free_bytes,
                        "available_vram_bytes": available_vram_bytes,
                        "estimated_memory_bytes": estimate.total_bytes,
                        "status": status,
                    },
                )
            )

        else:
            recommendations.append(
                Recommendation(
                    title="GPU insuficiente para residencia completa",
                    explanation=(
                        f"El modelo {model.name} no puede residir "
                        f"completamente en {best_gpu.model} con el "
                        f"contexto solicitado. Se debe considerar "
                        f"CPU offload o una estrategia multi-GPU."
                    ),
                    severity="warning",
                    data={
                        "strategy": "cpu_offload_or_multi_gpu",
                        "model": model.name,
                        "context_length": (
                            context_length
                            if context_length is not None
                            else model.context_length
                        ),
                        "gpu": best_gpu.model,
                        "vram_bytes": best_gpu.vram_total_bytes,
                        "vram_used_bytes": best_gpu.vram_used_bytes,
                        "vram_free_bytes": best_gpu.vram_free_bytes,
                        "available_vram_bytes": available_vram_bytes,
                        "estimated_memory_bytes": estimate.total_bytes,
                        "status": status,
                    },
                )
            )

        return recommendations

    def analyze_running_model(
        self,
        model: OllamaRunningModel,
    ) -> Recommendation:
        """Analiza cómo está distribuido realmente un modelo en ejecución."""

        vram_percentage = model.vram_percentage

        if vram_percentage >= 90.0:
            strategy = "gpu_majority"
            severity = "info"
            title = "Modelo ejecutándose principalmente en GPU"
            explanation = (
                f"El modelo {model.name} tiene "
                f"{vram_percentage:.2f}% de su memoria residente en VRAM."
            )

        elif vram_percentage >= 50.0:
            strategy = "gpu_cpu_shared"
            severity = "warning"
            title = "Modelo compartido entre GPU y CPU"
            explanation = (
                f"El modelo {model.name} utiliza "
                f"{vram_percentage:.2f}% de VRAM y "
                f"{100.0 - vram_percentage:.2f}% de memoria CPU/RAM."
            )

        else:
            strategy = "cpu_majority"
            severity = "warning"
            title = "Modelo ejecutándose principalmente en CPU/RAM"
            explanation = (
                f"El modelo {model.name} utiliza "
                f"{vram_percentage:.2f}% de VRAM y "
                f"{100.0 - vram_percentage:.2f}% de memoria CPU/RAM."
            )

        return Recommendation(
            title=title,
            explanation=explanation,
            severity=severity,
            data={
                "model": model.name,
                "strategy": strategy,
                "context_length": model.context_length,
                "size_bytes": model.size_bytes,
                "size_vram_bytes": model.size_vram_bytes,
                "cpu_memory_bytes": model.cpu_memory_bytes,
                "vram_percentage": vram_percentage,
            },
        )

    def _check_gpu(
        self,
        hardware: Hardware,
        recommendations: list[Recommendation],
    ) -> None:
        dedicated_gpus = [
            gpu
            for gpu in hardware.gpus
            if gpu.dedicated
            and gpu.vram_total_bytes > 0
        ]

        if not dedicated_gpus:
            recommendations.append(
                Recommendation(
                    title="Sin VRAM dedicada disponible",
                    explanation=(
                        "No se ha detectado una GPU con memoria "
                        "dedicada disponible para ejecutar modelos."
                    ),
                    severity="warning",
                )
            )
            return

        best_gpu = max(
            dedicated_gpus,
            key=lambda gpu: gpu.vram_total_bytes,
        )

        recommendations.append(
            Recommendation(
                title="GPU principal detectada",
                explanation=(
                    f"La GPU con mayor memoria dedicada es "
                    f"{best_gpu.model}."
                ),
                data={
                    "vendor": best_gpu.vendor,
                    "model": best_gpu.model,
                    "vram_bytes": best_gpu.vram_total_bytes,
                    "compute_api": best_gpu.compute_api,
                },
            )
        )

    def _check_memory(
        self,
        hardware: Hardware,
        recommendations: list[Recommendation],
    ) -> None:
        ram_profiles = self.knowledge_base.get(
            "memory",
            "ram",
            default={},
        )

        if not ram_profiles:
            return

        total_ram = hardware.memory.total_bytes

        if total_ram >= ram_profiles.get(
            "very_high",
            float("inf"),
        ):
            level = "very_high"

        elif total_ram >= ram_profiles.get(
            "high",
            float("inf"),
        ):
            level = "high"

        elif total_ram >= ram_profiles.get(
            "medium",
            float("inf"),
        ):
            level = "medium"

        else:
            level = "low"

        recommendations.append(
            Recommendation(
                title="Nivel de memoria RAM",
                explanation=(
                    f"La memoria RAM pertenece al nivel "
                    f"'{level}'."
                ),
                data={
                    "level": level,
                    "total_bytes": total_ram,
                },
            )
        )

    def _check_multi_gpu(
        self,
        hardware: Hardware,
        recommendations: list[Recommendation],
    ) -> None:
        dedicated_gpus = [
            gpu
            for gpu in hardware.gpus
            if gpu.dedicated
            and gpu.vram_total_bytes > 0
        ]

        if len(dedicated_gpus) <= 1:
            return

        multi_gpu = self.knowledge_base.get(
            "model_execution",
            "multi_gpu",
            default={},
        )

        if not multi_gpu.get(
            "enabled",
            False,
        ):
            return

        recommendations.append(
            Recommendation(
                title="Configuración multi-GPU disponible",
                explanation=(
                    "Se han detectado varias GPUs con memoria "
                    "dedicada. Se priorizará una sola GPU cuando "
                    "el modelo pueda residir completamente en ella "
                    "y se considerará dividirlo cuando sea necesario."
                ),
                data={
                    "gpu_count": len(dedicated_gpus),
                    "prefer_single_gpu": multi_gpu.get(
                        "prefer_single_gpu_when_model_fits",
                        False,
                    ),
                    "allow_split": multi_gpu.get(
                        "allow_split_when_required",
                        False,
                    ),
                },
            )
        )
