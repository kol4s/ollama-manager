from dataclasses import dataclass, field
from typing import Any

from core.hardware import Hardware
from core.knowledge_base import KnowledgeBase
from core.model_info import ModelInfo
from core.model_sizer import ModelSizer


@dataclass
class ParameterRecommendation:
    """Parámetro recomendado para una ejecución de Ollama."""

    name: str
    value: Any
    explanation: str
    category: str = "runtime"


@dataclass
class ModelParameterProfile:
    """Configuración recomendada completa para un modelo."""

    model: str
    context_length: int
    parameters: list[ParameterRecommendation] = field(
        default_factory=list
    )
    fit_status: str = "cpu"
    execution_strategy: str = "cpu"
    available_vram_bytes: int = 0
    estimated_memory_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            item.name: item.value
            for item in self.parameters
        }


class ParameterRecommender:
    """Genera recomendaciones de parámetros según hardware y modelo."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.sizer = ModelSizer(knowledge_base)

    def recommend(
        self,
        hardware: Hardware,
        model: ModelInfo,
        context_length: int | None = None,
    ) -> ModelParameterProfile:
        effective_context = (
            model.context_length
            if context_length is None
            else int(context_length)
        )

        if effective_context <= 0:
            raise ValueError(
                "context_length must be greater than zero"
            )

        estimate = self.sizer.estimate(
            model,
            effective_context,
        )

        dedicated_gpus = [
            gpu
            for gpu in hardware.gpus
            if gpu.dedicated and gpu.vram_total_bytes > 0
        ]

        best_gpu = None

        if dedicated_gpus:
            best_gpu = max(
                dedicated_gpus,
                key=lambda gpu: gpu.vram_total_bytes,
            )

        fit_status = "cpu"
        available_vram_bytes = 0

        if best_gpu is not None:
            available_vram_bytes = (
                best_gpu.vram_free_bytes
                if best_gpu.vram_free_bytes > 0
                else best_gpu.vram_total_bytes
            )

            fit_status = self.sizer.capacity_status(
                estimate,
                available_vram_bytes,
            )

        parameters: list[ParameterRecommendation] = []

        parameters.append(
            ParameterRecommendation(
                name="num_ctx",
                value=effective_context,
                explanation=(
                    "Ventana de contexto utilizada por la ejecución."
                ),
                category="memory",
            )
        )

        if fit_status == "recommended":
            batch = 512
        elif fit_status == "tight":
            batch = 256
        else:
            batch = 128

        parameters.append(
            ParameterRecommendation(
                name="num_batch",
                value=batch,
                explanation=(
                    "Tamaño del lote de procesamiento. Se reduce "
                    "cuando la memoria disponible es más ajustada."
                ),
                category="memory",
            )
        )

        if fit_status == "recommended":
            execution_strategy = "gpu"
            gpu_explanation = (
                "Permite a Ollama cargar las capas del modelo en GPU "
                "cuando la memoria disponible es suficiente."
            )
        elif fit_status == "tight":
            execution_strategy = "gpu_tight"
            gpu_explanation = (
                "Permite a Ollama gestionar la carga GPU, pero la "
                "memoria disponible es ajustada."
            )
        elif fit_status == "not_fit":
            execution_strategy = "cpu_offload_or_multi_gpu"
            gpu_explanation = (
                "Ollama puede gestionar automáticamente la carga GPU, "
                "pero la VRAM disponible no permite residencia completa; "
                "se debe considerar CPU offload o multi-GPU."
            )
        else:
            execution_strategy = "cpu"
            gpu_explanation = (
                "No se dispone de una GPU dedicada adecuada para "
                "residencia del modelo."
            )

        parameters.append(
            ParameterRecommendation(
                name="num_gpu",
                value=-1,
                explanation=gpu_explanation,
                category="load",
            )
        )

        num_threads = max(
            1,
            int(hardware.cpu.logical_cores),
        )

        parameters.append(
            ParameterRecommendation(
                name="num_thread",
                value=num_threads,
                explanation=(
                    "Número recomendado de hilos de CPU basado "
                    "en los hilos lógicos detectados en el sistema."
                ),
                category="load",
            )
        )

        parameters.append(
            ParameterRecommendation(
                name="temperature",
                value=0.7,
                explanation=(
                    "Equilibrio general entre determinismo y variedad."
                ),
                category="generation",
            )
        )

        parameters.append(
            ParameterRecommendation(
                name="top_k",
                value=40,
                explanation=(
                    "Limita el conjunto de candidatos durante "
                    "la generación."
                ),
                category="generation",
            )
        )

        parameters.append(
            ParameterRecommendation(
                name="top_p",
                value=0.9,
                explanation=(
                    "Controla la masa acumulada de probabilidad."
                ),
                category="generation",
            )
        )

        parameters.append(
            ParameterRecommendation(
                name="min_p",
                value=0.05,
                explanation=(
                    "Descarta tokens con probabilidad relativa "
                    "demasiado baja."
                ),
                category="generation",
            )
        )

        parameters.append(
            ParameterRecommendation(
                name="repeat_penalty",
                value=1.1,
                explanation=(
                    "Reduce repeticiones excesivas."
                ),
                category="generation",
            )
        )

        parameters.append(
            ParameterRecommendation(
                name="repeat_last_n",
                value=64,
                explanation=(
                    "Ventana utilizada para detectar repeticiones."
                ),
                category="generation",
            )
        )

        parameters.append(
            ParameterRecommendation(
                name="num_predict",
                value=8192,
                explanation=(
                    "Límite máximo de tokens generados."
                ),
                category="generation",
            )
        )

        parameters.append(
            ParameterRecommendation(
                name="seed",
                value=0,
                explanation=(
                    "Semilla de generación."
                ),
                category="generation",
            )
        )

        parameters.append(
            ParameterRecommendation(
                name="keep_alive",
                value="10m",
                explanation=(
                    "Tiempo recomendado para mantener el modelo cargado."
                ),
                category="lifecycle",
            )
        )

        return ModelParameterProfile(
            model=model.name,
            context_length=effective_context,
            parameters=parameters,
            fit_status=fit_status,
            execution_strategy=execution_strategy,
            available_vram_bytes=available_vram_bytes,
            estimated_memory_bytes=estimate.total_bytes,
        )
