from dataclasses import dataclass

from core.model_info import ModelInfo
from core.knowledge_base import KnowledgeBase


@dataclass
class ModelMemoryEstimate:
    """Estimación de memoria necesaria para ejecutar un modelo."""

    model_bytes: int
    kv_cache_bytes: int
    runtime_overhead_bytes: int
    total_bytes: int


class ModelSizer:
    """Calcula una estimación conservadora de memoria del modelo."""

    QUANTIZATION_BYTES_PER_PARAMETER = {
        "Q2": 0.35,
        "Q3": 0.45,
        "Q4": 0.55,
        "Q5": 0.70,
        "Q6": 0.80,
        "Q8": 1.00,
        "FP16": 2.00,
        "F16": 2.00,
    }

    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self.knowledge_base = knowledge_base

    def estimate(
        self,
        model: ModelInfo,
        context_length: int | None = None,
    ) -> ModelMemoryEstimate:
        """
        Estima la memoria necesaria para ejecutar un modelo.

        Si context_length se especifica, se utiliza ese contexto
        en lugar del contexto máximo declarado por el modelo.
        """

        if context_length is None:
            effective_context = model.context_length
        else:
            effective_context = int(context_length)

        if effective_context <= 0:
            raise ValueError(
                "context_length must be greater than zero"
            )

        bytes_per_parameter = self._bytes_per_parameter(
            model.quantization
        )

        model_bytes = int(
            model.parameters_b
            * 1_000_000_000
            * bytes_per_parameter
        )

        kv_cache_bytes = self._estimate_kv_cache(
            model,
            effective_context,
        )

        runtime_overhead_bytes = int(
            model_bytes * 0.10
        )

        total_bytes = (
            model_bytes
            + kv_cache_bytes
            + runtime_overhead_bytes
        )

        return ModelMemoryEstimate(
            model_bytes=model_bytes,
            kv_cache_bytes=kv_cache_bytes,
            runtime_overhead_bytes=runtime_overhead_bytes,
            total_bytes=total_bytes,
        )

    def recommended_vram_bytes(
        self,
        vram_total_bytes: int,
    ) -> int:
        """Devuelve la VRAM utilizable aplicando el margen de seguridad."""

        margin = 0.10

        if self.knowledge_base is not None:
            margin = self.knowledge_base.get(
                "model_execution",
                "vram_safety_margin",
                default=0.10,
            )

        margin = max(
            0.0,
            min(float(margin), 0.90),
        )

        return int(
            vram_total_bytes * (1.0 - margin)
        )

    def capacity_status(
        self,
        estimate: ModelMemoryEstimate,
        vram_total_bytes: int,
    ) -> str:
        """Clasifica el modelo según la VRAM disponible."""

        recommended_limit = self.recommended_vram_bytes(
            vram_total_bytes
        )

        if estimate.total_bytes <= recommended_limit:
            return "recommended"

        if estimate.total_bytes <= vram_total_bytes:
            return "tight"

        return "not_fit"

    def _bytes_per_parameter(
        self,
        quantization: str,
    ) -> float:
        """Obtiene el tamaño medio estimado por parámetro."""

        normalized = quantization.upper().replace(
            "-",
            "",
        )

        for key, value in self.QUANTIZATION_BYTES_PER_PARAMETER.items():
            if normalized.startswith(key):
                return value

        return 1.0

    def _estimate_kv_cache(
        self,
        model: ModelInfo,
        context_length: int,
    ) -> int:
        """
        Estima la memoria del KV cache usando metadatos arquitectónicos.

        La estimación considera:
        - capas del transformer,
        - número de KV heads,
        - longitud de K/V,
        - contexto solicitado,
        - ventana deslizante cuando está disponible,
        - capas con KV compartido.

        Se utiliza FP16 (2 bytes) como referencia.
        """

        if (
            model.block_count <= 0
            or model.head_count_kv <= 0
            or model.key_length <= 0
            or model.value_length <= 0
        ):
            return 0

        effective_context = context_length

        if model.sliding_window > 0:
            effective_context = min(
                effective_context,
                model.sliding_window,
            )

        normal_kv_layers = model.block_count

        if model.shared_kv_layers > 0:
            normal_kv_layers = min(
                model.block_count,
                model.shared_kv_layers,
            )

        bytes_per_value = 2

        key_bytes = (
            normal_kv_layers
            * model.head_count_kv
            * model.key_length
            * effective_context
            * bytes_per_value
        )

        value_bytes = (
            normal_kv_layers
            * model.head_count_kv
            * model.value_length
            * effective_context
            * bytes_per_value
        )

        return int(
            key_bytes + value_bytes
        )
