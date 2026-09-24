from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Información detallada de un modelo de Ollama."""

    name: str
    parameters_b: float
    quantization: str
    context_length: int
    size_bytes: int = 0

    architecture: str = ""
    parameter_count: int = 0
    embedding_length: int = 0
    block_count: int = 0
    head_count: int = 0
    head_count_kv: int = 0

    key_length: int = 0
    value_length: int = 0

    key_length_swa: int = 0
    value_length_swa: int = 0

    sliding_window: int = 0
    sliding_window_pattern: str = ""

    shared_kv_layers: int = 0

    @property
    def parameters_millions(self) -> float:
        """Devuelve el número de parámetros en millones."""
        return self.parameters_b * 1000
