from core.ollama_client import OllamaModel


def build_variant_name(
    model: OllamaModel,
    suffix: str = "optimized",
) -> str:
    """
    Genera un nombre de variante a partir del nombre del modelo.

    Ejemplo:
        gemma4:e4b -> gemma4:e4b-optimized
    """

    base = model.name.strip()

    if not base:
        raise ValueError("Model name cannot be empty")

    suffix = suffix.strip()

    if not suffix:
        raise ValueError("Suffix cannot be empty")

    return f"{base}-{suffix}"


def build_available_variant_name(
    model: OllamaModel,
    existing_names,
    suffix: str = "optimized",
) -> str:
    """
    Devuelve el primer nombre de variante disponible.

    Ejemplo:
        gemma4:e4b-optimized
        gemma4:e4b-optimized-2
        gemma4:e4b-optimized-3
    """

    existing = set(existing_names)

    base_name = build_variant_name(
        model,
        suffix=suffix,
    )

    if base_name not in existing:
        return base_name

    index = 2

    while True:
        candidate = f"{base_name}-{index}"

        if candidate not in existing:
            return candidate

        index += 1
