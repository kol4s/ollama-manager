from core.model_variant import build_variant_name
from core.ollama_client import OllamaModel


def test_build_variant_name():
    model = OllamaModel(
        name="gemma4:e4b",
        size_bytes=0,
        details={},
    )

    assert build_variant_name(model) == (
        "gemma4:e4b-optimized"
    )


def test_build_variant_name_custom_suffix():
    model = OllamaModel(
        name="qwen3.8:27b",
        size_bytes=0,
        details={},
    )

    assert build_variant_name(
        model,
        "65k-tuned",
    ) == "qwen3.8:27b-65k-tuned"


def test_empty_model_name():
    model = OllamaModel(
        name="",
        size_bytes=0,
        details={},
    )

    try:
        build_variant_name(model)
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )


def test_build_available_variant_name_first_free():
    from core.ollama_client import OllamaModel
    from core.model_variant import build_available_variant_name

    model = OllamaModel(
        name="gemma4:e4b",
        size_bytes=0,
        details={},
    )

    existing = {
        "gemma4:e4b-optimized",
        "gemma4:e4b-optimized-2",
    }

    result = build_available_variant_name(
        model,
        existing_names=existing,
    )

    assert result == "gemma4:e4b-optimized-3"


def test_build_available_variant_name_uses_base_when_free():
    from core.ollama_client import OllamaModel
    from core.model_variant import build_available_variant_name

    model = OllamaModel(
        name="gemma4:e4b",
        size_bytes=0,
        details={},
    )

    result = build_available_variant_name(
        model,
        existing_names=set(),
    )

    assert result == "gemma4:e4b-optimized"


def test_build_available_variant_name_skips_multiple_existing_names():
    from core.ollama_client import OllamaModel
    from core.model_variant import build_available_variant_name

    model = OllamaModel(
        name="gemma4:e4b",
        size_bytes=0,
        details={},
    )

    existing = {
        "gemma4:e4b-optimized",
        "gemma4:e4b-optimized-2",
        "gemma4:e4b-optimized-3",
    }

    result = build_available_variant_name(
        model,
        existing_names=existing,
    )

    assert result == "gemma4:e4b-optimized-4"
