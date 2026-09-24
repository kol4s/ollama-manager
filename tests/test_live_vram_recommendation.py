from core.hardware import (
    CPU,
    GPU,
    Hardware,
    Memory,
    System,
)
from core.knowledge_base import KnowledgeBase
from core.model_info import ModelInfo
from core.recommendation_engine import RecommendationEngine


def create_hardware(
    total_vram_gib: int,
    free_vram_gib: int,
) -> Hardware:
    return Hardware(
        system=System(
            os="test",
            kernel="test",
            architecture="x86_64",
        ),
        cpu=CPU(
            vendor="Intel",
            model="Test CPU",
            physical_cores=24,
            logical_cores=24,
        ),
        memory=Memory(
            total_bytes=48 * 1024**3,
            available_bytes=33 * 1024**3,
            used_bytes=15 * 1024**3,
            swap_total_bytes=8 * 1024**3,
            swap_used_bytes=0,
        ),
        gpus=[
            GPU(
                vendor="NVIDIA",
                model="Test GPU",
                memory_type="dedicated",
                dedicated=True,
                vram_total_bytes=total_vram_gib * 1024**3,
                vram_used_bytes=(
                    (total_vram_gib - free_vram_gib)
                    * 1024**3
                ),
                vram_free_bytes=free_vram_gib * 1024**3,
                compute_api=["CUDA"],
            )
        ],
    )


def create_engine():
    kb = KnowledgeBase(
        "knowledge/hardware_profiles.json"
    )
    kb.load()
    return RecommendationEngine(kb)


def create_model() -> ModelInfo:
    return ModelInfo(
        name="test-model",
        parameters_b=8.0,
        quantization="Q4_K_M",
        context_length=65536,
    )


def test_free_vram_is_used_for_capacity():
    engine = create_engine()

    recommendations = engine.analyze_model(
        create_hardware(
            total_vram_gib=12,
            free_vram_gib=3,
        ),
        create_model(),
        context_length=4096,
    )

    recommendation = recommendations[-1]

    assert recommendation.data["vram_bytes"] == (
        12 * 1024**3
    )

    assert recommendation.data["vram_free_bytes"] == (
        3 * 1024**3
    )

    assert recommendation.data["available_vram_bytes"] == (
        3 * 1024**3
    )


def test_full_vram_is_used_when_free_vram_is_unknown():
    engine = create_engine()

    hardware = create_hardware(
        total_vram_gib=12,
        free_vram_gib=12,
    )

    hardware.gpus[0].vram_free_bytes = 0

    recommendations = engine.analyze_model(
        hardware,
        create_model(),
        context_length=4096,
    )

    recommendation = recommendations[-1]

    assert recommendation.data["available_vram_bytes"] == (
        12 * 1024**3
    )
