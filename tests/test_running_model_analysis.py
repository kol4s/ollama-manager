from core.knowledge_base import KnowledgeBase
from core.ollama_client import OllamaRunningModel
from core.recommendation_engine import RecommendationEngine


def create_engine():
    kb = KnowledgeBase("knowledge/hardware_profiles.json")
    kb.load()
    return RecommendationEngine(kb)


def test_running_model_gpu_majority():
    model = OllamaRunningModel(
        name="gpu-model",
        size_bytes=1000,
        size_vram_bytes=1000,
        context_length=4096,
        digest="",
        details={},
        expires_at="",
    )

    recommendation = create_engine().analyze_running_model(model)

    assert recommendation.data["strategy"] == "gpu_majority"
    assert recommendation.data["vram_percentage"] == 100.0


def test_running_model_gpu_cpu_shared():
    model = OllamaRunningModel(
        name="shared-model",
        size_bytes=1000,
        size_vram_bytes=600,
        context_length=4096,
        digest="",
        details={},
        expires_at="",
    )

    recommendation = create_engine().analyze_running_model(model)

    assert recommendation.data["strategy"] == "gpu_cpu_shared"
    assert recommendation.data["vram_percentage"] == 60.0


def test_running_model_cpu_majority():
    model = OllamaRunningModel(
        name="cpu-model",
        size_bytes=1000,
        size_vram_bytes=440,
        context_length=4096,
        digest="",
        details={},
        expires_at="",
    )

    recommendation = create_engine().analyze_running_model(model)

    assert recommendation.data["strategy"] == "cpu_majority"
    assert recommendation.data["vram_percentage"] == 44.0
