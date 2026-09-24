from core.knowledge_base import KnowledgeBase
from core.model_info import ModelInfo
from core.recommendation_engine import RecommendationEngine


def create_engine():
    knowledge_base = KnowledgeBase(
        "knowledge/hardware_profiles.json"
    )
    knowledge_base.load()
    return RecommendationEngine(
        knowledge_base
    )


def test_model_analysis_contract():
    model = ModelInfo(
        name="test-model",
        parameters_b=4.0,
        quantization="Q4_K_M",
        context_length=65536,
        size_bytes=0,
    )

    assert create_engine().model_sizer.estimate(
        model,
        65536,
    ).total_bytes > 0
