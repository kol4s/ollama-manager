from hardware_detector import HardwareDetector
from core.knowledge_base import KnowledgeBase
from core.recommendation_engine import RecommendationEngine


def test_recommendation_engine():
    detector = HardwareDetector()
    hardware = detector.detect_normalized()

    kb = KnowledgeBase("knowledge/hardware_profiles.json")
    kb.load()

    engine = RecommendationEngine(kb)
    recommendations = engine.analyze(hardware)

    assert recommendations
    assert any(
        recommendation.title == "GPU principal detectada"
        for recommendation in recommendations
    )
    assert any(
        recommendation.title == "Nivel de memoria RAM"
        for recommendation in recommendations
    )
