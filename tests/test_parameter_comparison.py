from core.parameter_comparison import (
    ParameterComparison,
    build_parameter_comparison,
)
from core.parameter_recommender import ParameterRecommendation


def test_build_parameter_comparison():
    recommended = [
        ParameterRecommendation(
            name="temperature",
            value=0.7,
            explanation="Temperatura.",
            category="generation",
        ),
        ParameterRecommendation(
            name="num_ctx",
            value=65536,
            explanation="Contexto.",
            category="memory",
        ),
    ]

    current = {
        "temperature": 1.0,
        "num_ctx": 65536,
    }

    comparisons = build_parameter_comparison(
        current,
        recommended,
    )

    assert len(comparisons) == 2

    temperature = comparisons[0]

    assert temperature.current == 1.0
    assert temperature.recommended == 0.7
    assert temperature.proposed == 0.7
    assert temperature.changed is True

    context = comparisons[1]

    assert context.current == 65536
    assert context.recommended == 65536
    assert context.proposed == 65536
    assert context.changed is False


def test_missing_current_parameter():
    recommended = [
        ParameterRecommendation(
            name="num_thread",
            value=24,
            explanation="Hilos CPU.",
            category="load",
        ),
    ]

    comparisons = build_parameter_comparison(
        {},
        recommended,
    )

    assert len(comparisons) == 1
    assert comparisons[0].current is None
    assert comparisons[0].recommended == 24
    assert comparisons[0].proposed == 24
    assert comparisons[0].changed is True
