from core.application_builder import (
    build_application_plan_from_comparisons,
)
from core.parameter_comparison import ParameterComparison
from core.ollama_client import OllamaModel


def create_model():
    return OllamaModel(
        name="gemma4:e4b",
        size_bytes=100,
        details={},
        current_parameters={
            "temperature": 1.0,
            "top_k": 64,
        },
        modelfile=(
            "FROM gemma4:e4b\n"
            "PARAMETER temperature 1\n"
            "PARAMETER top_k 64\n"
        ),
    )


def test_build_application_plan_from_comparisons():
    comparisons = [
        ParameterComparison(
            name="temperature",
            current=1.0,
            recommended=0.7,
            proposed="0.7",
            explanation="",
            category="generation",
            changed=True,
        ),
        ParameterComparison(
            name="top_k",
            current=64,
            recommended=40,
            proposed="40",
            explanation="",
            category="generation",
            changed=True,
        ),
    ]

    plan = build_application_plan_from_comparisons(
        create_model(),
        comparisons,
    )

    assert plan.source_model == "gemma4:e4b"
    assert plan.target_model == (
        "gemma4:e4b-optimized"
    )
    assert plan.parameters["temperature"] == 0.7
    assert plan.parameters["top_k"] == 40
    assert plan.is_safe_default is True
