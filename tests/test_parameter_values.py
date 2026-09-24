from core.parameter_comparison import ParameterComparison
from core.parameter_values import (
    build_proposed_parameters,
    coerce_parameter_value,
)


def test_coerce_integer():
    assert coerce_parameter_value(
        "24",
        12,
    ) == 24


def test_coerce_float():
    assert coerce_parameter_value(
        "0.7",
        1.0,
    ) == 0.7


def test_coerce_boolean():
    assert coerce_parameter_value(
        "true",
        False,
    ) is True


def test_coerce_string():
    assert coerce_parameter_value(
        "10m",
        "5m",
    ) == "10m"


def test_build_proposed_parameters():
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
            name="num_thread",
            current=24,
            recommended=24,
            proposed="24",
            explanation="",
            category="load",
            changed=False,
        ),
    ]

    values = build_proposed_parameters(
        comparisons
    )

    assert values["temperature"] == 0.7
    assert values["num_thread"] == 24
    assert isinstance(
        values["temperature"],
        float,
    )
    assert isinstance(
        values["num_thread"],
        int,
    )



def test_recommended_type_has_priority_over_current_type():
    comparison = ParameterComparison(
        name="temperature",
        current=1,
        recommended=0.7,
        proposed="0.7",
        explanation="",
        category="generation",
        changed=True,
    )

    values = build_proposed_parameters(
        [comparison]
    )

    assert values["temperature"] == 0.7
    assert isinstance(
        values["temperature"],
        float,
    )
