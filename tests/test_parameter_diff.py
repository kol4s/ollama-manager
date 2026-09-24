from core.parameter_comparison import ParameterComparison
from core.parameter_diff import build_parameter_diff


def test_build_parameter_diff():
    comparisons = [
        ParameterComparison(
            name="temperature",
            current=1.0,
            recommended=0.7,
            proposed=0.7,
            explanation="Temperatura",
            category="generation",
            changed=True,
        ),
        ParameterComparison(
            name="num_ctx",
            current=65536,
            recommended=65536,
            proposed=65536,
            explanation="Contexto",
            category="memory",
            changed=False,
        ),
    ]

    changes = build_parameter_diff(comparisons)

    assert len(changes) == 1
    assert changes[0].name == "temperature"
    assert changes[0].current == 1.0
    assert changes[0].proposed == 0.7


def test_empty_parameter_diff():
    comparisons = [
        ParameterComparison(
            name="num_thread",
            current=24,
            recommended=24,
            proposed=24,
            explanation="Hilos",
            category="load",
            changed=False,
        ),
    ]

    changes = build_parameter_diff(comparisons)

    assert changes == []
