from dataclasses import dataclass
from typing import Any


@dataclass
class ParameterComparison:
    """Comparación entre valor actual, recomendado y propuesto."""

    name: str
    current: Any
    recommended: Any
    proposed: Any
    explanation: str
    category: str
    changed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "current": self.current,
            "recommended": self.recommended,
            "proposed": self.proposed,
            "explanation": self.explanation,
            "category": self.category,
            "changed": self.changed,
        }


def build_parameter_comparison(
    current_parameters: dict[str, Any],
    recommended_parameters,
) -> list[ParameterComparison]:
    """
    Construye la comparación entre la configuración actual
    y el perfil recomendado.

    Por defecto, el valor propuesto comienza siendo el recomendado.
    El usuario podrá modificarlo posteriormente en la GUI.
    """

    comparisons: list[ParameterComparison] = []

    for parameter in recommended_parameters:
        current = current_parameters.get(
            parameter.name,
            None,
        )

        recommended = parameter.value

        comparisons.append(
            ParameterComparison(
                name=parameter.name,
                current=current,
                recommended=recommended,
                proposed=recommended,
                explanation=parameter.explanation,
                category=parameter.category,
                changed=current != recommended,
            )
        )

    return comparisons
