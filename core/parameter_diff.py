from dataclasses import dataclass
from typing import Any


@dataclass
class ParameterChange:
    name: str
    current: Any
    proposed: Any
    category: str
    explanation: str


def build_parameter_diff(
    comparisons,
) -> list[ParameterChange]:
    changes: list[ParameterChange] = []

    for item in comparisons:
        if item.current != item.proposed:
            changes.append(
                ParameterChange(
                    name=item.name,
                    current=item.current,
                    proposed=item.proposed,
                    category=item.category,
                    explanation=item.explanation,
                )
            )

    return changes
