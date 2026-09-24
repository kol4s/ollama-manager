from dataclasses import dataclass, field
from typing import Any

from core.modelfile_builder import build_modelfile


@dataclass
class ConfigurationProposal:
    """Representa una configuración que todavía no se ha aplicado."""

    model: str
    parameters: dict[str, Any]
    modelfile: str
    changes: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.changes)


def build_configuration_proposal(
    model: str,
    current_modelfile: str,
    current_parameters: dict[str, Any],
    proposed_parameters: dict[str, Any],
) -> ConfigurationProposal:
    changes: list[str] = []

    all_names = sorted(
        set(current_parameters)
        | set(proposed_parameters)
    )

    for name in all_names:
        current = current_parameters.get(name)
        proposed = proposed_parameters.get(name)

        if current != proposed:
            changes.append(
                f"{name}: {current} -> {proposed}"
            )

    modelfile = build_modelfile(
        model,
        current_modelfile,
        proposed_parameters,
    )

    return ConfigurationProposal(
        model=model,
        parameters=dict(proposed_parameters),
        modelfile=modelfile,
        changes=changes,
    )
