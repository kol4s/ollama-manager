from dataclasses import dataclass
from typing import Any


@dataclass
class ApplicationPlan:
    """Plan de aplicación que todavía no ejecuta ninguna operación."""

    source_model: str
    target_model: str
    parameters: dict[str, Any]
    modelfile: str
    requires_confirmation: bool = True
    overwrite_source: bool = False

    @property
    def is_safe_default(self) -> bool:
        """
        La aplicación segura debe crear una variante y no sobrescribir
        el modelo de origen.
        """
        return (
            self.requires_confirmation
            and not self.overwrite_source
            and self.source_model != self.target_model
        )


def build_application_plan(
    source_model: str,
    target_model: str,
    parameters: dict[str, Any],
    modelfile: str,
) -> ApplicationPlan:
    source_model = source_model.strip()
    target_model = target_model.strip()

    if not source_model:
        raise ValueError(
            "Source model cannot be empty"
        )

    if not target_model:
        raise ValueError(
            "Target model cannot be empty"
        )

    if source_model == target_model:
        raise ValueError(
            "Target model must differ from source model"
        )

    return ApplicationPlan(
        source_model=source_model,
        target_model=target_model,
        parameters=dict(parameters),
        modelfile=modelfile,
        requires_confirmation=True,
        overwrite_source=False,
    )
