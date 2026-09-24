from dataclasses import dataclass
from typing import Any, Callable

from core.application_plan import ApplicationPlan


@dataclass
class ApplicationResult:
    """Resultado de una aplicación de configuración."""

    source_model: str
    target_model: str
    success: bool
    status: str
    details: dict[str, Any]


def apply_application_plan(
    client,
    plan: ApplicationPlan,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ApplicationResult:
    """
    Ejecuta un ApplicationPlan.

    La función exige que el plan sea seguro y confirmado externamente.
    No sobrescribe nunca el modelo de origen.
    """

    if not plan.is_safe_default:
        raise ValueError(
            "Application plan is not safe to execute"
        )

    result = client.create_model(
        plan.target_model,
        plan.modelfile,
        progress_callback=progress_callback,
    )

    status = str(
        result.get("status", "")
    )

    success = (
        status.lower() == "success"
        or bool(result.get("success", False))
    )

    return ApplicationResult(
        source_model=plan.source_model,
        target_model=plan.target_model,
        success=success,
        status=status,
        details=result,
    )
