from core.application_plan import ApplicationPlan
from core.configuration_proposal import (
    build_configuration_proposal,
)
from core.parameter_values import (
    build_proposed_parameters,
)
from core.model_variant import build_variant_name


def build_application_plan_from_comparisons(
    model,
    comparisons,
) -> ApplicationPlan:
    """
    Construye un plan seguro a partir de los valores editados
    en la tabla Actual/Recomendado/Propuesto.
    """

    proposed_parameters = (
        build_proposed_parameters(
            comparisons
        )
    )

    proposal = build_configuration_proposal(
        model.name,
        model.modelfile,
        model.current_parameters,
        proposed_parameters,
    )

    target_name = build_variant_name(
        model
    )

    from core.application_plan import (
        build_application_plan,
    )

    return build_application_plan(
        model.name,
        target_name,
        proposal.parameters,
        proposal.modelfile,
    )
