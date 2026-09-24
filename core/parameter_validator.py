from typing import Any


class ParameterValidationError(ValueError):
    pass


def _find_parameter_spec(
    parameter_name: str,
    knowledge_data: dict,
) -> dict | None:
    groups = knowledge_data.get(
        "ollama_parameters",
        {},
    )

    for parameters in groups.values():
        if parameter_name in parameters:
            return parameters[parameter_name]

    return None


def validate_parameter_value(
    name: str,
    value: Any,
    knowledge_data: dict,
) -> None:
    spec = _find_parameter_spec(
        name,
        knowledge_data,
    )

    if spec is None:
        return

    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    exclusive_minimum = spec.get(
        "exclusive_minimum"
    )

    if minimum is not None and value < minimum:
        raise ParameterValidationError(
            f"{name} debe ser >= {minimum}"
        )

    if maximum is not None and value > maximum:
        raise ParameterValidationError(
            f"{name} debe ser <= {maximum}"
        )

    if (
        exclusive_minimum is not None
        and value <= exclusive_minimum
    ):
        raise ParameterValidationError(
            f"{name} debe ser > {exclusive_minimum}"
        )


def validate_parameters(
    parameters: dict[str, Any],
    knowledge_data: dict,
) -> None:
    for name, value in parameters.items():
        validate_parameter_value(
            name,
            value,
            knowledge_data,
        )
