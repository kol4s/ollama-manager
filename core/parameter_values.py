from typing import Any


def coerce_parameter_value(
    value: str,
    reference: Any = None,
) -> Any:
    """
    Convierte el texto editado en la GUI al tipo apropiado.
    """

    text = str(value).strip()

    if reference is not None:
        if isinstance(reference, bool):
            lowered = text.lower()

            if lowered == "true":
                return True

            if lowered == "false":
                return False

            raise ValueError(
                f"debe ser true o false. Valor introducido: {value}"
            )

        if isinstance(reference, int) and not isinstance(
            reference,
            bool,
        ):
            try:
                return int(text)
            except ValueError as exc:
                raise ValueError(
                    f"debe ser un número entero. Valor introducido: {value}"
                ) from exc

        if isinstance(reference, float):
            try:
                return float(text)
            except ValueError as exc:
                raise ValueError(
                    f"debe ser un número. Valor introducido: {value}"
                ) from exc

    lowered = text.lower()

    if lowered == "true":
        return True

    if lowered == "false":
        return False

    try:
        if "." in text or "e" in lowered:
            return float(text)

        return int(text)
    except ValueError:
        return text


def build_proposed_parameters(
    comparisons,
) -> dict[str, Any]:
    """
    Extrae y tipa los valores de la columna Propuesto.
    """

    result: dict[str, Any] = {}

    for comparison in comparisons:
        # El valor recomendado define el tipo esperado del
        # parámetro. Esto evita que, por ejemplo, un valor actual
        # "1" fuerce temperature=0.7 a convertirse en entero.
        reference = comparison.recommended

        try:
            result[comparison.name] = coerce_parameter_value(
                comparison.proposed,
                reference,
            )
        except ValueError as exc:
            raise ValueError(
                f"Valor no válido para '{comparison.name}': {exc}"
            ) from exc

    return result
