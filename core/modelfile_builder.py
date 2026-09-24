from typing import Any


def build_modelfile(
    base_model: str,
    current_modelfile: str,
    parameters: dict[str, Any],
) -> str:
    """
    Genera un Modelfile derivado de un modelo existente.

    Conserva las directivas estructurales relevantes y los parámetros
    existentes que no hayan sido sustituidos explícitamente.
    """

    lines = current_modelfile.splitlines()

    structural: list[str] = []
    current_parameters: list[tuple[str, str]] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        parts = stripped.split(None, 2)

        if not parts:
            continue

        directive = parts[0].upper()

        if directive == "PARAMETER" and len(parts) >= 3:
            current_parameters.append(
                (parts[1], parts[2])
            )
            continue

        if directive in {
            "TEMPLATE",
            "SYSTEM",
            "ADAPTER",
            "MESSAGE",
            "REQUIRES",
        }:
            structural.append(stripped)

    result = [
        f"FROM {base_model}",
    ]

    result.extend(structural)

    # Conservamos parámetros actuales que no hayan sido propuestos.
    proposed_names = set(parameters)

    for name, value in current_parameters:
        if name in proposed_names:
            continue

        result.append(
            f"PARAMETER {name} {value}"
        )

    # Los parámetros propuestos sobrescriben los existentes.
    for name, value in parameters.items():
        if value is None:
            continue

        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)

        result.append(
            f"PARAMETER {name} {rendered}"
        )

    return "\n".join(result) + "\n"
