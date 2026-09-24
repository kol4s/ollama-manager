import json
from pathlib import Path
from typing import Any


class KnowledgeBase:
    """Carga y proporciona acceso a la base de conocimiento."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] = {}

    def load(self) -> None:
        """Carga la base de conocimiento desde JSON."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: {self.path}"
            )

        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid knowledge base JSON: {self.path}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Knowledge base root must be a JSON object"
            )

        self._data = data

    @property
    def data(self) -> dict[str, Any]:
        """Devuelve los datos cargados."""
        if not self._data:
            raise RuntimeError(
                "Knowledge base has not been loaded"
            )

        return self._data

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Obtiene un valor utilizando una ruta de claves.

        Ejemplo:
            kb.get("gpu", "compute_api_priority")
        """
        current: Any = self.data

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default

            current = current[key]

        return current
