from dataclasses import dataclass, field
from typing import Any

import json
import requests

from core.model_info import ModelInfo

class OllamaError(RuntimeError):
    """Error base para operaciones con Ollama."""


class OllamaConnectionError(OllamaError):
    """No se pudo conectar con el servicio Ollama."""


class OllamaTimeoutError(OllamaError):
    """La operación con Ollama superó el tiempo de espera."""


class OllamaHTTPError(OllamaError):
    """Ollama respondió con un error HTTP."""



@dataclass
class OllamaModel:
    """Representa un modelo disponible en Ollama."""

    name: str
    size_bytes: int
    details: dict[str, Any]
    current_parameters: dict[str, Any] = field(
        default_factory=dict
    )
    modelfile: str = ""

    architecture: str = ""
    parameter_count: int = 0
    context_length: int = 0
    embedding_length: int = 0
    block_count: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    key_length: int = 0
    value_length: int = 0
    key_length_swa: int = 0
    value_length_swa: int = 0
    sliding_window: int = 0
    shared_kv_layers: int = 0


@dataclass
class OllamaRunningModel:
    """Representa un modelo actualmente cargado en Ollama."""

    name: str
    size_bytes: int
    size_vram_bytes: int
    context_length: int
    digest: str
    details: dict[str, Any]
    expires_at: str

    @property
    def vram_percentage(self) -> float:
        """Porcentaje del modelo actualmente residente en VRAM."""
        if self.size_bytes <= 0:
            return 0.0

        return (
            self.size_vram_bytes / self.size_bytes
        ) * 100.0

    @property
    def cpu_memory_bytes(self) -> int:
        """Memoria del modelo que no está actualmente en VRAM."""
        return max(
            0,
            self.size_bytes - self.size_vram_bytes,
        )


class OllamaClient:
    """Cliente para comunicarse con la API local de Ollama."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
    ):
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _raise_request_error(exc: Exception) -> None:
        if isinstance(exc, requests.Timeout):
            raise OllamaTimeoutError(
                "Ollama no respondió dentro del tiempo esperado."
            ) from exc

        if isinstance(exc, requests.ConnectionError):
            raise OllamaConnectionError(
                "No se pudo conectar con el servicio Ollama."
            ) from exc

        if isinstance(exc, requests.HTTPError):
            raise OllamaHTTPError(
                f"Ollama respondió con un error HTTP: {exc}"
            ) from exc

        raise exc

    def list_models(self) -> list[OllamaModel]:
        """Devuelve los modelos disponibles en Ollama."""

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:
            self._raise_request_error(exc)

        data = response.json()
        models: list[OllamaModel] = []

        for item in data.get("models", []):
            models.append(
                OllamaModel(
                    name=item.get("name", ""),
                    size_bytes=int(
                        item.get("size", 0) or 0
                    ),
                    details=item.get("details", {}),
                )
            )

        return models

    def model_exists(self, name: str) -> bool:
        """Indica si un modelo ya existe en Ollama."""

        target = name.strip()

        if not target:
            return False

        return any(
            model.name == target
            for model in self.list_models()
        )

    @staticmethod
    def _parse_parameters(text: str) -> dict[str, Any]:
        """
        Convierte el bloque 'parameters' de /api/show
        en un diccionario de parámetros.
        """

        result: dict[str, Any] = {}

        for line in str(text or "").splitlines():
            line = line.strip()

            if not line:
                continue

            parts = line.split(None, 1)

            if len(parts) != 2:
                continue

            key, raw_value = parts
            value = raw_value.strip()

            if value.lower() == "true":
                parsed: Any = True
            elif value.lower() == "false":
                parsed = False
            else:
                try:
                    if (
                        "." in value
                        or "e" in value.lower()
                    ):
                        parsed = float(value)
                    else:
                        parsed = int(value)
                except ValueError:
                    parsed = value

            result[key] = parsed

        return result

    def show_model(self, name: str) -> OllamaModel:
        """Obtiene información detallada de un modelo."""

        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": name},
                timeout=10,
            )
            response.raise_for_status()
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:
            self._raise_request_error(exc)

        data = response.json()
        model_info = data.get(
            "model_info",
            {},
        )

        details = data.get(
            "details",
            {},
        )

        current_parameters = self._parse_parameters(
            data.get("parameters", "")
        )

        architecture = str(
            model_info.get(
                "general.architecture",
                "",
            )
        )

        def info_int(key: str) -> int:
            value = model_info.get(
                key,
                0,
            )

            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        return OllamaModel(
            name=name,
            size_bytes=int(
                data.get("size", 0) or 0
            ),
            details=details,
            current_parameters=current_parameters,
            modelfile=str(
                data.get("modelfile", "")
            ),
            architecture=architecture,
            parameter_count=info_int(
                "general.parameter_count"
            ),
            context_length=info_int(
                f"{architecture}.context_length"
            ),
            embedding_length=info_int(
                f"{architecture}.embedding_length"
            ),
            block_count=info_int(
                f"{architecture}.block_count"
            ),
            head_count=info_int(
                f"{architecture}.attention.head_count"
            ),
            head_count_kv=info_int(
                f"{architecture}.attention.head_count_kv"
            ),
            key_length=info_int(
                f"{architecture}.attention.key_length"
            ),
            value_length=info_int(
                f"{architecture}.attention.value_length"
            ),
            key_length_swa=info_int(
                f"{architecture}.attention.key_length_swa"
            ),
            value_length_swa=info_int(
                f"{architecture}.attention.value_length_swa"
            ),
            sliding_window=info_int(
                f"{architecture}.attention.sliding_window"
            ),
            shared_kv_layers=info_int(
                f"{architecture}.attention.shared_kv_layers"
            ),
        )

    def list_running_models(
        self,
    ) -> list[OllamaRunningModel]:
        """Devuelve los modelos actualmente cargados."""

        try:
            response = requests.get(
                f"{self.base_url}/api/ps",
                timeout=10,
            )
            response.raise_for_status()
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:
            self._raise_request_error(exc)

        data = response.json()
        models: list[OllamaRunningModel] = []

        for item in data.get("models", []):
            models.append(
                OllamaRunningModel(
                    name=item.get("name", ""),
                    size_bytes=int(
                        item.get("size", 0) or 0
                    ),
                    size_vram_bytes=int(
                        item.get("size_vram", 0) or 0
                    ),
                    context_length=int(
                        item.get("context_length", 0) or 0
                    ),
                    digest=str(
                        item.get("digest", "")
                    ),
                    details=item.get(
                        "details",
                        {},
                    ),
                    expires_at=str(
                        item.get("expires_at", "")
                    ),
                )
            )

        return models

    def create_model(
        self,
        name: str,
        modelfile: str,
        *,
        progress_callback=None,
    ) -> dict[str, Any]:
        """
        Crea una nueva variante usando la API /api/create.

        Ollama 0.33.x utiliza `from` como modelo base y `parameters`
        para los parámetros del modelo. El Modelfile recibido se
        interpreta aquí para construir ese payload.
        """

        source_model = None
        parameters: dict[str, Any] = {}

        for raw_line in modelfile.splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split(None, 2)

            if not parts:
                continue

            directive = parts[0].upper()

            if directive == "FROM" and len(parts) >= 2:
                source_model = parts[1]

            elif (
                directive == "PARAMETER"
                and len(parts) >= 3
            ):
                parameter_name = parts[1]
                raw_value = parts[2].strip()

                lowered = raw_value.lower()

                if lowered == "true":
                    value: Any = True
                elif lowered == "false":
                    value = False
                else:
                    try:
                        if (
                            "." in raw_value
                            or "e" in lowered
                        ):
                            value = float(raw_value)
                        else:
                            value = int(raw_value)
                    except ValueError:
                        value = raw_value

                parameters[parameter_name] = value

        if not source_model:
            raise ValueError(
                "Modelfile does not contain a FROM instruction"
            )

        payload = {
            "model": name,
            "from": source_model,
            "parameters": parameters,
            "stream": True,
        }

        try:
            with requests.post(
                f"{self.base_url}/api/create",
                json=payload,
                stream=True,
                timeout=600,
            ) as response:
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    body = response.text.strip()

                    if body:
                        raise requests.HTTPError(
                            f"{exc} - Ollama: {body}",
                            response=response,
                        ) from exc

                    raise

                last_status: dict[str, Any] = {}

                for line in response.iter_lines(
                    decode_unicode=True
                ):
                    if not line:
                        continue

                    data = json.loads(line)
                    last_status = data

                    if progress_callback is not None:
                        progress_callback(data)

                return last_status

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:
            self._raise_request_error(exc)

    def delete_model(self, name: str) -> None:
        """Elimina un modelo instalado."""

        try:
            response = requests.delete(
                f"{self.base_url}/api/delete",
                json={"name": name},
                timeout=30,
            )
            response.raise_for_status()
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:
            self._raise_request_error(exc)

    def pull_model(
        self,
        name: str,
        *,
        insecure: bool = False,
        progress_callback=None,
    ) -> dict[str, Any]:
        """
        Descarga un modelo desde el registro de Ollama.

        La API devuelve objetos JSON en streaming. Si se proporciona
        progress_callback, se invoca con cada objeto recibido.
        """

        payload = {
            "model": name,
            "stream": True,
        }

        if insecure:
            payload["insecure"] = True

        try:
            with requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                stream=True,
                timeout=60,
            ) as response:
                response.raise_for_status()

                last_status: dict[str, Any] = {}

                for line in response.iter_lines(
                    decode_unicode=True
                ):
                    if not line:
                        continue

                    data = json.loads(line)
                    last_status = data

                    if progress_callback is not None:
                        progress_callback(data)

                return last_status

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:
            self._raise_request_error(exc)

    def stop_model(self, name: str) -> dict[str, Any]:
        """
        Descarga un modelo actualmente cargado de memoria.

        Ollama documenta keep_alive=0 con una petición de generación
        vacía como mecanismo de unload.
        """

        return self.generate(
            name,
            "",
            keep_alive=0,
            stream=False,
        )

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        context_length: int | None = None,
        keep_alive: str | int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Ejecuta una generación mediante la API de Ollama.

        Esta función no modifica automáticamente ninguna
        configuración persistente del modelo.
        """

        options: dict[str, Any] = {}

        if context_length is not None:
            options["num_ctx"] = int(
                context_length
            )

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        if options:
            payload["options"] = options

        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:
            self._raise_request_error(exc)

        return response.json()

    def to_model_info(
        self,
        model: OllamaModel,
    ) -> ModelInfo:
        """Convierte un modelo de Ollama al modelo interno."""

        details = model.details

        parameter_count = int(
            model.parameter_count or 0
        )

        parameter_size = details.get(
            "parameter_size",
            "0B",
        )

        parameter_text = (
            str(parameter_size)
            .upper()
            .strip()
        )

        if parameter_text.endswith("B"):
            parameters_b = float(
                parameter_text[:-1]
            )

        elif parameter_text.endswith("M"):
            parameters_b = (
                float(parameter_text[:-1])
                / 1000.0
            )

        elif parameter_count:
            parameters_b = (
                parameter_count
                / 1_000_000_000
            )

        else:
            parameters_b = 0.0

        quantization = details.get(
            "quantization_level",
            "unknown",
        )

        return ModelInfo(
            name=model.name,
            parameters_b=parameters_b,
            quantization=str(
                quantization
            ),
            context_length=(
                model.context_length
                or 8192
            ),
            size_bytes=model.size_bytes,
            architecture=model.architecture,
            parameter_count=parameter_count,
            embedding_length=model.embedding_length,
            block_count=model.block_count,
            head_count=model.head_count,
            head_count_kv=model.head_count_kv,
            key_length=model.key_length,
            value_length=model.value_length,
            key_length_swa=model.key_length_swa,
            value_length_swa=model.value_length_swa,
            sliding_window=model.sliding_window,
            sliding_window_pattern="",
            shared_kv_layers=model.shared_kv_layers,
        )
