from unittest.mock import Mock, patch

from core.ollama_client import OllamaClient


def test_generate_sends_context():
    client = OllamaClient()

    response = Mock()
    response.json.return_value = {
        "response": "OK"
    }
    response.raise_for_status.return_value = None

    with patch(
        "core.ollama_client.requests.post",
        return_value=response,
    ) as post:
        result = client.generate(
            "test:model",
            "hola",
            context_length=65000,
            keep_alive="10m",
        )

    assert result["response"] == "OK"

    _, kwargs = post.call_args

    assert kwargs["json"]["model"] == "test:model"
    assert kwargs["json"]["prompt"] == "hola"
    assert kwargs["json"]["stream"] is False
    assert kwargs["json"]["options"]["num_ctx"] == 65000
    assert kwargs["json"]["keep_alive"] == "10m"


def test_delete_model():
    client = OllamaClient()

    response = Mock()
    response.raise_for_status.return_value = None

    with patch(
        "core.ollama_client.requests.delete",
        return_value=response,
    ) as delete:
        client.delete_model("test:model")

    delete.assert_called_once()

    _, kwargs = delete.call_args

    assert kwargs["json"]["name"] == "test:model"


def test_pull_model_streaming():
    client = OllamaClient()

    response = Mock()
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = [
        '{"status":"pulling manifest"}',
        '{"status":"downloading","total":1000,"completed":250}',
        '{"status":"success"}',
    ]
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)

    progress = []

    with patch(
        "core.ollama_client.requests.post",
        return_value=response,
    ) as post:
        result = client.pull_model(
            "test:model",
            progress_callback=progress.append,
        )

    assert result["status"] == "success"
    assert len(progress) == 3
    assert progress[1]["completed"] == 250

    _, kwargs = post.call_args

    assert kwargs["json"]["model"] == "test:model"
    assert kwargs["json"]["stream"] is True
    assert kwargs["stream"] is True


def test_stop_model_unloads_with_keep_alive_zero():
    client = OllamaClient()

    with patch.object(
        client,
        "generate",
        return_value={
            "done": True,
            "done_reason": "unload",
        },
    ) as generate:
        result = client.stop_model("test:model")

    assert result["done_reason"] == "unload"

    generate.assert_called_once_with(
        "test:model",
        "",
        keep_alive=0,
        stream=False,
    )


def test_parse_parameters():
    client = OllamaClient()

    result = client._parse_parameters(
        """temperature 1
top_k 64
top_p 0.95
use_mmap true
num_ctx 65000
name example"""
    )

    assert result["temperature"] == 1
    assert result["top_k"] == 64
    assert result["top_p"] == 0.95
    assert result["use_mmap"] is True
    assert result["num_ctx"] == 65000
    assert result["name"] == "example"


def test_show_model_preserves_modelfile():
    client = OllamaClient()

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "size": 100,
        "parameters": "temperature 1",
        "modelfile": "FROM test:model\nPARAMETER temperature 1\n",
        "details": {},
        "model_info": {},
    }

    with patch(
        "core.ollama_client.requests.post",
        return_value=response,
    ):
        model = client.show_model("test:model")

    assert model.modelfile == (
        "FROM test:model\nPARAMETER temperature 1\n"
    )


def test_create_model_streaming():
    client = OllamaClient()

    response = Mock()
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = [
        '{"status":"creating model layer"}',
        '{"status":"success"}',
    ]
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.text = ""

    with patch(
        "core.ollama_client.requests.post",
        return_value=response,
    ) as post:
        result = client.create_model(
            "gemma4:e4b-optimized-test",
            (
                "FROM gemma4:e4b\n"
                "PARAMETER temperature 0.7\n"
                "PARAMETER top_k 40\n"
                "PARAMETER num_ctx 34534\n"
            ),
        )

    assert result["status"] == "success"

    _, kwargs = post.call_args

    assert kwargs["json"]["model"] == (
        "gemma4:e4b-optimized-test"
    )
    assert kwargs["json"]["from"] == "gemma4:e4b"
    assert kwargs["json"]["parameters"] == {
        "temperature": 0.7,
        "top_k": 40,
        "num_ctx": 34534,
    }
    assert kwargs["json"]["stream"] is True
    assert kwargs["stream"] is True


def test_create_model_requires_from():
    client = OllamaClient()

    try:
        client.create_model(
            "test:model",
            "PARAMETER temperature 0.7\n",
        )
    except ValueError as exc:
        assert "FROM" in str(exc)
        return

    raise AssertionError(
        "Expected ValueError"
    )

def test_model_exists_true():
    from core.ollama_client import OllamaClient, OllamaModel

    client = OllamaClient()

    client.list_models = lambda: [
        OllamaModel(
            name="gemma4:e4b",
            size_bytes=0,
            details={},
        ),
        OllamaModel(
            name="gemma4:e4b-optimized",
            size_bytes=0,
            details={},
        ),
    ]

    assert client.model_exists(
        "gemma4:e4b-optimized"
    ) is True


def test_model_exists_false():
    from core.ollama_client import OllamaClient, OllamaModel

    client = OllamaClient()

    client.list_models = lambda: [
        OllamaModel(
            name="gemma4:e4b",
            size_bytes=0,
            details={},
        ),
    ]

    assert client.model_exists(
        "gemma4:e4b-optimized"
    ) is False


def test_model_exists_empty_name():
    from core.ollama_client import OllamaClient

    client = OllamaClient()

    client.list_models = lambda: []

    assert client.model_exists("   ") is False


def test_list_models_connection_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaConnectionError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.get",
        side_effect=requests.ConnectionError("offline"),
    ):
        try:
            client.list_models()
        except OllamaConnectionError as exc:
            assert "conectar" in str(exc).lower()
            return

    raise AssertionError(
        "Expected OllamaConnectionError"
    )


def test_list_models_timeout_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaTimeoutError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.get",
        side_effect=requests.Timeout("timeout"),
    ):
        try:
            client.list_models()
        except OllamaTimeoutError as exc:
            assert "tiempo" in str(exc).lower()
            return

    raise AssertionError(
        "Expected OllamaTimeoutError"
    )


def test_list_models_http_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaHTTPError,
    )

    client = OllamaClient()

    response = Mock()
    response.raise_for_status.side_effect = (
        requests.HTTPError("500 Server Error")
    )

    with patch(
        "core.ollama_client.requests.get",
        return_value=response,
    ):
        try:
            client.list_models()
        except OllamaHTTPError as exc:
            assert "http" in str(exc).lower()
            return

    raise AssertionError(
        "Expected OllamaHTTPError"
    )


def test_show_model_connection_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaConnectionError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.post",
        side_effect=requests.ConnectionError("offline"),
    ):
        try:
            client.show_model("test:model")
        except OllamaConnectionError:
            return

    raise AssertionError(
        "Expected OllamaConnectionError"
    )


def test_show_model_timeout_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaTimeoutError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.post",
        side_effect=requests.Timeout("timeout"),
    ):
        try:
            client.show_model("test:model")
        except OllamaTimeoutError:
            return

    raise AssertionError(
        "Expected OllamaTimeoutError"
    )


def test_show_model_http_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaHTTPError,
    )

    client = OllamaClient()

    response = Mock()
    response.raise_for_status.side_effect = (
        requests.HTTPError("404 Not Found")
    )

    with patch(
        "core.ollama_client.requests.post",
        return_value=response,
    ):
        try:
            client.show_model("missing:model")
        except OllamaHTTPError:
            return

    raise AssertionError(
        "Expected OllamaHTTPError"
    )


def test_delete_model_connection_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaConnectionError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.delete",
        side_effect=requests.ConnectionError("offline"),
    ):
        try:
            client.delete_model("test:model")
        except OllamaConnectionError:
            return

    raise AssertionError(
        "Expected OllamaConnectionError"
    )


def test_delete_model_timeout_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaTimeoutError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.delete",
        side_effect=requests.Timeout("timeout"),
    ):
        try:
            client.delete_model("test:model")
        except OllamaTimeoutError:
            return

    raise AssertionError(
        "Expected OllamaTimeoutError"
    )


def test_delete_model_http_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaHTTPError,
    )

    client = OllamaClient()

    response = Mock()
    response.raise_for_status.side_effect = (
        requests.HTTPError("404 Not Found")
    )

    with patch(
        "core.ollama_client.requests.delete",
        return_value=response,
    ):
        try:
            client.delete_model("missing:model")
        except OllamaHTTPError:
            return

    raise AssertionError(
        "Expected OllamaHTTPError"
    )


def test_pull_model_connection_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaConnectionError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.post",
        side_effect=requests.ConnectionError("offline"),
    ):
        try:
            client.pull_model("test:model")
        except OllamaConnectionError:
            return

    raise AssertionError(
        "Expected OllamaConnectionError"
    )


def test_pull_model_timeout_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaTimeoutError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.post",
        side_effect=requests.Timeout("timeout"),
    ):
        try:
            client.pull_model("test:model")
        except OllamaTimeoutError:
            return

    raise AssertionError(
        "Expected OllamaTimeoutError"
    )


def test_pull_model_http_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaHTTPError,
    )

    client = OllamaClient()

    response = Mock()
    response.raise_for_status.side_effect = (
        requests.HTTPError("500 Server Error")
    )
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)

    with patch(
        "core.ollama_client.requests.post",
        return_value=response,
    ):
        try:
            client.pull_model("test:model")
        except OllamaHTTPError:
            return

    raise AssertionError(
        "Expected OllamaHTTPError"
    )


def test_create_model_connection_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaConnectionError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.post",
        side_effect=requests.ConnectionError("offline"),
    ):
        try:
            client.create_model(
                "test:optimized",
                "FROM test:model\n",
            )
        except OllamaConnectionError:
            return

    raise AssertionError(
        "Expected OllamaConnectionError"
    )


def test_create_model_timeout_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaTimeoutError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.post",
        side_effect=requests.Timeout("timeout"),
    ):
        try:
            client.create_model(
                "test:optimized",
                "FROM test:model\n",
            )
        except OllamaTimeoutError:
            return

    raise AssertionError(
        "Expected OllamaTimeoutError"
    )


def test_create_model_http_error_preserves_ollama_body():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaHTTPError,
    )

    client = OllamaClient()

    response = Mock()
    response.raise_for_status.side_effect = (
        requests.HTTPError("400 Bad Request")
    )
    response.text = '{"error":"invalid model configuration"}'
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)

    with patch(
        "core.ollama_client.requests.post",
        return_value=response,
    ):
        try:
            client.create_model(
                "test:optimized",
                "FROM test:model\n",
            )
        except OllamaHTTPError as exc:
            message = str(exc)

            assert "http" in message.lower()
            assert "invalid model configuration" in message
            return

    raise AssertionError(
        "Expected OllamaHTTPError"
    )


def test_generate_connection_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaConnectionError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.post",
        side_effect=requests.ConnectionError("offline"),
    ):
        try:
            client.generate(
                "test:model",
                "hola",
            )
        except OllamaConnectionError:
            return

    raise AssertionError(
        "Expected OllamaConnectionError"
    )


def test_generate_timeout_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaTimeoutError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.post",
        side_effect=requests.Timeout("timeout"),
    ):
        try:
            client.generate(
                "test:model",
                "hola",
            )
        except OllamaTimeoutError:
            return

    raise AssertionError(
        "Expected OllamaTimeoutError"
    )


def test_generate_http_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaHTTPError,
    )

    client = OllamaClient()

    response = Mock()
    response.raise_for_status.side_effect = (
        requests.HTTPError("500 Server Error")
    )

    with patch(
        "core.ollama_client.requests.post",
        return_value=response,
    ):
        try:
            client.generate(
                "test:model",
                "hola",
            )
        except OllamaHTTPError:
            return

    raise AssertionError(
        "Expected OllamaHTTPError"
    )


def test_list_running_models_connection_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaConnectionError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.get",
        side_effect=requests.ConnectionError("offline"),
    ):
        try:
            client.list_running_models()
        except OllamaConnectionError:
            return

    raise AssertionError(
        "Expected OllamaConnectionError"
    )


def test_list_running_models_timeout_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaTimeoutError,
    )

    client = OllamaClient()

    with patch(
        "core.ollama_client.requests.get",
        side_effect=requests.Timeout("timeout"),
    ):
        try:
            client.list_running_models()
        except OllamaTimeoutError:
            return

    raise AssertionError(
        "Expected OllamaTimeoutError"
    )


def test_list_running_models_http_error():
    import requests

    from core.ollama_client import (
        OllamaClient,
        OllamaHTTPError,
    )

    client = OllamaClient()

    response = Mock()
    response.raise_for_status.side_effect = (
        requests.HTTPError("500 Server Error")
    )

    with patch(
        "core.ollama_client.requests.get",
        return_value=response,
    ):
        try:
            client.list_running_models()
        except OllamaHTTPError:
            return

    raise AssertionError(
        "Expected OllamaHTTPError"
    )
