from unittest.mock import Mock, patch

from core.knowledge_base import KnowledgeBase
from core.ollama_client import OllamaClient


def test_show_model_keeps_model_information():
    client = OllamaClient()

    payload = {
        "details": {
            "family": "qwen35",
            "parameter_size": "27B",
            "quantization_level": "Q4_K_M",
        },
        "model_info": {
            "general.architecture": "qwen35",
            "general.parameter_count": 27000000000,
            "qwen35.context_length": 262144,
            "qwen35.embedding_length": 5120,
            "qwen35.block_count": 64,
            "qwen35.attention.head_count_kv": 4,
        },
    }

    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None

    with patch(
        "core.ollama_client.requests.post",
        return_value=response,
    ):
        model = client.show_model(
            "qwen3.8:27b"
        )

    assert model.name == "qwen3.8:27b"
    assert model.architecture == "qwen35"
    assert model.parameter_count == 27000000000
    assert model.context_length == 262144
    assert model.embedding_length == 5120
