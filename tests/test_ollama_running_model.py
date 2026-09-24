from core.ollama_client import OllamaRunningModel


def test_running_model_memory_properties():
    model = OllamaRunningModel(
        name="test",
        size_bytes=1000,
        size_vram_bytes=400,
        context_length=65000,
        digest="test",
        details={},
        expires_at="",
    )

    assert model.vram_percentage == 40.0
    assert model.cpu_memory_bytes == 600


def test_running_model_zero_size():
    model = OllamaRunningModel(
        name="test",
        size_bytes=0,
        size_vram_bytes=0,
        context_length=4096,
        digest="test",
        details={},
        expires_at="",
    )

    assert model.vram_percentage == 0.0
    assert model.cpu_memory_bytes == 0


print("RUNNING MODEL TEST CREATED")
