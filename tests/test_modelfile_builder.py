from core.modelfile_builder import build_modelfile


def test_build_modelfile_uses_model_name():
    modelfile = build_modelfile(
        "gemma4:e4b",
        """
# comment
FROM /some/internal/blob
TEMPLATE {{ .Prompt }}
PARAMETER temperature 1
LICENSE \"\"\"ignored\"\"\"
""",
        {
            "temperature": 0.7,
            "top_k": 40,
            "num_ctx": 65536,
        },
    )

    assert modelfile.startswith(
        "FROM gemma4:e4b\n"
    )

    assert (
        "FROM /some/internal/blob"
        not in modelfile
    )

    assert (
        "TEMPLATE {{ .Prompt }}"
        in modelfile
    )

    assert "PARAMETER temperature 0.7" in modelfile
    assert "PARAMETER top_k 40" in modelfile
    assert "PARAMETER num_ctx 65536" in modelfile
    assert "LICENSE" not in modelfile


def test_build_modelfile_handles_boolean():
    modelfile = build_modelfile(
        "test:model",
        "TEMPLATE {{ .Prompt }}",
        {
            "use_mmap": True,
        },
    )

    assert "PARAMETER use_mmap true" in modelfile


def test_build_modelfile_preserves_structure():
    modelfile = build_modelfile(
        "test:model",
        """
TEMPLATE {{ .Prompt }}
SYSTEM You are helpful.
ADAPTER adapter.gguf
""",
        {
            "temperature": 0.8,
        },
    )

    assert "TEMPLATE {{ .Prompt }}" in modelfile
    assert "SYSTEM You are helpful." in modelfile
    assert "ADAPTER adapter.gguf" in modelfile



def test_build_modelfile_preserves_unmanaged_parameters():
    modelfile = build_modelfile(
        "test:model",
        """
FROM test:model
PARAMETER temperature 1
PARAMETER top_k 64
PARAMETER custom_value abc
""",
        {
            "temperature": 0.7,
        },
    )

    assert "PARAMETER temperature 0.7" in modelfile
    assert "PARAMETER top_k 64" in modelfile
    assert "PARAMETER custom_value abc" in modelfile
    assert "PARAMETER temperature 1" not in modelfile



def test_build_modelfile_drops_internal_renderer_parser():
    modelfile = build_modelfile(
        "gemma4:e4b",
        """
FROM gemma4:e4b
TEMPLATE {{ .Prompt }}
RENDERER gemma4
PARSER gemma4
PARAMETER temperature 1
""",
        {
            "temperature": 0.7,
        },
    )

    assert "RENDERER" not in modelfile
    assert "PARSER" not in modelfile
    assert "TEMPLATE {{ .Prompt }}" in modelfile
    assert "PARAMETER temperature 0.7" in modelfile
