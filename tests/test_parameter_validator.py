import pytest

from core.parameter_validator import (
    ParameterValidationError,
    validate_parameter_value,
    validate_parameters,
)


KNOWLEDGE = {
    "ollama_parameters": {
        "memory": {
            "num_ctx": {
                "minimum": 1,
            },
        },
        "generation": {
            "top_p": {
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "repeat_penalty": {
                "exclusive_minimum": 0.0,
            },
        },
    }
}


def test_validate_parameter_value_valid():
    validate_parameter_value(
        "num_ctx",
        65000,
        KNOWLEDGE,
    )


def test_validate_parameter_value_minimum():
    with pytest.raises(
        ParameterValidationError
    ):
        validate_parameter_value(
            "num_ctx",
            0,
            KNOWLEDGE,
        )


def test_validate_parameter_value_maximum():
    with pytest.raises(
        ParameterValidationError
    ):
        validate_parameter_value(
            "top_p",
            1.5,
            KNOWLEDGE,
        )


def test_validate_parameter_value_exclusive_minimum():
    with pytest.raises(
        ParameterValidationError
    ):
        validate_parameter_value(
            "repeat_penalty",
            0.0,
            KNOWLEDGE,
        )


def test_validate_parameters():
    validate_parameters(
        {
            "num_ctx": 65000,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
        KNOWLEDGE,
    )
