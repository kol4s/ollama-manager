from core.application_plan import (
    build_application_plan,
)


def test_application_plan_is_safe_by_default():
    plan = build_application_plan(
        "gemma4:e4b",
        "gemma4:e4b-optimized",
        {
            "temperature": 0.7,
            "num_ctx": 65000,
        },
        (
            "FROM gemma4:e4b\n"
            "PARAMETER temperature 0.7\n"
        ),
    )

    assert plan.source_model == "gemma4:e4b"
    assert plan.target_model == "gemma4:e4b-optimized"
    assert plan.requires_confirmation is True
    assert plan.overwrite_source is False
    assert plan.is_safe_default is True


def test_application_plan_copies_parameters():
    params = {
        "temperature": 0.7,
        "top_k": 40,
    }

    plan = build_application_plan(
        "test:model",
        "test:model-optimized",
        params,
        "FROM test:model\n",
    )

    assert plan.parameters == params
    assert plan.parameters is not params


def test_same_model_is_rejected():
    try:
        build_application_plan(
            "test:model",
            "test:model",
            {},
            "FROM test:model\n",
        )
    except ValueError as exc:
        assert "differ" in str(exc)
        return

    raise AssertionError(
        "Expected ValueError"
    )


def test_empty_source_is_rejected():
    try:
        build_application_plan(
            "",
            "test:model-optimized",
            {},
            "FROM test:model\n",
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )
