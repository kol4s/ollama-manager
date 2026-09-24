from unittest.mock import Mock

from core.application_executor import (
    ApplicationResult,
    apply_application_plan,
)
from core.application_plan import (
    build_application_plan,
)


def create_plan():
    return build_application_plan(
        "gemma4:e4b",
        "gemma4:e4b-optimized",
        {
            "temperature": 0.7,
        },
        (
            "FROM gemma4:e4b\n"
            "PARAMETER temperature 0.7\n"
        ),
    )


def test_apply_application_plan_calls_create_model():
    client = Mock()

    client.create_model.return_value = {
        "status": "success",
    }

    plan = create_plan()

    result = apply_application_plan(
        client,
        plan,
    )

    assert isinstance(
        result,
        ApplicationResult,
    )

    assert result.success is True
    assert result.source_model == "gemma4:e4b"
    assert result.target_model == (
        "gemma4:e4b-optimized"
    )

    client.create_model.assert_called_once_with(
        "gemma4:e4b-optimized",
        (
            "FROM gemma4:e4b\n"
            "PARAMETER temperature 0.7\n"
        ),
        progress_callback=None,
    )


def test_apply_application_plan_forwards_progress():
    client = Mock()
    client.create_model.return_value = {
        "status": "success",
    }

    progress = []

    plan = create_plan()

    apply_application_plan(
        client,
        plan,
        progress_callback=progress.append,
    )

    args = client.create_model.call_args

    callback = args.kwargs["progress_callback"]

    assert callback.__self__ is progress
    assert callable(callback)


def test_unsafe_plan_is_rejected():
    client = Mock()

    plan = create_plan()
    plan.overwrite_source = True

    try:
        apply_application_plan(
            client,
            plan,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )
