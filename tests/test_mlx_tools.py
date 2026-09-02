import sys
from types import ModuleType, SimpleNamespace

import pytest

from spark_mlx_llm import mlx_tools
from spark_mlx_llm import model as spark_model
from spark_mlx_llm.registration import MODEL_MODULE, register_model


def test_register_model_exposes_spark2_5():
    previous = sys.modules.pop(MODEL_MODULE, None)
    try:
        registered = register_model()

        assert registered is spark_model
        assert sys.modules[MODEL_MODULE] is spark_model
    finally:
        if previous is None:
            sys.modules.pop(MODEL_MODULE, None)
        else:
            sys.modules[MODEL_MODULE] = previous


def test_register_model_prefers_native_mlx_lm_module(monkeypatch):
    native_module = ModuleType(MODEL_MODULE)
    monkeypatch.setitem(sys.modules, MODEL_MODULE, native_module)

    assert register_model() is native_module


@pytest.mark.parametrize(
    ("entry_point", "command"),
    [
        (mlx_tools.convert_main, "convert"),
        (mlx_tools.chat_main, "chat"),
        (mlx_tools.server_main, "server"),
    ],
)
def test_mlx_lm_entry_points_register_before_delegating(
    monkeypatch, entry_point, command
):
    events = []

    monkeypatch.setattr(
        mlx_tools,
        "register_model",
        lambda: events.append("register"),
    )
    monkeypatch.setattr(
        mlx_tools.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(
            main=lambda: events.append(module_name),
        ),
    )

    entry_point()

    assert events == ["register", f"mlx_lm.{command}"]
