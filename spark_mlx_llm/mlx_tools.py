import importlib

from .registration import register_model


def _run(command: str):
    register_model()
    return importlib.import_module(f"mlx_lm.{command}").main()


def convert_main():
    return _run("convert")


def chat_main():
    return _run("chat")


def server_main():
    return _run("server")
