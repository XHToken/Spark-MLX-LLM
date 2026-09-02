import importlib
import sys
from types import ModuleType

MODEL_MODULE = "mlx_lm.models.spark2_5"


def register_model() -> ModuleType:
    """Expose Spark2_5 to MLX LM's native model resolver."""
    try:
        return importlib.import_module(MODEL_MODULE)
    except ModuleNotFoundError as error:
        if error.name != MODEL_MODULE:
            raise

    from . import model

    sys.modules[MODEL_MODULE] = model
    return model
