import os
from pathlib import Path

import mlx.core as mx
import pytest

from spark_mlx_llm.loader import load

MODEL_PATH = os.environ.get("SPARK25_MODEL")
MODEL_DTYPE = os.environ.get("SPARK25_DTYPE")


@pytest.mark.integration
@pytest.mark.skipif(not MODEL_PATH, reason="SPARK25_MODEL is not set")
def test_real_checkpoint_loads_strictly():
    model_path = Path(MODEL_PATH)
    model, tokenizer, config = load(
        model_path,
        strict=True,
        lazy=True,
        dtype=MODEL_DTYPE,
        return_config=True,
    )

    assert config["model_type"] == "spark2_5"
    assert len(model.layers) == 28
    assert tokenizer.eos_token_id == 1

    logits = model(mx.array([[0]]))
    mx.eval(logits)
    assert logits.shape == (1, 1, 131072)
