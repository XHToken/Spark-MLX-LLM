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
    assert tokenizer.has_tool_calling
    assert tokenizer.tool_call_start == "<tool_call>"
    assert tokenizer.tool_call_end == "</tool_call>"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_state",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "active": {"type": "boolean"},
                        "items": {"type": "array", "items": {"type": "string"}},
                        "metadata": {"type": "object"},
                    },
                },
            },
        }
    ]
    tool_call = tokenizer.tool_parser(
        "set_state"
        "<arg_key>name</arg_key><arg_value>上海</arg_value>"
        "<arg_key>count</arg_key><arg_value>42</arg_value>"
        "<arg_key>ratio</arg_key><arg_value>2.5</arg_value>"
        "<arg_key>active</arg_key><arg_value>true</arg_value>"
        '<arg_key>items</arg_key><arg_value>["alpha", "测试"]</arg_value>'
        '<arg_key>metadata</arg_key><arg_value>{"source": "spark", '
        '"options": {"strict": true}}</arg_value>',
        tools,
    )
    assert tool_call == {
        "name": "set_state",
        "arguments": {
            "name": "上海",
            "count": 42,
            "ratio": 2.5,
            "active": True,
            "items": ["alpha", "测试"],
            "metadata": {"source": "spark", "options": {"strict": True}},
        },
    }

    logits = model(mx.array([[0]]))
    mx.eval(logits)
    assert logits.shape == (1, 1, 131072)
