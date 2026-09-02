import mlx.core as mx
from mlx.utils import tree_flatten

from spark_mlx_llm.model import Model, ModelArgs


def make_args(**overrides):
    values = {
        "model_type": "spark2_5",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 32,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.5,
                "rope_theta": 1_000_000,
            },
            "sliding_attention": {
                "partial_rotary_factor": 1.0,
                "rope_theta": 10_000,
            },
        },
        "sliding_window": 8,
    }
    values.update(overrides)
    return ModelArgs(**values)


def test_config_matches_spark2_5_constraints():
    args = make_args()
    assert args.layer_types == ["sliding_attention", "full_attention"]
    assert args.tie_word_embeddings is True


def test_parameter_names_and_shapes_match_checkpoint_layout():
    model = Model(make_args())
    parameters = dict(tree_flatten(model.parameters()))

    assert parameters["model.embedding.weight"].shape == (32, 16)
    assert parameters["model.layers.0.self_attn.q_k_v_proj.weight"].shape == (
        32,
        16,
    )
    assert parameters["model.layers.0.self_attn.g_proj.weight"].shape == (4, 16)
    assert parameters["model.layers.0.self_attn.out_proj.weight"].shape == (16, 16)
    assert "lm_head.weight" not in parameters


def test_layer_specific_rope_configuration():
    model = Model(make_args())
    sliding_rope = model.layers[0].self_attn.rope
    full_rope = model.layers[1].self_attn.rope

    assert sliding_rope.dims == 4
    assert sliding_rope.base == 10_000
    assert full_rope.dims == 2
    assert full_rope.base == 1_000_000


def test_forward_shape():
    model = Model(make_args())
    tokens = mx.array([[1, 2, 3]])
    logits = model(tokens)
    mx.eval(logits)

    assert logits.shape == (1, 3, 32)


def test_cached_decode_matches_full_forward():
    model = Model(make_args())
    tokens = mx.array([[1, 2, 3]])

    full_logits = model(tokens)
    caches = model.make_cache()
    cached_logits = mx.concatenate(
        [model(tokens[:, idx : idx + 1], cache=caches) for idx in range(3)],
        axis=1,
    )
    mx.eval(full_logits, cached_logits)

    # CUDA attention kernels can differ slightly between batched prefill and
    # token-by-token decode because their reduction orders are different.
    assert mx.allclose(full_logits, cached_logits, rtol=2e-3, atol=2e-3)


def test_attention_gates_are_excluded_from_quantization():
    model = Model(make_args())
    predicate = model.quant_predicate

    assert not predicate(
        "model.layers.0.self_attn.g_proj",
        model.layers[0].self_attn.g_proj,
    )
    assert predicate(
        "model.layers.0.self_attn.out_proj",
        model.layers[0].self_attn.out_proj,
    )
