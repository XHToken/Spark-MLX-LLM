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


def wide_head_args(**overrides):
    """Geometry of Spark-X2.5-4B, where hidden_size is not n_heads * head_dim."""
    values = {
        "hidden_size": 20,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
    }
    values.update(overrides)
    return make_args(**values)


def test_attention_shapes_when_hidden_size_differs_from_head_geometry():
    args = wide_head_args()
    assert args.hidden_size != args.num_attention_heads * args.head_dim

    parameters = dict(tree_flatten(Model(args).parameters()))
    q_size = args.num_attention_heads * args.head_dim
    kv_size = args.num_key_value_heads * args.head_dim

    assert parameters["model.layers.0.self_attn.q_k_v_proj.weight"].shape == (
        q_size + 2 * kv_size,
        args.hidden_size,
    )
    assert parameters["model.layers.0.self_attn.out_proj.weight"].shape == (
        args.hidden_size,
        q_size,
    )


def test_forward_shape_when_hidden_size_differs_from_head_geometry():
    args = wide_head_args()
    logits = Model(args)(mx.array([[1, 2, 3]]))
    mx.eval(logits)

    assert logits.shape == (1, 3, args.vocab_size)


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
