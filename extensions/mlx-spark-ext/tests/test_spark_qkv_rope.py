# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx_spark_ext as ext


class TestSparkQKVRoPE(unittest.TestCase):
    def test_official_core_is_unchanged(self):
        self.assertFalse(hasattr(mx.fast, "spark_qkv_rope"))
        self.assertEqual(ext.spark_qkv_rope.__module__, "mlx_spark_ext._ext")

    def test_compile_and_stream(self):
        mx.random.seed(20)
        x = mx.random.normal((1, 1, 256))
        weight, scales, biases = mx.quantize(
            mx.random.normal((256, 256)), group_size=128, bits=4
        )

        def operation(x, weight, scales, biases):
            return ext.spark_qkv_rope(
                x, weight, scales, biases, 128, 64, 4, 2, 32, 32, 10000.0, 37
            )

        expected = operation(x, weight, scales, biases)
        compiled = mx.compile(operation)(x, weight, scales, biases)
        explicit_stream = ext.spark_qkv_rope(
            x,
            weight,
            scales,
            biases,
            128,
            64,
            4,
            2,
            32,
            32,
            10000.0,
            37,
            stream=mx.default_stream(mx.default_device()),
        )
        mx.eval(expected, compiled, explicit_stream)
        for result in (compiled, explicit_stream):
            for actual, reference in zip(result, expected):
                self.assertTrue(mx.array_equal(actual, reference))

    def test_spark_qkv_rope(self):
        mx.random.seed(19)
        hidden_size, head_dim = 256, 32
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            for group_size, bits in ((32, 4), (64, 4), (128, 4), (64, 8)):
                for num_heads, num_kv_heads, rope_dims, offset in (
                    (4, 2, 16, 0),
                    (4, 4, 32, 37),
                    (2, 1, 32, 32768),
                ):
                    with self.subTest(
                        dtype=dtype,
                        group_size=group_size,
                        bits=bits,
                        heads=num_heads,
                        kv_heads=num_kv_heads,
                        rope_dims=rope_dims,
                        offset=offset,
                    ):
                        q_size = num_heads * head_dim
                        kv_size = num_kv_heads * head_dim
                        x = mx.random.normal((1, 1, hidden_size * 2)).astype(dtype)
                        x = x[..., ::2]
                        weight = mx.random.normal(
                            (q_size + 2 * kv_size, hidden_size)
                        ).astype(dtype)
                        weight, scales, biases = mx.quantize(
                            weight, group_size=group_size, bits=bits
                        )
                        actual = ext.spark_qkv_rope(
                            x,
                            weight,
                            scales,
                            biases,
                            q_size,
                            kv_size,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                            rope_dims,
                            10000.0,
                            offset,
                            group_size,
                            bits,
                        )
                        qkv = mx.quantized_matmul(
                            x,
                            weight,
                            scales,
                            biases,
                            transpose=True,
                            group_size=group_size,
                            bits=bits,
                        )
                        expected = [
                            mx.contiguous(
                                part.reshape(1, 1, heads, head_dim).transpose(
                                    0, 2, 1, 3
                                )
                            )
                            for part, heads in zip(
                                mx.split(qkv, (q_size, q_size + kv_size), axis=-1),
                                (num_heads, num_kv_heads, num_kv_heads),
                            )
                        ]
                        for i in (0, 1):
                            expected[i] = mx.fast.rope(
                                expected[i],
                                rope_dims,
                                traditional=False,
                                base=10000.0,
                                scale=1.0,
                                offset=offset,
                            )
                        mx.eval(actual, expected)
                        for result, reference in zip(actual, expected):
                            self.assertEqual(result.shape, reference.shape)
                            self.assertEqual(result.dtype, dtype)
                            self.assertTrue(mx.array_equal(result, reference))

    def test_spark_qkv_rope_invalid_inputs(self):
        args = {
            "x": mx.ones((1, 1, 256)),
            "weight": mx.zeros((256, 32), dtype=mx.uint32),
            "scales": mx.ones((256, 2)),
            "biases": mx.zeros((256, 2)),
            "q_size": 128,
            "kv_size": 64,
            "num_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "rope_dims": 32,
            "rope_base": 10000.0,
            "offset": 0,
            "group_size": 128,
            "bits": 4,
        }
        invalid = (
            {"x": mx.ones((256,))},
            {"x": mx.ones((2, 1, 256))},
            {"x": mx.ones((1, 2, 256))},
            {"x": mx.ones((1, 1, 0))},
            {"x": mx.ones((1, 1, 128))},
            {"x": mx.ones((1, 1, 256), dtype=mx.int32)},
            {"weight": mx.zeros((8192,), dtype=mx.uint32)},
            {"weight": mx.zeros((255, 32), dtype=mx.uint32)},
            {"weight": mx.zeros((256, 31), dtype=mx.uint32)},
            {"weight": mx.zeros((256, 32))},
            {"scales": mx.ones((256,))},
            {"scales": mx.ones((255, 2)), "biases": mx.zeros((255, 2))},
            {"scales": mx.ones((256, 1)), "biases": mx.zeros((256, 1))},
            {"scales": mx.ones((256, 2), dtype=mx.int32)},
            {"biases": mx.zeros((256,))},
            {"biases": mx.zeros((256, 1))},
            {"biases": mx.zeros((256, 2), dtype=mx.int32)},
            {"num_heads": 0},
            {"num_key_value_heads": 0},
            {"num_heads": 3, "q_size": 96},
            {"head_dim": 0},
            {"q_size": 127},
            {"kv_size": 63},
            {"rope_dims": 0},
            {"rope_dims": 31},
            {"rope_dims": 34},
            {"rope_base": 0.0},
            {"rope_base": -1.0},
            {"rope_base": float("nan")},
            {"rope_base": float("inf")},
            {"group_size": 0},
            {"group_size": 16},
            {"bits": 0},
            {"bits": 7},
        )
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(ValueError):
                ext.spark_qkv_rope(**(args | update))


if __name__ == "__main__":
    unittest.main()
