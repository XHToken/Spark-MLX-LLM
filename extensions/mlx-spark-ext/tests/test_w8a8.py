"""Check TensorOps arithmetic against independent CPU integer references."""

import unittest

import mlx.core as mx
import mlx_spark_ext as ext
import numpy as np


def reference_linear(x, weight, scales, bias=None):
    x = np.array(x.astype(mx.float32))
    shape = x.shape[:-1] + (weight.shape[0],)
    x = x.reshape(-1, x.shape[-1])
    amax = np.max(np.abs(x), axis=1, keepdims=True)
    activation_scales = np.where(amax > 0, amax / np.float32(127), np.float32(1))
    quantized = np.clip(np.rint(x / activation_scales), -127, 127).astype(np.int32)
    accumulated = quantized @ np.asarray(weight, dtype=np.int32).T
    output = accumulated.astype(np.float32) * activation_scales
    output *= np.asarray(scales, dtype=np.float32).reshape(1, -1)
    if bias is not None:
        output += np.asarray(bias, dtype=np.float32)
    return output.reshape(shape)


@unittest.skipUnless(ext.is_w8a8_available(), "requires Apple M5+ and macOS 26.2+")
class TestW8A8(unittest.TestCase):
    def test_raw_int32_matches_cpu(self):
        rng = np.random.default_rng(42)
        # Partial M/N tiles, both kernel variants, and K's 512-element tail.
        for m, n, k in (
            (1, 1, 32),
            (2, 17, 64),
            (31, 127, 512),
            (32, 128, 544),
            (33, 129, 256),
            (64, 19, 1024),
            (65, 257, 544),
            (129, 131, 32),
            (513, 129, 64),
            (2, 2048, 2048),
            (2, 2560, 4096),
        ):
            with self.subTest(m=m, n=n, k=k):
                x = rng.integers(-128, 128, (m, k), dtype=np.int8)
                w = rng.integers(-128, 128, (n, k), dtype=np.int8)
                actual = ext.int8_matmul_int32(mx.array(x), mx.array(w))
                expected = x.astype(np.int32) @ w.astype(np.int32).T
                self.assertEqual(actual.dtype, mx.int32)
                np.testing.assert_array_equal(np.array(actual), expected)

    def test_linear_dtypes_shapes_and_bias(self):
        rng = np.random.default_rng(43)
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            for shape, n in (
                ((1, 32), 1),
                ((3, 544), 129),
                ((2, 3, 64), 17),
                ((65, 512), 133),
            ):
                with self.subTest(dtype=dtype, shape=shape, n=n):
                    x = mx.array(rng.normal(size=shape).astype(np.float32)).astype(
                        dtype
                    )
                    w = rng.integers(-128, 128, (n, shape[-1]), dtype=np.int8)
                    scales = rng.uniform(0.001, 0.01, n).astype(np.float32)
                    bias = rng.normal(size=n).astype(np.float32)
                    actual = ext.w8a8_linear(
                        x, mx.array(w), mx.array(scales[:, None]), mx.array(bias)
                    )
                    expected = reference_linear(x, w, scales, bias)
                    expected = np.array(
                        mx.array(expected).astype(dtype).astype(mx.float32)
                    )
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.shape, shape[:-1] + (n,))
                    rtol, atol = (2e-5, 2e-5) if dtype == mx.float32 else (0.008, 0.008)
                    np.testing.assert_allclose(
                        np.array(actual.astype(mx.float32)),
                        expected,
                        rtol=rtol,
                        atol=atol,
                    )

    def test_zero_rows_and_round_to_even_for_single_token(self):
        x = np.zeros((1, 32), dtype=np.float32)
        x[0, :7] = [127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]
        w = np.eye(32, dtype=np.int8)
        scales = mx.ones((32,), dtype=mx.float32)
        y = ext.w8a8_linear(mx.array(x), mx.array(w), scales)
        np.testing.assert_array_equal(np.array(y)[0, :7], [127, 0, 2, 2, 0, -2, -2])
        zero = ext.w8a8_linear(mx.zeros((5, 32)), mx.array(w), scales, mx.ones((32,)))
        np.testing.assert_array_equal(np.array(zero), np.ones((5, 32)))

    def test_bfloat16_does_not_round_through_float16(self):
        rng = np.random.default_rng(44)
        x = mx.array((rng.normal(size=(3, 64)) * 1e8).astype(np.float32)).astype(
            mx.bfloat16
        )
        w = rng.integers(-16, 17, (17, 64), dtype=np.int8)
        scales = np.full(17, 1e-8, dtype=np.float32)
        y = ext.w8a8_linear(x, mx.array(w), mx.array(scales))
        expected = reference_linear(x, w, scales)
        self.assertTrue(bool(mx.all(mx.isfinite(y))))
        np.testing.assert_allclose(
            np.array(y.astype(mx.float32)), expected, rtol=0.008, atol=0.25
        )

    def test_noncontiguous_compile_and_stream(self):
        rng = np.random.default_rng(45)
        x = mx.array(rng.normal(size=(4, 128)).astype(np.float32))[:, ::2]
        weight = mx.array(rng.integers(-127, 128, (64, 19), dtype=np.int8)).T
        scales = mx.full((19,), 0.01)
        fn = lambda a, w, s: ext.w8a8_linear(a, w, s)
        compiled = mx.compile(fn)(x, weight, scales)
        stream = mx.new_stream(mx.gpu)
        streamed = ext.w8a8_linear(x, weight, scales, stream=stream)
        expected = reference_linear(x, np.array(weight), np.array(scales))
        mx.eval(compiled, streamed)
        for actual in (compiled, streamed):
            np.testing.assert_allclose(np.array(actual), expected, rtol=2e-5, atol=2e-5)
        raw = ext.int8_matmul_int32(weight, weight, stream=stream)
        np.testing.assert_array_equal(
            np.array(raw),
            np.array(weight).astype(np.int32) @ np.array(weight).astype(np.int32).T,
        )

    def test_module_preserves_existing_weights(self):
        weight = mx.full((17, 64), -128, dtype=mx.int8)
        scales = mx.full((17, 1), 0.01)
        layer = ext.W8A8Linear(weight, scales)
        self.assertIs(layer.weight, weight)
        self.assertIs(layer.scales, scales)
        self.assertEqual((layer.input_dims, layer.output_dims), (64, 17))
        self.assertEqual(layer.trainable_parameters(), {})
        actual = layer(mx.ones((2, 3, 64)))
        np.testing.assert_allclose(np.array(actual), -81.92, rtol=2e-6)

    def test_invalid_inputs(self):
        x = mx.ones((2, 64))
        w = mx.ones((17, 64), dtype=mx.int8)
        s = mx.ones((17,))
        for args in (
            (mx.ones((2, 63)), w, s),
            (x, w.astype(mx.uint8), s),
            (x.astype(mx.int8), w, s),
            (x, w, s.astype(mx.float16)),
            (x, w, mx.ones((16,))),
            (x, w, mx.ones((1, 17))),
            (mx.ones((0, 64)), w, s),
            (mx.ones((2, 33)), mx.ones((17, 33), dtype=mx.int8), s),
        ):
            with (
                self.subTest(shapes=[a.shape for a in args]),
                self.assertRaises(ValueError),
            ):
                ext.w8a8_linear(*args)
        with self.assertRaises(ValueError):
            ext.w8a8_linear(x, w, s, mx.ones((16,)))
        with self.assertRaises(ValueError):
            ext.w8a8_linear(x, w, s, stream=mx.cpu)
        with self.assertRaises(ValueError):
            ext.int8_matmul_int32(x, w)
        with self.assertRaises(ValueError):
            ext.int8_matmul_int32(w[0], w)


if __name__ == "__main__":
    unittest.main()
