# Third-party sources

## Cider

`src/kernels/w8a8_matmul.metal` is adapted from
[Mininglamp-AI/cider](https://github.com/Mininglamp-AI/cider), specifically
`cider/kernels/w8a8_matmul.metal`, as retrieved on 2026-10-20.
The retrieved upstream shader's SHA256 is
`055f2f654ef95f63128cb64c7000facb6a57eae5321fb9e4d03225c952389cc9`.

Local changes mask loads at partial M/N tiles and add FP16/BF16/FP32 output
variants. Activation quantization and the MLX primitive wrappers are local
implementations. No external `cider` package is required.

Copyright (c) 2026 Mininglamp contributors. The MIT license is reproduced in
`LICENSES/cider-MIT.txt` and included in the wheel's license metadata.

## MLX

Several kernels adapt code or implementation structures from
[Apple MLX](https://github.com/ml-explore/mlx), version 0.32.4:

- BF16 and quantized attention: `sdpa_vector.h` online softmax and merge.
- Quantized prefill: `steel/attn/mma.h` SIMD fragment coordinate mapping.
- W4A16 projection: `quantized.h` W4/P2 expressions.
- W8A8 projection: `steel_gemm_thin_nax.h` tensor-view and split-K design.

Local wrappers and kernels describe their adaptations in their source comments.
Apple copyright notices are retained; the MLX MIT license text is reproduced
in the package's `LICENSE` file. This package links against the installed MLX
library rather than distributing a replacement MLX runtime.
