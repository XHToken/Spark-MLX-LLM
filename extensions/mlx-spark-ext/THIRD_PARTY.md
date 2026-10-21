# Third-party sources

`src/kernels/w8a8_matmul.metal` is adapted from
[Mininglamp-AI/cider](https://github.com/Mininglamp-AI/cider), specifically
[`cider/kernels/w8a8_matmul.metal`](https://github.com/Mininglamp-AI/cider/blob/main/cider/kernels/w8a8_matmul.metal)
as retrieved on 2026-10-20. Its MIT license is included in
`LICENSES/cider-MIT.txt` and in the wheel's license metadata.
The retrieved upstream shader's SHA256 is
`055f2f654ef95f63128cb64c7000facb6a57eae5321fb9e4d03225c952389cc9`.

Local changes mask loads at partial M/N tiles and add native BF16 and FP32
output variants. The extension's activation quantizer and MLX primitive
wrappers are local implementations. They preserve the input floating dtype
and perform INT8 activation quantization even for a single token. Cider's
automatic model conversion, W4A8, SDPA, and float-activation decode fallback
are not included. No external `cider` package is required.
