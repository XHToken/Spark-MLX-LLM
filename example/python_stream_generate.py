"""Python API 示例：stream_generate() 逐步返回响应，适合实时显示。"""

from mlx_lm.sample_utils import make_sampler

from spark_mlx_llm import load, stream_generate

# 与 python_generate.py 使用相同的模型、输入和参数，仅获取结果的方式不同。
MODEL_PATH = "/path/to/converted-mlx-model"
MESSAGES = [{"role": "user", "content": "请用一句话介绍你自己。"}]


def main():
    model, tokenizer = load(MODEL_PATH, w8_qmv_backend="native")
    prompt = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # response.text 是本次增量，不是累计的完整回答。
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,
        sampler=make_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0.0),
        kv_bits=None,  # 默认保留模型精度（BF16 模型使用 BF16 KV）；设为 8 可启用量化。
        kv_group_size=32,
        quantized_kv_start=0,
        kv_quantization_scope="full-attention",  # 启用量化时仅作用于全局注意力 KV。
        prefill_step_size=512,
    ):
        print(response.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
