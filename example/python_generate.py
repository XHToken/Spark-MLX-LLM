"""Python API 示例：generate() 返回完整字符串，可直接交给业务代码处理。"""

from mlx_lm.sample_utils import make_sampler

from spark_mlx_llm import generate, load

# 修改为实际模型目录；也可以填写 Hugging Face 仓库 ID。
MODEL_PATH = "/path/to/converted-mlx-model"
MESSAGES = [{"role": "user", "content": "请用一句话介绍你自己。"}]


def main():
    # load() 负责模型与算子配置；native 选择已有 W8 层的原生实现。
    model, tokenizer = load(MODEL_PATH, w8_qmv_backend="native")
    # generate() 接收已构造的 prompt；思考开关属于聊天模板参数。
    prompt = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,  # 本次最多生成的 token 数。
        sampler=make_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0.0),
        kv_bits=None,  # 默认保留模型精度（BF16 模型使用 BF16 KV）；设为 8 可启用量化。
        kv_group_size=32,
        quantized_kv_start=0,
        kv_quantization_scope="full-attention",  # 启用量化时仅作用于全局注意力 KV。
        prefill_step_size=512,  # 输入 prompt 的分块大小。
    )
    # text 是完整的 str，可以保存、返回给调用方，或继续处理。
    print(text)


if __name__ == "__main__":
    main()
