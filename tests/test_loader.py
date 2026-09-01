from pathlib import Path

from spark_mlx_llm import loader


def test_tokenizer_loading_delegates_to_mlx_lm(monkeypatch):
    model_path = Path("spark2_5")
    tokenizer_config = {"trust_remote_code": True}
    expected = object()

    def fake_load_tokenizer(path, *, tokenizer_config_extra, eos_token_ids):
        assert path == model_path
        assert tokenizer_config_extra == tokenizer_config
        assert eos_token_ids == 1
        return expected

    monkeypatch.setattr(loader, "load_tokenizer", fake_load_tokenizer)

    tokenizer = loader._load_tokenizer(model_path, tokenizer_config, 1)

    assert tokenizer is expected
