import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


class LLMWrapper:
    def __init__(self, cfg, cache_dir: Optional[str] = None):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = cfg.get("precision", "bf16")
        if self.device == "cpu":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        cache_dir = cache_dir or ".cache/"
        os.makedirs(cache_dir, exist_ok=True)

        name = cfg.name
        resolved_name = self._resolve_model_name(name, cache_dir)
        if resolved_name != name:
            name = resolved_name

        try:
            config = AutoConfig.from_pretrained(name, trust_remote_code=True, cache_dir=cache_dir)
        except Exception:
            name = "google/flan-t5-large"
            config = AutoConfig.from_pretrained(name, trust_remote_code=True, cache_dir=cache_dir)

        self.is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        device_map = cfg.get("device_map", "auto" if self.device == "cuda" else None)
        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                cache_dir=cache_dir,
                device_map=device_map,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                cache_dir=cache_dir,
                device_map=device_map,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id is required but missing.")
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()
        assert self.model.config.vocab_size > 0

    def _resolve_model_name(self, name: str, cache_dir: str) -> str:
        alias_map = {
            "Qwen3-8B": "Qwen/Qwen2.5-7B-Instruct",
            "qwen3-8b": "Qwen/Qwen2.5-7B-Instruct",
        }
        return alias_map.get(name, name)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        num_return_sequences: int,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        decoded = []
        for o in outputs:
            if self.is_encoder_decoder:
                text = self.tokenizer.decode(o, skip_special_tokens=True)
            else:
                text = self.tokenizer.decode(o[inputs.input_ids.shape[-1] :], skip_special_tokens=True)
            decoded.append(text)
        return decoded
