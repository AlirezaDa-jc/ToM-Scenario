from __future__ import annotations

import torch

from benchmark.adapter import ModelAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class HuggingFaceAdapter(ModelAdapter):
    """
    Local inference adapter using HuggingFace Transformers.
    Loads model once at initialization — reused across all prompts.
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 64,
        device: str = "auto",
    ):
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens

        print(f"Loading model: {model_id} ...")

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
        )

        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        print(f"Model loaded: {model_id}")

    def generate(self, prompt: str) -> str:
        # Use chat template if available, otherwise raw prompt
        if self._tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            output = self._pipe(messages)
            return output[0]["generated_text"][-1]["content"].strip()
        else:
            output = self._pipe(prompt)
            # Strip the input prompt from output
            generated = output[0]["generated_text"]
            if generated.startswith(prompt):
                generated = generated[len(prompt):]
            return generated.strip()

    @property
    def name(self) -> str:
        # e.g. "Qwen/Qwen2.5-7B-Instruct" → "Qwen2.5-7B-Instruct"
        return self._model_id.split("/")[-1]