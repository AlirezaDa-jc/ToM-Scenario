from __future__ import annotations

import os
from dotenv import load_dotenv
import google.generativeai as genai

from benchmark.adapter import ModelAdapter

load_dotenv()


class GeminiAdapter(ModelAdapter):

    def __init__(self, model: str = "gemini-3-flash-preview"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self._model_name = model

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text

    @property
    def name(self) -> str:
        return self._model_name