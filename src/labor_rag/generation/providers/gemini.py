"""Gemini API provider using Google's current ``google-genai`` SDK."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseLLMProvider, ProviderResponse


class GeminiProvider(BaseLLMProvider):
    provider_name = "gemini"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client: Any | None = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "GeminiProvider requires the 'google-genai' package. "
                "Install: pip install -r requirements-generation.txt"
            ) from exc

        if not self.api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Add it to the environment/.env before "
                "running the Gemini provider."
            )

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> ProviderResponse:
        try:
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "GeminiProvider requires the 'google-genai' package."
            ) from exc

        client = self._ensure_client()
        response = client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

        text = getattr(response, "text", None) or ""
        usage = self._usage_dict(getattr(response, "usage_metadata", None))
        return ProviderResponse(
            text=text,
            provider=self.provider_name,
            model=self.model_name,
            usage=usage,
        )

    @staticmethod
    def _usage_dict(metadata: Any) -> dict[str, Any]:
        if metadata is None:
            return {}

        fields = (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "cached_content_token_count",
            "thoughts_token_count",
        )
        result: dict[str, Any] = {}
        for field in fields:
            value = getattr(metadata, field, None)
            if value is not None:
                result[field] = value
        return result
