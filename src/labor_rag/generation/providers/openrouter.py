"""OpenRouter chat-completions provider.

OpenRouter exposes an OpenAI-compatible endpoint.  The optional ``models``
payload field expresses an ordered fallback list, so routing stays within a
single provider request and the API key never leaves the backend.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from ..base import BaseLLMProvider, ProviderResponse


class OpenRouterProvider(BaseLLMProvider):
    provider_name = "openrouter"

    def __init__(
        self,
        *,
        model: str | None = None,
        fallback_models: tuple[str, ...] = (),
        api_key: str | None = None,
        base_url: str | None = None,
        referer: str | None = None,
        title: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.model_name = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-chat-latest")
        self.fallback_models = tuple(item for item in fallback_models if item and item != self.model_name)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.referer = referer or os.getenv("OPENROUTER_REFERER")
        self.title = title or os.getenv("OPENROUTER_TITLE", "Labor Law RAG")
        self.timeout = float(timeout)
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY. Configure it on the backend environment.")

    def generate(self, *, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> ProviderResponse:
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        if self.fallback_models:
            payload["models"] = [self.model_name, *self.fallback_models]
        else:
            payload["model"] = self.model_name

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "X-OpenRouter-Metadata": "enabled"}
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-OpenRouter-Title"] = self.title
        http_request = request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail or exc.reason}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Cannot connect to OpenRouter: {exc.reason}") from exc

        choices = data.get("choices") or []
        message = choices[0].get("message") if choices else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return ProviderResponse(
            text=str(content or ""), provider=self.provider_name,
            model=str(data.get("model") or self.model_name), usage=dict(data.get("usage") or {}),
            raw_metadata={"id": data.get("id"), "finish_reason": choices[0].get("finish_reason") if choices else None, "routing": data.get("openrouter_metadata")},
        )
