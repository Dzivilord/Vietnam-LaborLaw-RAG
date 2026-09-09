"""Local Ollama provider using the native ``/api/chat`` endpoint."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from ..base import BaseLLMProvider, ProviderResponse


class OllamaProvider(BaseLLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,
        think: bool = False,
    ) -> None:
        self.model_name = model or os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = float(timeout)
        self.think = bool(think)

        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0.")

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> ProviderResponse:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running and the model has been pulled. "
                f"Original error: {exc.reason}"
            ) from exc

        data = json.loads(raw)
        message = data.get("message") or {}
        text = str(message.get("content") or "")

        usage = {
            key: data[key]
            for key in (
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
            )
            if key in data
        }
        metadata: dict[str, Any] = {
            "done_reason": data.get("done_reason"),
            "thinking_enabled": self.think,
        }
        if self.think and message.get("thinking"):
            metadata["thinking_returned"] = True

        return ProviderResponse(
            text=text,
            provider=self.provider_name,
            model=str(data.get("model") or self.model_name),
            usage=usage,
            raw_metadata=metadata,
        )
