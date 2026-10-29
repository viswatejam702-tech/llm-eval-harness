"""
LLM Provider Adapters — Unified interface for querying any LLM.

Supports: OpenAI, Anthropic, Google Gemini, Ollama (local), HuggingFace Inference,
and any OpenAI-compatible endpoint (vLLM, LM Studio, Together AI, Groq, etc.)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("eval-harness.providers")


# ─── Data Structures ──────────────────────────────────────────────────────────

class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass
class LLMResponse:
    """Standardised response container from any provider."""
    text: str
    model: str
    provider: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms <= 0 or self.completion_tokens <= 0:
            return 0.0
        return (self.completion_tokens / self.latency_ms) * 1000.0


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    provider_type: ProviderType
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    timeout_s: float = 120.0
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def display_name(self) -> str:
        return f"{self.provider_type.value}:{self.model}"


# ─── Abstract Base ─────────────────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    """Abstract LLM provider interface."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_s)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def generate_multi_turn(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        ...


# ─── OpenAI Provider ──────────────────────────────────────────────────────────

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API (GPT-4o, GPT-4.1, o3, etc.)"""

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.generate_multi_turn(messages, **kwargs)

    async def generate_multi_turn(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        client = await self._get_client()
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY", "")
        base_url = self.config.base_url or "https://api.openai.com/v1"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            **self.config.extra_params,
        }

        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            latency = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            usage = data.get("usage", {})

            return LLMResponse(
                text=choice["message"]["content"],
                model=data.get("model", self.config.model),
                provider="openai",
                latency_ms=latency,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                finish_reason=choice.get("finish_reason", ""),
                raw=data,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(f"OpenAI call failed: {e}")
            return LLMResponse(
                text="", model=self.config.model, provider="openai",
                latency_ms=latency, error=str(e),
            )


# ─── Anthropic Provider ───────────────────────────────────────────────────────

class AnthropicProvider(BaseLLMProvider):
    """Anthropic API (Claude Opus, Sonnet, Haiku)"""

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_multi_turn(messages, system_prompt=system_prompt, **kwargs)

    async def generate_multi_turn(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        client = await self._get_client()
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        base_url = self.config.base_url or "https://api.anthropic.com"

        system_prompt = kwargs.pop("system_prompt", None)
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if system_prompt:
            payload["system"] = system_prompt

        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{base_url}/v1/messages",
                json=payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )
            latency = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()

            text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            usage = data.get("usage", {})

            return LLMResponse(
                text="\n".join(text_blocks),
                model=data.get("model", self.config.model),
                provider="anthropic",
                latency_ms=latency,
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                finish_reason=data.get("stop_reason", ""),
                raw=data,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(f"Anthropic call failed: {e}")
            return LLMResponse(
                text="", model=self.config.model, provider="anthropic",
                latency_ms=latency, error=str(e),
            )


# ─── Google Gemini Provider ───────────────────────────────────────────────────

class GeminiProvider(BaseLLMProvider):
    """Google Gemini API (gemini-2.5-pro, gemini-2.5-flash, etc.)"""

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "user", "content": system_prompt})
            messages.append({"role": "model", "content": "Understood."})
        messages.append({"role": "user", "content": prompt})
        return await self.generate_multi_turn(messages, **kwargs)

    async def generate_multi_turn(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        client = await self._get_client()
        api_key = self.config.api_key or os.getenv("GEMINI_API_KEY", "")
        model = self.config.model

        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ("user", "system") else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
                "topP": kwargs.get("top_p", self.config.top_p),
            },
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, json=payload)
            latency = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()

            candidates = data.get("candidates", [{}])
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "\n".join(p.get("text", "") for p in parts)
            usage = data.get("usageMetadata", {})

            return LLMResponse(
                text=text,
                model=model,
                provider="gemini",
                latency_ms=latency,
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
                total_tokens=usage.get("totalTokenCount", 0),
                finish_reason=candidates[0].get("finishReason", ""),
                raw=data,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(f"Gemini call failed: {e}")
            return LLMResponse(
                text="", model=model, provider="gemini",
                latency_ms=latency, error=str(e),
            )


# ─── Ollama (Local) Provider ─────────────────────────────────────────────────

class OllamaProvider(BaseLLMProvider):
    """Ollama local models (llama3, mistral, qwen, deepseek, etc.)"""

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.generate_multi_turn(messages, **kwargs)

    async def generate_multi_turn(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        client = await self._get_client()
        base_url = self.config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
            },
        }

        t0 = time.perf_counter()
        try:
            resp = await client.post(f"{base_url}/api/chat", json=payload)
            latency = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()

            return LLMResponse(
                text=data.get("message", {}).get("content", ""),
                model=data.get("model", self.config.model),
                provider="ollama",
                latency_ms=latency,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                finish_reason=data.get("done_reason", ""),
                raw=data,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(f"Ollama call failed: {e}")
            return LLMResponse(
                text="", model=self.config.model, provider="ollama",
                latency_ms=latency, error=str(e),
            )


# ─── HuggingFace Inference Provider ──────────────────────────────────────────

class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API (serverless or dedicated endpoints)"""

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.generate_multi_turn(messages, **kwargs)

    async def generate_multi_turn(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        client = await self._get_client()
        api_key = self.config.api_key or os.getenv("HUGGINGFACE_API_KEY", "")
        base_url = self.config.base_url or "https://api-inference.huggingface.co"
        model = self.config.model

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False,
        }

        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{base_url}/models/{model}/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            latency = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})

            return LLMResponse(
                text=choice["message"]["content"],
                model=data.get("model", model),
                provider="huggingface",
                latency_ms=latency,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                finish_reason=choice.get("finish_reason", ""),
                raw=data,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(f"HuggingFace call failed: {e}")
            return LLMResponse(
                text="", model=model, provider="huggingface",
                latency_ms=latency, error=str(e),
            )


# ─── OpenAI-Compatible Provider ──────────────────────────────────────────────

class OpenAICompatibleProvider(OpenAIProvider):
    """Any OpenAI-compatible endpoint (vLLM, LM Studio, Together AI, Groq, etc.)
    Just set base_url and (optionally) api_key in config."""

    async def generate_multi_turn(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        result = await super().generate_multi_turn(messages, **kwargs)
        result.provider = "openai_compatible"
        return result


# ─── Factory ──────────────────────────────────────────────────────────────────

_PROVIDER_MAP = {
    ProviderType.OPENAI: OpenAIProvider,
    ProviderType.ANTHROPIC: AnthropicProvider,
    ProviderType.GEMINI: GeminiProvider,
    ProviderType.OLLAMA: OllamaProvider,
    ProviderType.HUGGINGFACE: HuggingFaceProvider,
    ProviderType.OPENAI_COMPATIBLE: OpenAICompatibleProvider,
}


def create_provider(config: ProviderConfig) -> BaseLLMProvider:
    """Factory: instantiate the right provider from a ProviderConfig."""
    cls = _PROVIDER_MAP.get(config.provider_type)
    if cls is None:
        raise ValueError(f"Unknown provider type: {config.provider_type}")
    return cls(config)


def create_provider_from_dict(d: Dict[str, Any]) -> BaseLLMProvider:
    """Create a provider from a plain dict (e.g. loaded from YAML/JSON config)."""
    ptype = ProviderType(d["provider"])
    cfg = ProviderConfig(
        provider_type=ptype,
        model=d["model"],
        api_key=d.get("api_key"),
        base_url=d.get("base_url"),
        temperature=d.get("temperature", 0.0),
        max_tokens=d.get("max_tokens", 2048),
        top_p=d.get("top_p", 1.0),
        timeout_s=d.get("timeout_s", 120.0),
        extra_params=d.get("extra_params", {}),
    )
    return create_provider(cfg)
