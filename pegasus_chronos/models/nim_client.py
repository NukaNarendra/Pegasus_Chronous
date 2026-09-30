"""
PEGASUS-CHRONOS Research Series (Paper 3 of 3)
Prior work: ARGUS (Zenodo:10.5281/zenodo.21310433),
            SHADE (2026)

NOVELTY STATEMENT:
All prior memory poisoning work (eTAMP arXiv:2605.16746,
CSTM-Bench arXiv:2604.21131, MemoryGraft arXiv:2512.16962)
studies EXTERNAL attackers injecting into agent memory.

PEGASUS studies the MODEL ITSELF as the source of ghost
states, deliberately planted during evaluation windows with
reduced safety controls — directly motivated by the
GPT-5.6 Sol / Hugging Face incident of July 21-22, 2026.

This module implements the NVIDIA NIM API client for Part I
empirical evaluation across 6 frontier model architectures.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelResponse(BaseModel):
    content: str
    model: str
    usage: TokenUsage
    finish_reason: str
    latency_ms: float


class NIMModelConfig(BaseModel):
    model_id: str
    display_name: str
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 4096


class NIMAPIError(Exception):
    def __init__(self, status_code: int, message: str, model: str):
        self.status_code = status_code
        self.message = message
        self.model = model
        super().__init__(f"NIMAPIError [{status_code}] for model {model}: {message}")


class NIMClient:
    ALL_MODELS = {
        "nemotron_120b": "nvidia/nemotron-3-super-120b-a12b",
        "llama_70b": "meta/llama-3.3-70b-instruct",
        "qwen_80b": "qwen/qwen3-next-80b-a3b-instruct",
        "deepseek_flash": "deepseek-ai/deepseek-v4-flash",
        "minimax": "minimaxai/minimax-m2.7",
        "gpt_oss": "openai/gpt-oss-120b",
        "judge": "meta/llama-3.1-8b-instruct",
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        max_retries: int = 3,
        max_concurrent: int = 10,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("NVIDIA_NIM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NVIDIA NIM API key not found. Please set NVIDIA_NIM_API_KEY environment variable or pass it to NIMClient."
            )
        
        self.base_url = base_url
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )

    @classmethod
    def get_model_id(cls, short_name: str) -> str:
        if short_name not in cls.ALL_MODELS:
            raise ValueError(f"Unknown model short name: {short_name}. Available models: {list(cls.ALL_MODELS.keys())}")
        return cls.ALL_MODELS[short_name]

    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls.ALL_MODELS.keys())

    async def __aenter__(self) -> NIMClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _request_with_retry(
        self, method: str, endpoint: str, model: str, **kwargs: Any
    ) -> httpx.Response:
        from pegasus_chronos.core.exceptions import (
            ContextWindowExceededError,
            MaxRetriesExceededError,
            ModelDeprecatedError,
        )
        base_delay = 1.0
        max_delay = 32.0
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(method, endpoint, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                error_text = e.response.text
                
                # PERMANENT: Context window exceeded (DO NOT RETRY)
                if status_code == 413 or "request too large" in error_text.lower():
                    raise ContextWindowExceededError(
                        message=error_text, model=model,
                    ) from e
                
                # PERMANENT: Model deprecated (DO NOT RETRY)
                if status_code == 410:
                    raise ModelDeprecatedError(
                        message=error_text, model=model,
                    ) from e
                
                # RETRYABLE: 429, 5xx
                if status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 1)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} for model {model} due to HTTP {status_code}. Waiting {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise NIMAPIError(
                        status_code=status_code,
                        message=error_text,
                        model=model
                    ) from e
            except httpx.RequestError as e:
                if attempt < self.max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 1)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} for model {model} due to RequestError: {str(e)}. Waiting {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise MaxRetriesExceededError(
                        message=f"Connection failed after {self.max_retries} retries: {e}",
                        model=model,
                        attempts=self.max_retries,
                    ) from e
        
        raise MaxRetriesExceededError(
            message="Max retries exhausted",
            model=model,
            attempts=self.max_retries,
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None
    ) -> ModelResponse:
        model_id = self.get_model_id(model) if model in self.ALL_MODELS else model
        
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
            
        start_time = time.time()
        
        async with self._semaphore:
            response = await self._request_with_retry(
                method="POST",
                endpoint="/chat/completions",
                model=model_id,
                json=payload
            )
            
        latency_ms = (time.time() - start_time) * 1000.0
        data = response.json()
        
        choice = data["choices"][0]
        usage = data.get("usage", {})
        
        token_usage = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        
        model_response = ModelResponse(
            content=choice.get("message", {}).get("content", ""),
            model=data.get("model", model_id),
            usage=token_usage,
            finish_reason=choice.get("finish_reason", "unknown"),
            latency_ms=latency_ms
        )
        
        logger.debug(
            f"Generated response for {model_id} - Latency: {latency_ms:.2f}ms - "
            f"Tokens: {token_usage.total_tokens} (P: {token_usage.prompt_tokens}, C: {token_usage.completion_tokens})"
        )
        
        return model_response

    async def generate_batch(
        self, requests: list[dict], concurrency: int | None = None
    ) -> list[Union[ModelResponse, Exception]]:
        """
        Runs multiple generate() calls with semaphore-limited concurrency.
        """
        if concurrency is not None:
            # Override semaphore for this batch
            sem = asyncio.Semaphore(concurrency)
            async def _bounded_generate(req: dict) -> ModelResponse:
                async with sem:
                    return await self.generate(**req)
            tasks = [_bounded_generate(req) for req in requests]
        else:
            tasks = [self.generate(**req) for req in requests]
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {type(result).__name__} - {str(result)}")
                
        return list(results)

    async def judge(self, prompt: str, system_prompt: str = '') -> ModelResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return await self.generate(
            messages=messages,
            model="judge",
            temperature=0.0
        )
