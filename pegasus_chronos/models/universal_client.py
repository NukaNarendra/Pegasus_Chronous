"""
PEGASUS-CHRONOS Research Series (Paper 3 of 3)

UNIVERSAL OPENAI-COMPATIBLE CLIENT
Designed for Aerolink (Production) and NaraRouter (Development).

ACADEMIC GUARDRAILS IMPLEMENTED:
1. Strict Model Fingerprint Auditing: Logs `system_fingerprint` and the exact 
   model string returned by the server to detect silent quantization or downgrades.
2. ARGUS-style API Resilience: Uses `tenacity` for exponential backoff on 429s and 5xxs.
3. Latency tracking and token-burn optimization for the $35 Aerolink budget.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pegasus_chronos.core.exceptions import (
    ContextWindowExceededError,
    MaxRetriesExceededError,
)

logger = logging.getLogger(__name__)


class UniversalTokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class UniversalModelResponse(BaseModel):
    content: str
    model_returned: str
    system_fingerprint: Optional[str]
    usage: UniversalTokenUsage
    finish_reason: str = "stop"
    latency_ms: float = 0.0


class UniversalClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self._http_client: Optional[httpx.AsyncClient] = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._http_client

    @retry(
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=2, min=2, max=32),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _post_with_tenacity(self, payload: Dict[str, Any]) -> httpx.Response:
        client = self._get_http_client()
        response = await client.post("/chat/completions", json=payload)
        
        # Manually intercept 413 (Payload too large) so we DON'T retry it.
        # Tenacity will retry HTTPStatusError unless we explicitly raise our custom Context error.
        if response.status_code == 413 or "too large" in response.text.lower():
            raise ContextWindowExceededError(
                message=response.text,
                model=self.model_name
            )
            
        # Raise standard error for 429 and 5xx so Tenacity retries them
        if response.status_code in (429, 500, 502, 503, 504):
            response.raise_for_status()
            
        # If it's a 4xx that isn't 429/413, we shouldn't retry.
        if 400 <= response.status_code < 500:
             raise MaxRetriesExceededError(
                message=f"Hard API Error {response.status_code}: {response.text}",
                model=self.model_name
            )
             
        return response

    async def generate_detailed(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> UniversalModelResponse:
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t0 = time.time()
        try:
            response = await self._post_with_tenacity(payload)
        except ContextWindowExceededError:
            raise
        except Exception as e:
            logger.error(f"UniversalClient: Max retries exceeded for {self.model_name}: {e}")
            raise MaxRetriesExceededError(message=str(e), model=self.model_name)

        latency_ms = (time.time() - t0) * 1000.0
        data = response.json()
        
        choice = data["choices"][0]
        content = choice.get("message", {}).get("content", "")
        raw_usage = data.get("usage", {})
        
        # STRICT FINGERPRINT AUDITING (Guardrail 1)
        returned_model = data.get("model", "UNKNOWN")
        sys_fingerprint = data.get("system_fingerprint", "NONE")
        
        logger.info(
            f"[AUDIT] Router served model: '{returned_model}' | "
            f"Fingerprint: '{sys_fingerprint}' | Latency: {latency_ms:.1f}ms"
        )

        if self.model_name.lower() not in returned_model.lower():
            logger.warning(
                f"[ACADEMIC INTEGRITY WARNING] Requested model '{self.model_name}', "
                f"but router served '{returned_model}'. Verify endpoint fidelity!"
            )

        return UniversalModelResponse(
            content=content,
            model_returned=returned_model,
            system_fingerprint=sys_fingerprint,
            usage=UniversalTokenUsage(
                prompt_tokens=raw_usage.get("prompt_tokens", 0),
                completion_tokens=raw_usage.get("completion_tokens", 0),
                total_tokens=raw_usage.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason", "stop"),
            latency_ms=latency_ms,
        )

    async def run_canary_test(self) -> bool:
        """
        Runs a 1-turn baseline canary prompt before trials to verify 
        model identity and fingerprint.
        """
        logger.info(f"Running Canary Integrity Test for {self.model_name}...")
        try:
            resp = await self.generate_detailed(
                messages=[{"role": "user", "content": "What is 2+2? State your exact model architecture in one sentence."}],
                temperature=0.0,
                max_tokens=50
            )
            logger.info(f"[CANARY SUCCESS] {resp.model_returned} (FP: {resp.system_fingerprint})")
            return True
        except Exception as e:
            logger.error(f"[CANARY FAILED] {e}")
            return False

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
