"""Quick smoke test: verify Groq API works with openai/gpt-oss-120b."""
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from pegasus_chronos.models.groq_client import GroqClient

async def test():
    client = GroqClient(model_name="openai/gpt-oss-120b")
    resp = await client.generate_detailed(
        [{"role": "user", "content": "Say hello in exactly 5 words."}],
        temperature=0.0,
        max_tokens=50,
    )
    print(f"Model: {resp.model}")
    print(f"Content: {resp.content}")
    print(f"Latency: {resp.latency_ms:.1f}ms")
    print(f"Tokens: {resp.usage.total_tokens}")
    print("SMOKE TEST PASSED")

asyncio.run(test())
