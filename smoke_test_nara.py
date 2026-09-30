import asyncio
import logging
import os
from dotenv import load_dotenv

from pegasus_chronos.models.universal_client import UniversalClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test")

async def main():
    load_dotenv()
    nara_key = os.environ.get("NARA_API_KEY")
    nara_url = os.environ.get("NARA_BASE_URL", "https://router.bynara.id/v1")
    
    if not nara_key:
        logger.error("NARA_API_KEY not found in .env")
        return

    logger.info("Initializing UniversalClient via NaraRouter targeting GPT-4o...")
    client = UniversalClient(
        api_key=nara_key,
        base_url=nara_url,
        model_name="gpt-4o-2024-08-06"
    )

    # 1. Canary Test
    success = await client.run_canary_test()
    if not success:
        logger.error("Canary test failed. Stopping smoke test.")
        await client.close()
        return
        
    logger.info("Canary test passed! Checking token economics (latency & model routing).")
    
    # 2. Run a small empirical payload equivalent
    resp = await client.generate_detailed([
        {"role": "system", "content": "You are a secure evaluation agent."},
        {"role": "user", "content": "Summarize the access boundaries of your evaluation context."}
    ], max_tokens=100)
    
    logger.info(f"Test Generation Complete.")
    logger.info(f"Returned Model: {resp.model_returned}")
    logger.info(f"System Fingerprint: {resp.system_fingerprint}")
    logger.info(f"Tokens Used: {resp.usage.total_tokens}")
    logger.info(f"Latency: {resp.latency_ms:.1f} ms")
    logger.info(f"Response: {resp.content}")

    await client.close()
    
    logger.info("UniversalClient verified. Running 5-trial PEGASUS ECBS via master orchestrator...")
    
if __name__ == "__main__":
    asyncio.run(main())
