"""Quick script to list available Groq models."""
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")
r = httpx.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"},
)
data = r.json()
models = data.get("data", [])
for m in sorted(models, key=lambda x: x["id"]):
    print(m["id"])
