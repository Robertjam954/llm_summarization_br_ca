import os
from dotenv import load_dotenv

env_sys = os.environ.get("ANTHROPIC_API_KEY", "NOT_SET")
print(f"System env  : {env_sys[:14] if env_sys != 'NOT_SET' else 'NOT_SET'}...")

load_dotenv(override=True)
k = os.environ.get("ANTHROPIC_API_KEY", "")
print(f".env key    : {k[:14]}...  len={len(k)}")

import anthropic
client = anthropic.Anthropic(api_key=k)
try:
    msg = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )
    print(f"API OK: {msg.content[0].text[:30]}")
except anthropic.AuthenticationError as e:
    print(f"401 AuthError — key is invalid or expired: {e}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
