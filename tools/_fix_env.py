"""One-shot .env key fixer — strips accidental quote wrapping."""
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)

KEYS_TO_FIX = ("ANTHROPIC_API_KEY", "VOYAGEAI_API_KEY", "OPENAI_API_KEY")
fixed = []
for ln in lines:
    stripped = ln.rstrip("\n\r")
    matched = False
    for key in KEYS_TO_FIX:
        if stripped.startswith(key + "="):
            val = stripped[len(key) + 1:]
            val = val.strip().strip('"').strip("'")
            fixed.append(f"{key}={val}\n")
            matched = True
            break
    if not matched:
        fixed.append(ln)

env_path.write_text("".join(fixed), encoding="utf-8")
print("Fixed. Verifying...")

from dotenv import load_dotenv
import os
load_dotenv(override=True)
k = os.environ.get("ANTHROPIC_API_KEY", "")
print(f"ANTHROPIC prefix : {k[:14]}...  len={len(k)}")
