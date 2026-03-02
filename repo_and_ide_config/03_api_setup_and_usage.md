# API Setup and Usage

Reference for all external APIs used in this project.

---

## Environment Variable Setup

All keys are stored in `.env` (never committed). Template is in `.env.example`.

```powershell
# Copy template and fill in keys
Copy-Item .env.example .env
# Then edit .env in VSCode
```

`.env` structure:
```ini
PROJECT_ROOT=C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca
DATA_PRIVATE_DIR=C:\Users\jamesr4\loc\data_private

ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Load in any notebook or script:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
```

---

## Anthropic (Claude) API

**Used in:** NB04 (source document text extraction via Claude Vision + Transcription)

### Installation
```bash
pip install anthropic
```

### Get API Key
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Settings → API Keys → Create Key
3. Add to `.env` as `ANTHROPIC_API_KEY`

### Basic Usage
```python
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Extract clinical features from this text: ..."}]
)
print(message.content[0].text)
```

### Vision / Image Input (NB04 use case)
```python
import base64

with open("path/to/page.png", "rb") as f:
    img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
            {"type": "text", "text": "Transcribe all text from this scanned medical document."}
        ]
    }]
)
```

### Rate Limits & Cost Notes
- Tier 1 (new accounts): ~50K input tokens/min
- Claude Sonnet is ~5–10x cheaper than Opus for bulk extraction runs
- NB04 batch processing: use `time.sleep(1)` between calls to avoid 429 errors

---

## OpenAI API

**Used in:** Prompt benchmarking (NB07), optional LLM-as-judge comparisons

### Installation
```bash
pip install openai
```

### Get API Key
1. [platform.openai.com](https://platform.openai.com) → API Keys → Create
2. Add to `.env` as `OPENAI_API_KEY`

### Basic Usage
```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this clinical note: ..."}]
)
print(response.choices[0].message.content)
```

---

## DeepEval (LLM-as-Judge Framework)

**Used in:** `src/llm_eval_by_llm/deep_eval_llm_judge_api.py`

### Installation
```bash
pip install deepeval
deepeval login   # generates a key at app.confident-ai.com
```

### Basic Usage Pattern (from project source)
```python
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is the biopsy result?",
    actual_output="The biopsy was malignant.",
    context=["The biopsy showed invasive ductal carcinoma."]
)
metric = HallucinationMetric(threshold=0.5)
evaluate([test_case], [metric])
```

---

## Google Colab API Access

When running from Colab, use Colab Secrets instead of a `.env` file:

1. Left sidebar → lock icon → **Secrets**
2. Add `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GITHUB_PAT`

```python
from google.colab import userdata
import os

os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
os.environ["OPENAI_API_KEY"]    = userdata.get("OPENAI_API_KEY")
```

> See `docs/colab_pipeline_guide.md` for the full Colab session setup cell.

---

## Security Rules

- **Never** hardcode API keys in notebooks or scripts
- **Never** commit `.env` (it is in `.gitignore`)
- Rotate keys immediately if accidentally pushed to GitHub
- Use `ANTHROPIC_API_KEY` variable name consistently — all notebooks and `src/` scripts expect this name
- If a key is exposed: revoke it at the provider console first, then rotate
