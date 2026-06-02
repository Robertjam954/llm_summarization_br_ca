from dotenv import load_dotenv
import os, anthropic

load_dotenv(override=True)
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
models = client.models.list()
for m in models.data:
    print(m.id)
