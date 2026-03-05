import anthropic
import os

# Set your API key - either hardcode it or use an environment variable (recommended)
# export ANTHROPIC_API_KEY="your-api-key-here"
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def chat(user_message: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    return message.content[0].text

if __name__ == "__main__":
    # Simple one-shot example
    response = chat("Hello! What can you help me with today?")
    print(response)