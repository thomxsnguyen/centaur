from transformers import pipeline

# Quick test for TinySwallow-1.5B using Hugging Face transformers pipeline
pipe = pipeline("text-generation", model="SakanaAI/TinySwallow-1.5B")
messages = [
    {"role": "user", "content": "What is the capital of France?"},
]
result = pipe(messages)
print(result)