import os
from dotenv import load_dotenv
from transformers import pipeline

# Load .env (if present) so HF_TOKEN is available in os.environ
load_dotenv()

# Optionally use HF_TOKEN for authenticated downloads (transformers will pick up)
hf_token = os.environ.get("HF_TOKEN")

# Quick test for TinySwallow-1.5B using Hugging Face transformers pipeline
# If you need to pass credentials explicitly to the pipeline (rare), set
# `use_auth_token=hf_token` when calling `pipeline(...)`.
pipe = pipeline("text-generation", model="SakanaAI/TinySwallow-1.5B")
messages = [
    {"role": "user", "content": "What is the capital of France?"},
]
result = pipe(messages)
print(result)