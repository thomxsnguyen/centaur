import os
from dotenv import load_dotenv
from transformers import pipeline


load_dotenv()

hf_token = os.environ.get("HF_TOKEN")


pipeline_kwargs = dict(
    model="SakanaAI/TinySwallow-1.5B",
    trust_remote_code=True,
)
if hf_token:
    pipeline_kwargs["use_auth_token"] = hf_token


try:
    pipe = pipeline("text-generation", **pipeline_kwargs, device_map="auto")
except Exception:
    pipe = pipeline("text-generation", **pipeline_kwargs)


messages = [
    {"role": "user", "content": "What is the capital of France?"},
]

gen_kwargs = dict(max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.95)
result = pipe(messages, **gen_kwargs)
print(result)
