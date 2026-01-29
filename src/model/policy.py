"""Policy wrapper for Centaur using the TinySwallow model.

Expose `choose_action(prompt) -> 'Red'|'White'|'Black'`.
"""
import re
from typing import Optional

from dotenv import load_dotenv

# Import `get_pipe` robustly so running this file as a script won't fail
try:
    # normal package import when used as `from tinyswallow_llm import policy`
    from .model import get_pipe, get_generation_config
except Exception:
    # running as a script (no parent package); try absolute import first
    try:
        from model.model import get_pipe
    except Exception:
        # Last-resort: load model.py directly from this package directory
        import importlib.util
        import os

        model_path = os.path.join(os.path.dirname(__file__), "model.py")
        spec = importlib.util.spec_from_file_location("tinyswallow_llm.model", model_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        get_pipe = mod.get_pipe


def _parse_action(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b(Red|White|Black)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return None


def choose_action(prompt: str, max_retries: int = 1, default: str = "Red") -> str:
    """Choose action from model completion.

    - Ensures `CHOOSE_ACTION:` cue is present.
    - Uses `get_pipe()` from model package to obtain pipeline.
    - Parses first valid action; retries once with sampling if needed.
    """
    load_dotenv()
    pipe = get_pipe()

    if "CHOOSE_ACTION" not in prompt:
        if not prompt.endswith("\n"):
            prompt = prompt + "\n"
        prompt = prompt + "CHOOSE_ACTION: "

    # Load generation defaults from config; fall back to sensible defaults
    gen_cfg = get_generation_config() or {}
    gen_kwargs = dict(
        max_new_tokens=gen_cfg.get("max_new_tokens", 12),
        do_sample=gen_cfg.get("do_sample", False),
        temperature=gen_cfg.get("temperature", 0.0),
        top_p=gen_cfg.get("top_p", 1.0),
    )
    for attempt in range(max_retries + 1):
        try:
            out = pipe(prompt, **gen_kwargs)
        except TypeError:
            out = pipe(prompt)

        text = ""
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                text = first.get("generated_text", "")
            elif isinstance(first, str):
                text = first
        elif isinstance(out, dict):
            text = out.get("generated_text", "")
        elif isinstance(out, str):
            text = out

        if text.startswith(prompt):
            text = text[len(prompt) :]

        action = _parse_action(text)
        if action in {"Red", "White", "Black"}:
            return action

        if attempt < max_retries:
            retry_cfg = gen_cfg.get("retry", {})
            gen_kwargs = dict(
                max_new_tokens=retry_cfg.get("max_new_tokens", 16),
                do_sample=retry_cfg.get("do_sample", True),
                temperature=retry_cfg.get("temperature", 0.7),
                top_p=retry_cfg.get("top_p", 0.95),
            )

    return default


if __name__ == "__main__":
    sample_prompt = (
        "PARTICIPANT: TEST\n"
        "CONTEXT: driftwood_beach\n"
        "AVAILABLE_ACTIONS: Red, White, Black\n"
        "CHOOSE_ACTION: "
    )
    
    print("Chosen action:", choose_action(sample_prompt))
