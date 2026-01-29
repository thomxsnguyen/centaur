import os
from typing import Optional
from dotenv import load_dotenv
import yaml


_PIPE = None


def get_pipe(
    model_name: str = "SakanaAI/TinySwallow-1.5B",
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = True,
):
    """Lazily initialize and return a text-generation pipeline (singleton).

    Authentication: if `HF_TOKEN` is set in the environment or .env, prefer
    using `huggingface_hub.login(token=...)`. As a fallback the token is left
    in `os.environ` so the transformers/hub code can pick it up.
    """
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    # Prefer explicit login so huggingface_hub handles auth; fallback to env var
    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token)
        except Exception:
            os.environ["HF_TOKEN"] = hf_token

    from transformers import pipeline

    # Load config from one of several locations (env, package resources, package-relative)
    def _load_config_file():
        # 1) explicit env var override
        env_path = os.environ.get("TINYSWALLOW_CONFIG") or os.environ.get("TINY_CONFIG_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # 2) try package resources (works when installed)
        try:
            import importlib.resources as pkg_resources

            pkg_files = pkg_resources.files("tinyswallow_llm")
            candidate = pkg_files.joinpath("config").joinpath("tinyswallow_config.yaml")
            if candidate.exists():
                return str(candidate)
        except Exception:
            pass

        # 3) package-relative path (development checkout)
        pkg_rel = os.path.join(os.path.dirname(__file__), "config", "tinyswallow_config.yaml")
        if os.path.exists(pkg_rel):
            return pkg_rel

        # 4) repo-root config (in case script run from project root)
        repo_rel = os.path.join(os.getcwd(), "src", "tinyswallow_llm", "config", "tinyswallow_config.yaml")
        if os.path.exists(repo_rel):
            return repo_rel

        return None

    cfg = {}
    config_file = _load_config_file()
    if config_file:
        try:
            with open(config_file, "r", encoding="utf-8") as cf:
                cfg = yaml.safe_load(cf) or {}
        except Exception:
            cfg = {}

    # Allow config to override parameters passed to get_pipe
    model_name = cfg.get("model_name", model_name)
    device_map = cfg.get("device_map", device_map)
    trust_remote_code = cfg.get("trust_remote_code", trust_remote_code)

    pipeline_kwargs = dict(model=model_name, trust_remote_code=trust_remote_code)
    try:
        _PIPE = pipeline("text-generation", **pipeline_kwargs, device_map=device_map)
    except Exception:
        _PIPE = pipeline("text-generation", **pipeline_kwargs)

    return _PIPE


def get_generation_config() -> dict:
    """Return generation configuration loaded from package config file.

    This centralizes generation defaults so the policy wrapper doesn't
    hard-code arguments and the user can tune via YAML.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config", "tinyswallow_config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as cf:
            cfg = yaml.safe_load(cf) or {}
    except FileNotFoundError:
        cfg = {}

    return cfg.get("generation", {})


if __name__ == "__main__":
    # quick smoke test (won't run on import)
    p = get_pipe()
    print("Pipeline ready:", type(p))
