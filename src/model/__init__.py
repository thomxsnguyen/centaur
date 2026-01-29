"""TinySwallow LLM integration package for Centaur project.

This package exposes a minimal API for loading the model and running
policy-style inference via `get_pipe()` and `choose_action()`.
"""

from .model import get_pipe  # re-export for convenience
from .policy import choose_action  # type: ignore

__all__ = ["get_pipe", "choose_action"]
