"""M3-Agent model implementation package."""

from . import sitecustomize  # noqa: F401 - ensures startup tweaks run on import
from .m3_agent import M3Agent

__all__ = ["M3Agent"]
