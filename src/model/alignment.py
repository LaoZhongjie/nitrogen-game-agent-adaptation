"""Backward-compatibility shim — action encoding has moved to action_encoder.py.

The old vocabulary-based action alignment is no longer used. This module
re-exports the new ``GamepadActionEncoder`` for any code that still
imports from ``src.model.alignment``.
"""

from src.model.action_encoder import ActionEncoder, GamepadActionEncoder

__all__ = ["ActionEncoder", "GamepadActionEncoder"]
