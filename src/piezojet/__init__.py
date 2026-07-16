"""PiezoJet: equivariant crystal piezoelectric response learning.

The top-level package stays lightweight so data acquisition CLIs do not import
PyTorch Geometric, e3nn, and SciPy before they need a model.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PiezoJet", "ResponsePotential"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .model import PiezoJet, ResponsePotential

        return {"PiezoJet": PiezoJet, "ResponsePotential": ResponsePotential}[name]
    raise AttributeError(name)
