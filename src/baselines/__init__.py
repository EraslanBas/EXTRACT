"""Baseline factor models behind one interface.

    from baselines import get, available
    model = get("linear_ica")(n_factors=20).fit(X, perturbations, contexts)

Implemented: ``pca``, ``linear_ica`` (FastICA), ``context_jd`` (joint
diagonalization of per-context covariances).
Adapters awaiting wiring: ``mofa``, ``muvi``, ``svae_plus`` -- these raise a
message naming what they need.
"""

from . import external, linear_ica  # noqa: F401  (populate the registry)
from .joint_diagonalization import joint_diagonalize, off_diagonal_energy
from .registry import FactorModel, available, get, register

__all__ = [
    "get",
    "available",
    "register",
    "FactorModel",
    "joint_diagonalize",
    "off_diagonal_energy",
]
