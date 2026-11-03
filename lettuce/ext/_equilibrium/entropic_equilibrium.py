"""Positive tensor-product D2Q9 entropy-minimizing equilibrium."""

import torch

from ... import Equilibrium
from .._stencil import D2Q9

__all__ = ['EntropicEquilibrium']


class EntropicEquilibrium(Equilibrium):
    """Minimize ``sum(f_i log(f_i / w_i))`` at fixed mass and momentum.

    This closed-form D2Q9 equilibrium is positive for positive density and
    lattice velocity components strictly between -1 and 1. It preserves mass
    and momentum exactly; its higher moments differ from the quadratic
    equilibrium. It is isothermal, not a thermal/compressible gas model.

    Reference: Hosseini and Karlin (2023),
    https://arxiv.org/abs/2303.08163, equations (16)--(17).
    """

    def __call__(self, flow, rho=None, u=None):
        if not isinstance(flow.stencil, D2Q9):
            raise ValueError('EntropicEquilibrium supports D2Q9 only.')
        rho = flow.rho() if rho is None else rho
        u = flow.u() if u is None else u
        root = torch.sqrt(1 + 3 * u.square())
        # Algebraically equivalent to the exponential/product expression,
        # avoiding powers of (1-u) and retaining a well-behaved derivative at 0.
        probabilities = torch.stack(((2 / 3) * (2 - root),
                                     (2 * root - 1 + 3 * u) / 6,
                                     (2 * root - 1 - 3 * u) / 6))
        e = flow.torch_stencil.e.to(dtype=torch.long)
        indices = torch.where(e == 0, 0, torch.where(e == 1, 1, 2))
        return rho * probabilities[indices[:, 0], 0] * probabilities[indices[:, 1], 1]

    def native_available(self):
        return False

    def native_generator(self):
        return None
