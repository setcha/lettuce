"""D2Q9 MRT with explicit shear, bulk, and kinetic relaxation times."""

import torch

from ... import Collision
from .._stencil import D2Q9
from ...util.moments import D2Q9Lallemand

__all__ = ['D2Q9MRTCollision']


class D2Q9MRTCollision(Collision):
    """Relax D2Q9 moments using a Lallemand--Luo basis.

    ``tau`` fixes shear viscosity. ``bulk_tau`` defaults to the same value;
    ``kinetic_tau`` damps the three non-hydrodynamic modes independently.
    Mass and momentum have zero relaxation rates. Equilibrium moments are
    transformed from ``flow.equilibrium`` so they remain consistent for both
    quadratic and entropic equilibria and non-unit density.

    Reference: Lallemand and Luo (2000),
    https://doi.org/10.1103/PhysRevE.61.6546.
    """

    def __init__(self, flow, tau=None, bulk_tau=None, kinetic_tau=1.0):
        if not isinstance(flow.stencil, D2Q9):
            raise ValueError('D2Q9MRTCollision supports D2Q9 only.')
        self.tau = flow.units.relaxation_parameter_lu if tau is None else tau
        self.bulk_tau = self.tau if bulk_tau is None else bulk_tau
        self.kinetic_tau = kinetic_tau
        if min(self.tau, self.bulk_tau, self.kinetic_tau) <= 0.5:
            raise ValueError('MRT relaxation times must exceed 0.5.')
        self.transform = D2Q9Lallemand(flow.stencil, flow.context)
        self.rates = flow.context.convert_to_tensor(
            [0, 0, 0, 1 / self.tau, 1 / self.tau, 1 / self.bulk_tau,
             1 / self.kinetic_tau, 1 / self.kinetic_tau, 1 / self.kinetic_tau])

    def __call__(self, flow):
        moments = self.transform.transform(flow.f)
        equilibrium = self.transform.transform(flow.equilibrium(flow))
        relaxed = moments - torch.einsum('q,q...->q...', self.rates,
                                         moments - equilibrium)
        return self.transform.inverse_transform(relaxed)

    def native_available(self):
        return False

    def native_generator(self, index):
        return None
