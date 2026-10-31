"""Non-equilibrium extrapolation pressure boundaries for D2Q9."""

from typing import List, Optional, Sequence, Union

import torch

from ... import Boundary, Context, Flow
from .._stencil import D2Q9

__all__ = ['NonEquilibriumExtrapolationPressureBoundary']


class NonEquilibriumExtrapolationPressureBoundary(Boundary):
    """Pressure/density boundary with adjacent-node non-equilibrium extrapolation.

    ``direction`` is the outward normal of a west, east, south, or north
    D2Q9 domain face. Specify exactly one of physical ``pressure`` or lattice
    ``density``. The prescribed density is used for the boundary equilibrium;
    velocity and the non-equilibrium population are copied from the adjacent
    interior node. This avoids prescribing a tangential velocity, which is
    useful where an oblique jet reaches a pressure outlet.

    Only non-corner cardinal-face nodes are supported. As with other local
    open boundaries, a pressure-flow calibration remains the caller's
    responsibility.

    References
    ----------
    The non-equilibrium extrapolation construction is due to Shi, *Non-
    equilibrium extrapolation method for velocity and pressure boundary
    conditions in the lattice Boltzmann method*, doi:10.1088/1009-1963/11/4/310.
    For the D2Q9 BGK pressure-boundary context used here, see Zou and He,
    https://arxiv.org/abs/comp-gas/9611001 .
    """

    def __init__(self, flow: Flow, direction: Sequence[int], *,
                 pressure: Optional[Union[float, torch.Tensor]] = None,
                 density: Optional[Union[float, torch.Tensor]] = None):
        if not isinstance(flow.stencil, D2Q9):
            raise ValueError('NonEquilibriumExtrapolationPressureBoundary supports D2Q9 only.')
        if tuple(direction) not in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
            raise ValueError('direction must be an axis-aligned outward normal.')
        if (pressure is None) == (density is None):
            raise ValueError('Specify exactly one of pressure or density.')
        self.direction = tuple(int(component) for component in direction)
        self._axis = 0 if self.direction[0] else 1
        self._sign = self.direction[self._axis]
        self._rho = flow.context.convert_to_tensor(
            flow.units.convert_pressure_pu_to_density_lu(pressure)
            if pressure is not None else density)
        e = torch.as_tensor(flow.stencil.e)
        self._incoming = torch.where(torch.matmul(e, torch.as_tensor(self.direction)) < 0)[0].tolist()

    def _index(self):
        edge = 0 if self._sign < 0 else -1
        if self._axis == 0:
            return edge, slice(1, -1)
        return slice(1, -1), edge

    def _neighbor(self):
        if self.direction == (-1, 0):
            return 1, slice(1, -1)
        if self.direction == (1, 0):
            return -2, slice(1, -1)
        if self.direction == (0, -1):
            return slice(1, -1), 1
        return slice(1, -1), -2

    def __call__(self, flow: Flow) -> torch.Tensor:
        """Set target equilibrium density and extrapolate non-equilibrium.

        The relation is ``f_b = f_eq(rho_b, u_i) + f_i - f_eq(rho_i, u_i)``;
        see the class references for the derivation and boundary context.
        """
        f_old = flow.f
        f = f_old.clone()
        index = self._index()
        neighbor = f_old[(slice(None), *self._neighbor())]
        rho_neighbor = neighbor.sum(dim=0)
        velocity_neighbor = torch.einsum('qd,q...->d...', flow.torch_stencil.e, neighbor) / rho_neighbor
        feq_neighbor = flow.equilibrium(flow, rho_neighbor, velocity_neighbor)
        rho_target = torch.ones_like(rho_neighbor) * self._rho
        feq_target = flow.equilibrium(flow, rho_target, velocity_neighbor)
        f[(slice(None), *index)] = feq_target + neighbor - feq_neighbor
        return f

    def make_no_collision_mask(self, shape: List[int], context: Context
                               ) -> Optional[torch.Tensor]:
        mask = context.zero_tensor(shape, dtype=torch.bool)
        mask[self._index()] = True
        return mask

    def make_no_streaming_mask(self, shape: List[int], context: Context
                               ) -> Optional[torch.Tensor]:
        mask = context.zero_tensor(shape, dtype=torch.bool)
        mask[self._incoming, *self._index()] = True
        return mask

    def native_available(self) -> bool:
        """The PyTorch implementation has no cuda_native generator yet."""
        return False

    def native_generator(self, index: int):
        return None
