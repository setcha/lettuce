"""D2Q9 Zou--He pressure boundary conditions."""

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from ... import Boundary, Context, Flow
from .._stencil import D2Q9

__all__ = ['ZouHePressureBoundary']


class ZouHePressureBoundary(Boundary):
    """Pressure/density boundary for an axis-aligned D2Q9 domain face.

    ``direction`` is the outward normal of a west, east, south, or north
    domain face.  The prescribed ``pressure`` is in physical units and is
    converted by :class:`~lettuce.UnitConversion`; alternatively ``density``
    can prescribe lattice density directly.  Exactly one must be supplied.

    The boundary reconstructs only the populations entering the domain after
    streaming.  Populations that streamed from the interior are retained.
    The normal velocity is determined by the known populations and the target
    density.  ``tangential_velocity`` is a physical velocity; its default of
    zero is appropriate for a quiescent reservoir.

    This implementation supports D2Q9 only and applies to the interior of a
    cardinal domain face.  It deliberately excludes corner nodes because a
    corner belongs to two faces and needs an explicit corner treatment.

    References
    ----------
    Zou and He, *On pressure and velocity flow boundary conditions and
    bounceback for the lattice Boltzmann BGK model*,
    https://arxiv.org/abs/comp-gas/9611001 .
    """

    def __init__(self, flow: Flow, direction: Sequence[int], *,
                 pressure: Optional[Union[float, torch.Tensor]] = None,
                 density: Optional[Union[float, torch.Tensor]] = None,
                 tangential_velocity: Union[float, torch.Tensor] = 0.0):
        if not isinstance(flow.stencil, D2Q9):
            raise ValueError('ZouHePressureBoundary supports D2Q9 only.')
        if tuple(direction) not in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
            raise ValueError(
                'direction must be an axis-aligned outward normal such as '
                '[-1, 0] (west) or [0, 1] (north).')
        if (pressure is None) == (density is None):
            raise ValueError('Specify exactly one of pressure or density.')

        self.direction = tuple(int(component) for component in direction)
        self._axis = 0 if self.direction[0] else 1
        self._sign = self.direction[self._axis]
        self._rho = flow.context.convert_to_tensor(
            flow.units.convert_pressure_pu_to_density_lu(pressure)
            if pressure is not None else density)
        self._tangential_velocity = flow.context.convert_to_tensor(
            flow.units.convert_velocity_to_lu(tangential_velocity))

        e = np.asarray(flow.stencil.e)
        self._incoming = np.flatnonzero(
            np.matmul(e, np.asarray(self.direction)) < 0).tolist()

    def _index(self):
        edge = 0 if self._sign < 0 else -1
        if self._axis == 0:
            return edge, slice(1, -1)
        return slice(1, -1), edge

    def __call__(self, flow: Flow) -> torch.Tensor:
        """Reconstruct incoming populations using the Zou--He relations.

        See https://arxiv.org/abs/comp-gas/9611001 for the population
        reconstruction and pressure-driven Poiseuille validation.
        """
        f_old = flow.f
        f = f_old.clone()
        index = self._index()
        here = (slice(None), *index)
        values = f_old[here]
        rho = self._rho
        tangent = self._tangential_velocity

        # Population order is the canonical Lettuce D2Q9 order:
        # rest, east, north, west, south, north-east, north-west,
        # south-west, south-east.
        if self.direction == (-1, 0):  # west
            normal = 1 - (values[0] + values[2] + values[4]
                          + 2 * (values[3] + values[6] + values[7])) / rho
            f[1, *index] = values[3] + (2 / 3) * rho * normal
            f[5, *index] = (values[7] + (1 / 6) * rho * normal
                            + 0.5 * (values[4] - values[2])
                            + 0.5 * rho * tangent)
            f[8, *index] = (values[6] + (1 / 6) * rho * normal
                            + 0.5 * (values[2] - values[4])
                            - 0.5 * rho * tangent)
        elif self.direction == (1, 0):  # east
            normal = -1 + (values[0] + values[2] + values[4]
                           + 2 * (values[1] + values[5] + values[8])) / rho
            f[3, *index] = values[1] - (2 / 3) * rho * normal
            f[6, *index] = (values[8] - (1 / 6) * rho * normal
                            + 0.5 * (values[4] - values[2])
                            + 0.5 * rho * tangent)
            f[7, *index] = (values[5] - (1 / 6) * rho * normal
                            + 0.5 * (values[2] - values[4])
                            - 0.5 * rho * tangent)
        elif self.direction == (0, -1):  # south
            normal = 1 - (values[0] + values[1] + values[3]
                          + 2 * (values[4] + values[7] + values[8])) / rho
            f[2, *index] = values[4] + (2 / 3) * rho * normal
            f[5, *index] = (values[7] + (1 / 6) * rho * normal
                            + 0.5 * (values[3] - values[1])
                            + 0.5 * rho * tangent)
            f[6, *index] = (values[8] + (1 / 6) * rho * normal
                            + 0.5 * (values[1] - values[3])
                            - 0.5 * rho * tangent)
        else:  # north, direction == (0, 1)
            normal = -1 + (values[0] + values[1] + values[3]
                           + 2 * (values[2] + values[5] + values[6])) / rho
            f[4, *index] = values[2] - (2 / 3) * rho * normal
            f[7, *index] = (values[5] - (1 / 6) * rho * normal
                            + 0.5 * (values[1] - values[3])
                            - 0.5 * rho * tangent)
            f[8, *index] = (values[6] - (1 / 6) * rho * normal
                            + 0.5 * (values[3] - values[1])
                            + 0.5 * rho * tangent)
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
