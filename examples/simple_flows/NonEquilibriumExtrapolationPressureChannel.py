"""Minimal D2Q9 pressure channel with non-equilibrium extrapolation.

Run with ``uv run --extra cpu python
examples/simple_flows/NonEquilibriumExtrapolationPressureChannel.py``.
"""

import numpy as np
import torch

from lettuce import (BGKCollision, BounceBackBoundary, Context, D2Q9,
                     NonEquilibriumExtrapolationPressureBoundary, Simulation,
                     UnitConversion)
from lettuce.ext import ExtFlow


class PressureDrivenChannel(ExtFlow):
    def make_resolution(self, resolution, stencil=None):
        return list(resolution)

    def make_units(self, reynolds_number, mach_number, resolution):
        return UnitConversion(reynolds_number, mach_number,
                              characteristic_length_lu=resolution[1] - 2)

    def initial_pu(self):
        zeros = np.zeros(self.resolution)
        return zeros[None, ...], np.stack([zeros, zeros])

    @property
    def post_boundaries(self):
        walls = self.context.zero_tensor(self.resolution, dtype=torch.bool)
        walls[:, [0, -1]] = True
        return [
            NonEquilibriumExtrapolationPressureBoundary(self, [-1, 0], density=1.001),
            NonEquilibriumExtrapolationPressureBoundary(self, [1, 0], density=1.0),
            BounceBackBoundary(walls),
        ]


if __name__ == '__main__':
    flow = PressureDrivenChannel(
        Context(device='cpu', dtype=torch.float64, use_native=False),
        resolution=[96, 24], reynolds_number=20, mach_number=0.05,
        stencil=D2Q9())
    Simulation(flow, BGKCollision(flow.units.relaxation_parameter_lu), reporter=[])(2000)
    print('mean channel velocity:', flow.u()[0, 2:-2, 1:-1].mean().item())
