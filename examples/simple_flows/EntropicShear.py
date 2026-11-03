"""A periodic D2Q9 shear wave using entropy-controlled relaxation.

Run from the checkout: uv run --extra cpu python examples/simple_flows/EntropicShear.py
There are no pressure or wall boundaries; this isolates collision stability.
"""

import numpy as np
import torch
from lettuce import (Context, D2Q9, EntropicCollision, EntropicEquilibrium,
                     Simulation, TaylorGreenVortex)


class PeriodicShear(TaylorGreenVortex):
    def initial_pu(self):
        x = np.arange(self.resolution[0])[:, None]
        u = np.zeros([2] + self.resolution)
        u[0] = 1.0
        u[1] = 0.02 * np.sin(2 * np.pi * x / self.resolution[0])
        return 0.0, u


if __name__ == '__main__':
    context = Context(device='cpu', dtype=torch.float64, use_native=False)
    flow = PeriodicShear(context, resolution=32, reynolds_number=100,
                         mach_number=0.15, stencil=D2Q9(),
                         equilibrium=EntropicEquilibrium())
    Simulation(flow, EntropicCollision(flow.units.relaxation_parameter_lu), [])(100)
    print('minimum population:', flow.f.min().item())
