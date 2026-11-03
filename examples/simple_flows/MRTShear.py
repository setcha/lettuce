"""Periodic D2Q9 MRT example with independent kinetic-mode damping."""

import torch
from lettuce import Context, D2Q9, D2Q9MRTCollision, Simulation, TaylorGreenVortex


if __name__ == '__main__':
    context = Context(device='cpu', dtype=torch.float64, use_native=False)
    flow = TaylorGreenVortex(context, resolution=32, reynolds_number=100,
                            mach_number=0.1, stencil=D2Q9())
    Simulation(flow, D2Q9MRTCollision(flow, bulk_tau=1.0), [])(100)
    print('minimum population:', flow.f.min().item())
