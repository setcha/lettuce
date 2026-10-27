"""Checks for the D2Q9 non-equilibrium extrapolation pressure boundary."""

import numpy as np
import pytest
import torch

from lettuce import (BGKCollision, BounceBackBoundary, Context, D2Q9,
                     NonEquilibriumExtrapolationPressureBoundary, Simulation,
                     UnitConversion)
from lettuce.ext import ExtFlow


def _macroscopic(f, stencil):
    rho = f.sum(dim=0)
    e = torch.as_tensor(stencil.e, dtype=f.dtype, device=f.device)
    return rho, torch.einsum('qa,q...->a...', e, f) / rho


class _BoundaryFlow(ExtFlow):
    def make_resolution(self, resolution, stencil=None):
        return list(resolution)

    def make_units(self, reynolds_number, mach_number, resolution):
        return UnitConversion(reynolds_number, mach_number,
                              characteristic_length_lu=resolution[0])

    def initial_pu(self):
        zeros = np.zeros(self.resolution)
        return zeros[None, ...], np.stack([zeros, zeros])


@pytest.mark.parametrize('direction', [(-1, 0), (1, 0), (0, -1), (0, 1)])
def test_non_equilibrium_extrapolation_sets_target_density_and_preserves_neighbor_velocity(direction):
    context = Context(device='cpu', dtype=torch.float64, use_native=False)
    flow = _BoundaryFlow(context, resolution=[9, 8], reynolds_number=20,
                         mach_number=0.05, stencil=D2Q9())
    boundary = NonEquilibriumExtrapolationPressureBoundary(flow, direction, density=1.002)
    torch.manual_seed(9)
    flow.f = 0.02 + torch.rand_like(flow.f)
    after = boundary(flow)
    index = boundary._index()
    neighbor = boundary._neighbor()
    rho, velocity = _macroscopic(after[(slice(None), *index)], flow.stencil)
    _, neighbor_velocity = _macroscopic(flow.f[(slice(None), *neighbor)], flow.stencil)
    assert torch.allclose(rho, torch.full_like(rho, 1.002), rtol=1e-12, atol=1e-12)
    assert torch.allclose(velocity, neighbor_velocity, rtol=1e-12, atol=1e-12)


def test_non_equilibrium_extrapolation_is_differentiable_and_converts_pressure():
    context = Context(device='cpu', dtype=torch.float64, use_native=False)
    flow = _BoundaryFlow(context, resolution=[9, 8], reynolds_number=20,
                         mach_number=0.05, stencil=D2Q9())
    flow.f = (0.02 + torch.rand_like(flow.f)).requires_grad_()
    boundary = NonEquilibriumExtrapolationPressureBoundary(flow, [-1, 0], pressure=0.02)
    assert torch.allclose(
        boundary._rho,
        context.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(0.02)),
    )
    boundary(flow).square().sum().backward()
    assert flow.f.grad is not None
    assert torch.isfinite(flow.f.grad).all()


class _PressureDrivenChannel(_BoundaryFlow):
    @property
    def post_boundaries(self):
        walls = self.context.zero_tensor(self.resolution, dtype=torch.bool)
        walls[:, [0, -1]] = True
        return [
            NonEquilibriumExtrapolationPressureBoundary(self, [-1, 0], density=1.001),
            NonEquilibriumExtrapolationPressureBoundary(self, [1, 0], density=1.0),
            BounceBackBoundary(walls),
        ]


def test_non_equilibrium_extrapolation_drives_a_finite_channel_flow():
    context = Context(device='cpu', dtype=torch.float64, use_native=False)
    flow = _PressureDrivenChannel(context, resolution=[48, 14], reynolds_number=20,
                                  mach_number=0.05, stencil=D2Q9())
    Simulation(flow, BGKCollision(flow.units.relaxation_parameter_lu), reporter=[])(800)
    rho, velocity = _macroscopic(flow.f, flow.stencil)
    assert torch.isfinite(flow.f).all()
    assert velocity[0, 2:-2, 1:-1].mean() > 0
    assert torch.allclose(rho[0, 1:-1], torch.full_like(rho[0, 1:-1], 1.001), rtol=2e-5, atol=2e-5)
