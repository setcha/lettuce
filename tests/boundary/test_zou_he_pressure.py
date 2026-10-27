"""Focused checks for the D2Q9 Zou--He pressure boundary."""

import pytest
import numpy as np
import torch

from lettuce import (BGKCollision, BounceBackBoundary, Context, D2Q9,
                     Simulation, UnitConversion, ZouHePressureBoundary)
from lettuce.ext import ExtFlow


def _macroscopic(f, stencil):
    rho = f.sum(dim=0)
    e = torch.as_tensor(stencil.e, dtype=f.dtype, device=f.device)
    u = torch.einsum('qa,q...->a...', e, f) / rho
    return rho, u


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
def test_zou_he_pressure_reconstructs_only_unknown_populations(direction):
    context = Context(device='cpu', dtype=torch.float64, use_native=False)
    flow = _BoundaryFlow(context, resolution=[9, 8], reynolds_number=20,
                         mach_number=0.05, stencil=D2Q9())
    target_density = 1.002
    target_tangent_lu = 0.01
    tangent_pu = flow.units.convert_velocity_to_pu(target_tangent_lu)
    boundary = ZouHePressureBoundary(
        flow, direction, density=target_density,
        tangential_velocity=tangent_pu)

    torch.manual_seed(3)
    flow.f = 0.02 + torch.rand_like(flow.f)
    before = flow.f.clone()
    after = boundary(flow)
    index = boundary._index()
    incoming = boundary._incoming
    known = [q for q in range(flow.stencil.q) if q not in incoming]

    assert torch.allclose(after[known, *index], before[known, *index])
    rho, velocity = _macroscopic(after[(slice(None), *index)], flow.stencil)
    assert torch.allclose(rho, torch.full_like(rho, target_density),
                          rtol=1e-12, atol=1e-12)
    tangent_axis = 1 - boundary._axis
    assert torch.allclose(velocity[tangent_axis],
                          torch.full_like(rho, target_tangent_lu),
                          rtol=1e-12, atol=1e-12)


def test_zou_he_pressure_converts_physical_pressure_to_target_density():
    context = Context(device='cpu', dtype=torch.float64, use_native=False)
    flow = _BoundaryFlow(context, resolution=[9, 8], reynolds_number=20,
                         mach_number=0.05, stencil=D2Q9())
    pressure_pu = 0.02
    boundary = ZouHePressureBoundary(flow, [-1, 0], pressure=pressure_pu)
    assert torch.allclose(
        boundary._rho,
        context.convert_to_tensor(
            flow.units.convert_pressure_pu_to_density_lu(pressure_pu)),
        rtol=1e-12, atol=1e-12)


def test_zou_he_pressure_reconstruction_is_differentiable():
    context = Context(device='cpu', dtype=torch.float64, use_native=False)
    flow = _BoundaryFlow(context, resolution=[9, 8], reynolds_number=20,
                         mach_number=0.05, stencil=D2Q9())
    flow.f = (0.02 + torch.rand_like(flow.f)).requires_grad_()
    boundary = ZouHePressureBoundary(flow, [-1, 0], density=1.001)

    boundary(flow).square().sum().backward()

    assert flow.f.grad is not None
    assert torch.isfinite(flow.f.grad).all()


class _PressureDrivenChannel(ExtFlow):
    """Small contribution-level smoke flow; SALAD owns convergence studies."""

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
            ZouHePressureBoundary(self, [-1, 0], density=1.001),
            ZouHePressureBoundary(self, [1, 0], density=1.0),
            BounceBackBoundary(walls),
        ]


def test_zou_he_pressure_drives_a_finite_channel_flow():
    context = Context(device='cpu', dtype=torch.float64, use_native=False)
    flow = _PressureDrivenChannel(context, resolution=[48, 14],
                                  reynolds_number=20, mach_number=0.05,
                                  stencil=D2Q9())
    simulation = Simulation(
        flow, BGKCollision(flow.units.relaxation_parameter_lu), reporter=[])
    simulation(800)

    rho, velocity = _macroscopic(flow.f, flow.stencil)
    assert torch.isfinite(flow.f).all()
    assert velocity[0, 2:-2, 1:-1].mean() > 0
    assert torch.allclose(rho[0, 1:-1], torch.full_like(rho[0, 1:-1], 1.001),
                          rtol=2e-5, atol=2e-5)
    assert torch.allclose(rho[-1, 1:-1], torch.ones_like(rho[-1, 1:-1]),
                          rtol=2e-5, atol=2e-5)
