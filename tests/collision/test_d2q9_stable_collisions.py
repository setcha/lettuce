"""Conservation, entropy, and autograd checks for D2Q9 solver extensions."""

import pytest
import torch

from lettuce import (Context, D2Q9, D2Q9MRTCollision, EntropicCollision,
                     EntropicEquilibrium, QuadraticEquilibrium)
from tests.conftest import TestFlow


def _flow(equilibrium=QuadraticEquilibrium):
    flow = TestFlow(Context(device='cpu', dtype=torch.float64, use_native=False),
                    resolution=[12, 12], reynolds_number=100,
                    mach_number=0.1, stencil=D2Q9())
    flow.equilibrium = equilibrium()
    return flow


@pytest.mark.parametrize('kind', ['mrt', 'entropic'])
def test_collision_conserves_mass_momentum_and_is_differentiable(kind):
    flow = _flow(EntropicEquilibrium if kind == 'entropic' else QuadraticEquilibrium)
    torch.manual_seed(17)
    f = (0.03 + torch.rand_like(flow.f)).requires_grad_()
    flow.f = f
    collision = (D2Q9MRTCollision(flow, tau=0.503, bulk_tau=1.0)
                 if kind == 'mrt' else EntropicCollision(tau=0.503))
    after = collision(flow)
    assert torch.allclose(after.sum(0), f.sum(0), atol=1e-12, rtol=1e-12)
    e = flow.torch_stencil.e
    assert torch.allclose(torch.einsum('qd,q...->d...', e, after),
                          torch.einsum('qd,q...->d...', e, f), atol=1e-12, rtol=1e-12)
    if kind == 'entropic':
        assert (after > 0).all()
        w = flow.torch_stencil.w[:, None, None]
        h_before = (f * (f / w).log()).sum(0)
        h_after = (after * (after / w).log()).sum(0)
        assert (h_after <= h_before + 1e-12).all()
    after.square().sum().backward()
    assert torch.isfinite(f.grad).all()


def test_entropic_implicit_derivative_matches_finite_difference():
    flow = _flow(EntropicEquilibrium)
    torch.manual_seed(4)
    base = 0.1 + torch.rand_like(flow.f)
    direction = torch.randn_like(base) * 0.02
    control = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)

    def value(x):
        flow.f = base + x * direction
        return EntropicCollision(tau=0.503, iterations=48)(flow).square().sum()

    loss = value(control)
    grad, = torch.autograd.grad(loss, control)
    fd = (value(control.detach() + 1e-4) - value(control.detach() - 1e-4)) / 2e-4
    assert torch.allclose(grad, fd, atol=2e-7, rtol=2e-5)


def test_entropic_equilibrium_is_positive_and_preserves_moments_at_high_speed():
    flow = _flow(EntropicEquilibrium)
    rho = torch.full((1, 12, 12), 1.02, dtype=torch.float64)
    u = torch.empty((2, 12, 12), dtype=torch.float64)
    u[0] = 0.42
    u[1] = -0.36
    eq = flow.equilibrium(flow, rho, u)
    assert (eq > 0).all()
    assert torch.allclose(eq.sum(0), rho[0], atol=1e-12, rtol=1e-12)
    assert torch.allclose(torch.einsum('qd,q...->d...', flow.torch_stencil.e, eq),
                          rho * u, atol=1e-12, rtol=1e-12)
