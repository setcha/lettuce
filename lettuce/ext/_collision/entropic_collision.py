"""Entropy-controlled D2Q9 relaxation with an implicit first derivative."""

import torch

from ... import Collision
from .._equilibrium import EntropicEquilibrium

__all__ = ['EntropicCollision']


class EntropicCollision(Collision):
    """Use the nontrivial entropy-balance root to restrict over-relaxation.

    ``f_post = f + alpha/(2*tau) * (f_eq-f)``. Alpha solves
    ``H(f + alpha*(f_eq-f)) = H(f)`` beyond alpha=1, subject to population
    positivity. If that root is outside the positive population polytope, use
    its interior endpoint instead, which still decreases H. No population
    clipping or post-hoc mass correction is performed.

    Root finding uses a fixed bisection horizon (32 iterations by default).
    The returned root carries the implicit first derivative of the entropy
    equation. Near equilibrium alpha=2 is used to avoid the singular 0/0
    derivative. Gradients are piecewise at positivity/root regime changes;
    higher derivatives are not supported by this implicit attachment.

    References: https://arxiv.org/abs/cond-mat/0005260 and
    https://arxiv.org/abs/1111.5994 (entropy roots need not always exist).
    """

    def __init__(self, tau, iterations=32):
        if tau <= 0.5 or iterations < 8:
            raise ValueError('tau must exceed 0.5 and iterations must be >= 8.')
        self.tau = tau
        self.iterations = iterations

    @staticmethod
    def _difference(f, delta, weights, alpha):
        # log(f_eq/w) is affine in conserved mass/momentum, so those terms
        # cancel. Compute the entropy relative to equilibrium to avoid losing
        # precision when H(f_post) and H(f) are nearly identical.
        eq = f + delta
        x = -delta / eq

        def relative(z):
            series = z.square() * (0.5 + z * (-1 / 6 + z * (1 / 12 + z * (-1 / 20 + z / 30))))
            direct = (1 + z) * torch.log1p(z) - z
            return torch.where(z.abs() < 1e-3, series, direct)

        return (eq * (relative((1 - alpha) * x) - relative(x))).sum(dim=0)

    def __call__(self, flow):
        if not isinstance(flow.equilibrium, EntropicEquilibrium):
            raise ValueError('EntropicCollision requires EntropicEquilibrium.')
        f = flow.f
        if not torch.isfinite(f).all() or (f <= 0).any():
            raise ValueError('EntropicCollision requires finite positive input populations.')
        eq = flow.equilibrium(flow)
        delta = eq - f
        weights = flow.torch_stencil.w.reshape((-1,) + (1,) * (f.ndim - 1))
        eps = torch.finfo(f.dtype).eps
        with torch.no_grad():
            near = (delta.abs() / f).amax(dim=0) < eps ** 0.4
            bounds = torch.where(delta < 0, -f / delta.clamp_max(-eps * eps), float('inf'))
            cap = bounds.amin(dim=0).clamp_max(4.0) * (1 - 16 * eps)
            low = torch.ones_like(cap)
            high = cap
            root_exists = self._difference(f, delta, weights, cap) > 0
            for _ in range(self.iterations):
                middle = (low + high) / 2
                positive = self._difference(f, delta, weights, middle) > 0
                high = torch.where(positive, middle, high)
                low = torch.where(positive, low, middle)
            alpha_value = torch.where(near, 2.0, torch.where(root_exists, (low + high) / 2, cap))

        # Differentiate G(alpha,f)=0 implicitly: d(alpha)=-dG/G_alpha.
        g = self._difference(f, delta, weights, alpha_value)
        derivative = (delta * torch.log1p((alpha_value - 1) * delta / eq)).sum(dim=0).detach()
        safe_derivative = torch.where(derivative.abs() > eps * eps, derivative, torch.ones_like(derivative))
        implicit = alpha_value - (g - g.detach()) / safe_derivative
        cap_with_gradient = torch.where(delta < 0, -f / delta.clamp_max(-eps * eps), float('inf')).amin(dim=0).clamp_max(4.0) * (1 - 16 * eps)
        alpha = torch.where(near, torch.full_like(g, 2.0), torch.where(root_exists, implicit, cap_with_gradient))
        self.last_alpha = alpha.detach()
        return f + (alpha / (2 * self.tau)) * delta

    def native_available(self):
        return False

    def native_generator(self, index):
        return None
