"""Our LocDiT sampler takes the same steps as the inference engine's (nanovllm_voxcpm solve_euler)."""

import importlib.util
import os

import pytest
import torch

from arabic_voxcpm.distill import sampler

ENGINE = os.environ.get("ENGINE", "/workspace/voxcpm2-arabic-dialects-turbo")


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(8, 8) / 4)

    def forward(self, x, mu, t, cond, dt):
        return torch.tanh(x + (mu @ self.w)[:, :, None] + t[:, None, None]) + 0.3 * cond


def test_matches_engine_solver():
    path = os.path.join(ENGINE, "nanovllm_voxcpm/models/voxcpm2/model_utils_cfm.py")
    if not os.path.exists(path):
        pytest.skip("inference library not available")
    spec = importlib.util.spec_from_file_location("engine_cfm", path)  # the module alone (the package needs flash-attn)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    torch.manual_seed(0)
    est = Toy()
    z, cond, mu = torch.randn(5, 8, 4), torch.randn(5, 8, 4), torch.randn(5, 8)
    ours = sampler.solve(est, z, mu, cond, cfg_value=2.0, steps=10)

    class Ops:
        estimator = est

        @staticmethod
        def optimized_scale(p, n):
            return E.compute_optimized_scale(p, n)

    inputs = E.EulerSolverInputs(x=z, t_span=E.build_cfm_t_span(10), mu=mu, cond=cond, cfg_value=torch.full((5,), 2.0))
    theirs = E.solve_euler(inputs, E.EulerSolverConfig(in_channels=8, mean_mode=False), Ops())
    torch.testing.assert_close(ours, theirs.float(), rtol=1e-5, atol=1e-5)
