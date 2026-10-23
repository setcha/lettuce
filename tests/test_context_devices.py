import pytest
import torch

from lettuce import Context


def test_context_auto_selects_available_backend():
    context = Context(device="auto", use_native=False)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    if (expected == "cpu" and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        expected = "mps"
    assert context.device.type == expected
    assert context.use_native is False


def test_context_keeps_cpu_and_cuda_defaults():
    assert Context(device="cpu").use_native is False
    if torch.cuda.is_available():
        context = Context(device="cuda")
        assert context.device.type == "cuda"
        assert context.use_native is True


def test_context_mps_device_when_available():
    if (not hasattr(torch.backends, "mps")
            or not torch.backends.mps.is_available()):
        pytest.skip("MPS is not available on this machine.")
    context = Context(device="mps")
    assert context.device.type == "mps"
    assert context.use_native is False
    assert context.zero_tensor([1]).device.type == "mps"


def test_context_rejects_native_on_non_cuda_device():
    with pytest.raises(AssertionError, match="cuda_native"):
        Context(device="cpu", use_native=True)
