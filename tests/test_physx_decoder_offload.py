"""Memory adaptation keeps official sampling and scoped gradient behavior."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from asset2sim.experiments.physx_decoder_offload import install_decoder_offload


def test_offload_occurs_after_sampling_and_gradients_are_restored():
    events = []
    state = {"grad": True}

    @contextmanager
    def no_grad():
        old = state["grad"]
        state["grad"] = False
        try:
            yield
        finally:
            state["grad"] = old

    class Model:
        def __init__(self, name):
            self.name = name

        def to(self, device):
            events.append((self.name, device, state["grad"]))

    class Pipeline:
        def __init__(self):
            self.models = {
                name: Model(name)
                for name in ("image_cond_model", "slat_flow_model", "slat_decoder_mesh")
            }

        def sample_slat(self, payload):
            events.append(("sample", payload, state["grad"]))
            return payload

        def run_decoder(self, payload):
            latent = self.sample_slat(payload)
            events.append(("decode", latent, state["grad"]))
            return latent

    fake_torch = SimpleNamespace(
        no_grad=no_grad, cuda=SimpleNamespace(empty_cache=lambda: events.append(("empty_cache",)))
    )
    install_decoder_offload(Pipeline, fake_torch)
    payload = object()
    assert Pipeline().run_decoder(payload) is payload
    assert events == [
        ("sample", payload, False),
        ("image_cond_model", "cpu", False),
        ("slat_flow_model", "cpu", False),
        ("empty_cache",),
        ("decode", payload, False),
    ]
    assert state["grad"] is True


def test_offload_rejects_unexpected_upstream_model_layout():
    class Pipeline:
        def __init__(self):
            self.models = {}

        def sample_slat(self):
            raise AssertionError("must validate layout before sampling")

        def run_decoder(self):
            return self.sample_slat()

    fake_torch = SimpleNamespace(no_grad=lambda: None)
    install_decoder_offload(Pipeline, fake_torch)
    with pytest.raises(RuntimeError, match="missing pipeline models"):
        Pipeline().sample_slat()
