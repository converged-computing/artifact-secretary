"""The model output cap is a separate thing from the source budget.

make_runner never sets one, so a long record_shape response can trip the Strands
max tokens exception. These pin that the two limits stay distinct and that the
wrapper degrades safely when Strands is missing, as it is in CI.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import secretary.runner as runner_mod
from secretary.runner import DEFAULT_MODEL_MAX_TOKENS, make_runner_with_output_cap


class _FakeStrandsRunner:
    def __init__(self):
        self.model = "us.anthropic.claude-sonnet-5"
        self.region = "us-east-1"
        self.calls = 0

    def _model(self):
        self.calls += 1
        return object()  # a stand-in BedrockModel


def test_non_aws_backend_is_returned_untouched(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(runner_mod, "make_runner", lambda b, m: sentinel)
    assert make_runner_with_output_cap("claude", None, 8192) is sentinel


def test_no_cap_requested_is_a_passthrough(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(runner_mod, "make_runner", lambda b, m: sentinel)
    assert make_runner_with_output_cap("aws", "x", None) is sentinel


def test_missing_strands_degrades_instead_of_failing(monkeypatch):
    """Strands is not installed in CI so the wrapper must not explode."""
    fake = _FakeStrandsRunner()
    monkeypatch.setattr(runner_mod, "make_runner", lambda b, m: fake)
    monkeypatch.setitem(sys.modules, "strands.models", None)
    out = make_runner_with_output_cap("aws", fake.model, 8192)
    assert out is fake  # returned as-is, no crash


def test_cap_is_applied_when_strands_available(monkeypatch):
    """With strands importable, _model() must be wrapped to carry max_tokens."""
    import types

    captured = {}

    class _BedrockModel:
        def __init__(self, **kw):
            captured.update(kw)

    mod = types.ModuleType("strands.models")
    mod.BedrockModel = _BedrockModel
    monkeypatch.setitem(sys.modules, "strands.models", mod)

    fake = _FakeStrandsRunner()
    monkeypatch.setattr(runner_mod, "make_runner", lambda b, m: fake)

    out = make_runner_with_output_cap("aws", fake.model, 12345)
    out._model()  # what StrandsRunner does per run

    assert captured["max_tokens"] == 12345, captured
    assert captured["model_id"] == fake.model
    assert captured["region_name"] == fake.region


def test_default_cap_is_generous_enough_to_be_useful():
    # the provider default is what broke so ours has to be clearly bigger
    assert DEFAULT_MODEL_MAX_TOKENS >= 8192
