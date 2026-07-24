"""Build the runner, with a say over how much the model may generate.

Two limits get confused. The ShapeTask budget bounds source served into context.
The model max_tokens bounds what it generates in one response, and blowing that is
what raises the Strands max tokens error. make_runner never sets the second, so
this wraps it without forking behalf.
"""

from __future__ import annotations

from typing import Optional

from behalf import make_runner

# bedrock takes a lot more than the provider default, and a record_shape call
# carrying many evidenced assertions is the case that needs it
DEFAULT_MODEL_MAX_TOKENS = 16384


def make_runner_with_output_cap(
    backend: str,
    model: Optional[str] = None,
    model_max_tokens: Optional[int] = None,
):
    """make_runner with the output token cap applied.

    Only the aws backend is touched since that is the one whose default bites.
    If the override cannot be applied we hand back the plain runner rather than
    failing the whole run.
    """
    runner = make_runner(backend, model)
    if backend != "aws" or not model_max_tokens:
        return runner

    try:
        from strands.models import BedrockModel
    except ImportError:
        return runner  # no strands so nothing to adjust

    # the runner builds its model per run, so wrap that and keep whatever model
    # and region it already resolved
    original = runner._model

    def _model_with_cap():
        base = original()
        if base is None:
            # no model set means behalf falls back to the default, so build one
            # here to make the cap still apply
            return BedrockModel(region_name=runner.region, max_tokens=model_max_tokens)
        return BedrockModel(
            model_id=runner.model,
            region_name=runner.region,
            max_tokens=model_max_tokens,
        )

    runner._model = _model_with_cap
    return runner
