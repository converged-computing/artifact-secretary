"""Model backends. Each submodule imports its own SDK at module load, so nothing
here is imported until you actually ask for a specific runner. This is what lets
`--backend aws` run with only Strands installed, without dragging in the Claude
or Gemini SDKs.

`from secretary.runner import SDKRunner` still works (resolved lazily below);
importing the submodule directly (`secretary.runner.sdk`) is equivalent.
"""

_RUNNERS = {
    "SDKRunner": ".sdk",
    "ADKRunner": ".adk",
    "StrandsRunner": ".strands",
}


def __getattr__(name):
    module = _RUNNERS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(_RUNNERS)
