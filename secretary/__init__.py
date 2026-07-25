"""Public API for the SDK-free core. Client pieces that need the Claude Agent
SDK live under `secretary.runner` and `secretary.cli` and are imported directly,
so `import secretary` never requires the SDK."""

try:
    from .framework import (
        AgentRunner,
        ConfirmFn,
        RunOutcome,
        Task,
        ToolSpec,
        default_confirm,
        inspection_tools,
        run_task,
    )
except ModuleNotFoundError:  # pragma: no cover
    # The agent framework re-exports from `behalf`, which is NOT copied into the
    # container. The in-container probe imports secretary.container.probe with only
    # inspection + pyelftools on the path, so importing this package must not
    # hard-require behalf. When behalf is absent these names are simply unbound.
    pass
from .inspection import Inspector, OutsideTargetError, Target
from .model import (
    LOOKUP_SCHEMA_VERSION,
    Artifact,
    Capability,
    LookupEntry,
    ManifestLookup,
    Provenance,
    Reproduce,
    Variant,
    derive_capability,
)
try:
    from .tasks import ProfileTask
except ModuleNotFoundError:  # pragma: no cover
    pass  # ProfileTask needs behalf; not required for the in-container probe
