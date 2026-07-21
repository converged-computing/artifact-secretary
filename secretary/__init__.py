"""Public API for the SDK-free core. Client pieces that need the Claude Agent
SDK live under `secretary.runner` and `secretary.cli` and are imported directly,
so `import secretary` never requires the SDK."""

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
from .tasks import ProfileTask
