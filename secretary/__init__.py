"""Public API for the SDK-free core. Client pieces that need the Claude Agent
SDK live under `secretary.runner` and `secretary.cli` and are imported directly,
so `import secretary` never requires the SDK."""
from .inspection import Target, OutsideTargetError, Inspector
from .model import (
    Artifact, Capability, Provenance, Variant, derive_capability,
    ManifestLookup, LookupEntry, Reproduce, LOOKUP_SCHEMA_VERSION,
)
from .framework import (
    Task, ToolSpec, AgentRunner, ConfirmFn, RunOutcome, run_task, default_confirm,
    inspection_tools,
)
from .tasks import ProfileTask
