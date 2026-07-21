"""A tiny framework for the pattern we keep wanting: give a Claude agent a
SCOPED task that first gathers intent from the user, freezes it into a
reproducible manifest, then executes toward one specific action with a fixed,
typed toolset.

    elicit (adaptive conversation, seeded)  ->  manifest (frozen, reviewable)
                                            ->  execute (agent + fixed tools)

The core knows nothing about containers or fleetq. A Task plugin supplies the
three things that vary: what to ask the user, what tools the agent may use, and
what a valid result is. Submitting a job, profiling a container, etc. are all
just Tasks — some read-only, some with an action tool that gates on user
confirmation, some chained into a pipeline.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol


# A tool the agent can call. `kind` distinguishes read (call freely) from action
# (mutates something -> may require the user to confirm the exact arguments).
@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Awaitable[dict]]
    kind: str = "read"  # "read" | "action"
    confirm: bool = False  # if action: pause for user approval before running


# confirm_fn(tool_name, args) -> bool. Supplied by the runtime (CLI prompt, test
# stub, or auto-approve). Only ever called for action tools with confirm=True.
ConfirmFn = Callable[[str, dict], bool]


class Task(ABC):
    """A scoped capability. Subclasses fill in the three varying pieces."""

    name: str = "task"

    # Whether the frozen setup manifest must be approved by the user before
    # execution. Customizable per task: profiling doesn't need it; submitting a
    # job does. (Action tools have their OWN per-call confirmation regardless.)
    requires_setup_approval: bool = False

    def setup_system_prompt(self) -> str:
        """Seeds the elicitation conversation: tells Claude what it already
        knows (schemas, fixed metadata) and what it must learn from the user.
        The user talks about goals; Claude handles structure."""
        return "Gather what you need from the user, then finalize a setup manifest."

    @abstractmethod
    def manifest_schema(self) -> dict:
        """JSON-ish schema the elicitation must produce and freeze."""

    def build_tools(self, manifest: dict) -> list[ToolSpec]:
        """The fixed toolset for the default single-run execute(). Iterating
        tasks (e.g. profile a catalog) override execute() and build tools
        per item instead, so they leave this as []."""
        return []

    @abstractmethod
    def execute_system_prompt(self, manifest: dict) -> str:
        """Instruction for the execution agent."""

    async def execute(
        self, runner: "AgentRunner", manifest: dict, confirm_fn: "ConfirmFn"
    ) -> Any:
        """Run the task. Default: a single agent run over the built tools. Tasks
        that iterate (e.g. profile a catalog) override this and call
        runner.run_agent() once per item."""
        tools = self.build_tools(manifest)
        return await runner.run_agent(
            system_prompt=self.execute_system_prompt(manifest),
            user_prompt=manifest.get("goal", "Complete the task."),
            tools=tools,
            confirm_fn=confirm_fn,
        )

    def validate_result(self, result: Any) -> None:
        """Raise if the produced result is not acceptable. Override per task."""
        return None


class AgentRunner(Protocol):
    """How the framework talks to a model. The real one wraps the Claude Agent
    SDK; tests inject a fake. This seam is what lets the whole flow be tested
    without a key."""

    async def converse(self, task: "Task") -> dict:
        """Run the seeded, adaptive setup conversation; return the manifest."""
        ...

    async def run_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolSpec],
        confirm_fn: "ConfirmFn",
    ) -> Any:
        """One agent run with a fixed toolset; action tools gate on confirm_fn."""
        ...


@dataclass
class RunOutcome:
    manifest: dict
    result: Any
    approved: bool = True


def default_confirm(tool_name: str, args: dict) -> bool:
    """CLI confirmation: show the exact call and ask. Safe default for actions."""
    print(f"\n[confirm] about to run action '{tool_name}' with:")
    print(json.dumps(args, indent=2))
    return input("proceed? [y/N] ").strip().lower() in ("y", "yes")


async def run_task(
    task: Task,
    runner: AgentRunner,
    manifest: Optional[dict] = None,
    confirm_fn: ConfirmFn = default_confirm,
    approve_fn: Optional[Callable[[dict], bool]] = None,
) -> RunOutcome:
    """The one general entrypoint every task flows through.

    - manifest given  -> skip the conversation (reproducible / batch replay).
    - manifest None   -> converse to produce it.
    Then optionally get setup approval (per task), then execute with the fixed
    toolset; action tools gate on confirm_fn.
    """
    if manifest is None:
        manifest = await runner.converse(task)

    approved = True
    if task.requires_setup_approval:
        approver = approve_fn or (lambda m: default_confirm("finalize-setup", m))
        approved = approver(manifest)
        if not approved:
            return RunOutcome(manifest=manifest, result=None, approved=False)

    result = await task.execute(runner, manifest, confirm_fn)
    task.validate_result(result)
    return RunOutcome(manifest=manifest, result=result, approved=approved)
