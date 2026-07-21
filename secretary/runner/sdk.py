"""Claude backend: an AgentRunner backed by the Claude Agent SDK.

The SDK provides the agent loop, tool dispatch, and permissions. This module
adapts our ToolSpecs into SDK tools (adding the action-confirmation gate) and
runs them two ways:

    run_agent()  one execution pass over a fixed toolset
    converse()   interactive setup that ends when the model calls finalize_setup

Live use needs ANTHROPIC_API_KEY and the Claude Code CLI on PATH.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    query,
    tool,
)

from ..framework.core import AgentRunner, ConfirmFn, Task, ToolSpec


def _say(text: str) -> dict:
    """An SDK tool result carrying a single text block."""
    return {"content": [{"type": "text", "text": text}]}


def _to_sdk_tool(spec: ToolSpec, confirm_fn: ConfirmFn):
    """Adapt one ToolSpec into an SDK tool. Action tools ask for confirmation of
    their exact arguments before the handler runs; read tools run directly."""

    @tool(spec.name, spec.description, spec.input_schema)
    async def _sdk_tool(args):
        if spec.kind == "action" and spec.confirm and not confirm_fn(spec.name, args):
            return _say("cancelled by user")
        return await spec.handler(args)

    return _sdk_tool


class SDKRunner(AgentRunner):
    def __init__(self, model: str | None = None, verbose: bool = True):
        self.model = model
        self.verbose = verbose

    def _echo(self, message: Any) -> None:
        if self.verbose:
            print(message)

    def _options(
        self, server_name: str, server, allowed: list[str], system_prompt: str
    ) -> ClaudeAgentOptions:
        # Our tools are read-only or already gated by confirm_fn, so the SDK's
        # own permission prompts are redundant here.
        return ClaudeAgentOptions(
            mcp_servers={server_name: server},
            allowed_tools=allowed,
            system_prompt=system_prompt,
            permission_mode="bypassPermissions",
            model=self.model,
            max_turns=40,
        )

    async def run_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolSpec],
        confirm_fn: ConfirmFn,
    ) -> Any:
        server = create_sdk_mcp_server(
            name="secretary",
            version="0.1.0",
            tools=[_to_sdk_tool(t, confirm_fn) for t in tools],
        )
        allowed = [f"mcp__secretary__{t.name}" for t in tools]
        options = self._options("secretary", server, allowed, system_prompt)

        async for message in query(prompt=user_prompt, options=options):
            self._echo(message)
        # Results are collected through the tools' side effects (e.g.
        # record_artifact writing into the task's sink), not the return value.
        return None

    async def converse(self, task: Task) -> dict:
        captured: dict = {}

        @tool(
            "finalize_setup",
            "Finalize the setup manifest (matching the task's schema) and end setup.",
            {"manifest": dict},
        )
        async def finalize(args):
            captured["manifest"] = args.get("manifest", args)
            return _say("setup finalized")

        server = create_sdk_mcp_server(name="setup", version="0.1.0", tools=[finalize])
        options = self._options(
            "setup", server, ["mcp__setup__finalize_setup"], task.setup_system_prompt()
        )

        # The model asks the user for what it needs and drives the structure; we
        # relay each of the user's replies until it calls finalize_setup.
        async with ClaudeSDKClient(options=options) as client:
            await client.query("Let's set up this task. Ask me what you need.")
            while "manifest" not in captured:
                async for message in client.receive_response():
                    self._echo(message)
                if "manifest" not in captured:
                    await client.query(input("\nyou> "))
        return captured["manifest"]
