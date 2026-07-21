"""A Gemini backend via Google's Agent Development Kit (ADK), as an alternative
AgentRunner to SDKRunner.

STATUS: scaffolded, NOT yet run. Google ADK isn't a base dependency and isn't
installed in the dev image, so this module is untested end to end. The shape
follows ADK's documented API (Agent + FunctionTool + Runner + session service),
but the spots marked TODO are where the ADK specifics must be confirmed against
the installed version before relying on it. Install with `pip install google-adk`.

It exists because the framework already abstracts the model behind AgentRunner,
so adding Gemini is a contained job: convert our ToolSpecs to ADK function tools
(carrying the same read/action confirmation gate) and drive ADK's runner. None
of ProfileTask / tools.py / the container code changes.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

# Top-level imports so a missing ADK fails here, at import, not mid-run. This
# module is only imported when you explicitly want the Gemini path
# (`from secretary.runner.adk import ADKRunner`), so it never burdens the
# Claude/SDK path or `import secretary`.
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from ..framework.core import AgentRunner, ConfirmFn, Task, ToolSpec

_PY_TYPE = {str: str, int: int, float: float, bool: bool, list: list, dict: dict}


def _to_adk_tool(ts: ToolSpec, confirm_fn: ConfirmFn) -> FunctionTool:
    """Wrap one ToolSpec as an ADK FunctionTool.

    ADK builds a tool's schema from the wrapped function's SIGNATURE (typed
    params + docstring), whereas our ToolSpec carries an input_schema dict. So we
    synthesize a function with matching parameters, then let ADK introspect it.

    The wrapper also applies the read/action confirmation gate (same semantics as
    SDKRunner) and adapts our handler's Anthropic-style {"content":[...]} return
    into the plain value ADK expects a tool to return.
    """

    async def _invoke(**kwargs) -> Any:
        if ts.kind == "action" and ts.confirm and not confirm_fn(ts.name, kwargs):
            return "cancelled by user"
        result = await ts.handler(kwargs)
        # our handlers return {"content": [{"type": "text", "text": "..."}]}
        try:
            text = result["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return result
        try:
            return json.loads(text)  # most of our tools emit JSON
        except (ValueError, TypeError):
            return text

    # give ADK a real signature + docstring to introspect
    params = [
        inspect.Parameter(
            name, inspect.Parameter.KEYWORD_ONLY, annotation=_PY_TYPE.get(typ, str)
        )
        for name, typ in ts.input_schema.items()
    ]
    _invoke.__name__ = ts.name
    _invoke.__doc__ = ts.description
    _invoke.__signature__ = inspect.Signature(
        params
    )  # TODO: confirm ADK reads __signature__ (vs real params)
    _invoke.__annotations__ = {
        n: _PY_TYPE.get(t, str) for n, t in ts.input_schema.items()
    }
    return FunctionTool(
        _invoke
    )  # TODO: confirm FunctionTool(func) is the current constructor


class ADKRunner(AgentRunner):
    """AgentRunner backed by Google ADK / Gemini. Same interface as SDKRunner."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        app_name: str = "artifact-secretary",
        verbose: bool = True,
    ):
        self.model = model
        self.app_name = app_name
        self.verbose = verbose

    async def _run_once(
        self, instruction: str, user_prompt: str, adk_tools: list
    ) -> list[str]:
        agent = Agent(
            name="secretary", model=self.model, instruction=instruction, tools=adk_tools
        )
        session_service = InMemorySessionService()
        # TODO: confirm create_session is sync vs async in the installed ADK
        session = session_service.create_session(
            app_name=self.app_name, user_id="local", session_id="s1"
        )
        runner = Runner(
            agent=agent, app_name=self.app_name, session_service=session_service
        )

        msg = types.Content(role="user", parts=[types.Part(text=user_prompt)])
        texts: list[str] = []
        # TODO: confirm run_async signature + how final text is surfaced on events
        async for event in runner.run_async(
            user_id="local", session_id=session.id, new_message=msg
        ):
            if self.verbose:
                print(event)
            content = getattr(event, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    if getattr(part, "text", None):
                        texts.append(part.text)
        return texts

    async def run_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolSpec],
        confirm_fn: ConfirmFn,
    ) -> Any:
        adk_tools = [_to_adk_tool(t, confirm_fn) for t in tools]
        await self._run_once(system_prompt, user_prompt, adk_tools)
        # results are collected via the tools' side effects (e.g. record_artifact
        # writing into ProfileTask's sink), same as SDKRunner.
        return None

    async def converse(self, task: Task) -> dict:
        """Interactive setup: give the agent a finalize tool and loop until it's
        called. Mirrors SDKRunner.converse. TODO: verify ADK multi-turn session
        reuse (re-running the runner with the same session_id to continue)."""
        captured: dict = {}

        async def finalize_setup(manifest: dict) -> str:
            """Finalize the setup manifest (matching the task's schema) and end setup."""
            captured["manifest"] = manifest
            return "setup finalized"

        finalize = FunctionTool(finalize_setup)
        agent = Agent(
            name="setup",
            model=self.model,
            instruction=task.setup_system_prompt(),
            tools=[finalize],
        )
        session_service = InMemorySessionService()
        session = session_service.create_session(
            app_name=self.app_name, user_id="local", session_id="setup"
        )
        runner = Runner(
            agent=agent, app_name=self.app_name, session_service=session_service
        )

        prompt = "Let's set up this task. Ask me what you need."
        while "manifest" not in captured:
            msg = types.Content(role="user", parts=[types.Part(text=prompt)])
            async for event in runner.run_async(
                user_id="local", session_id=session.id, new_message=msg
            ):
                if self.verbose:
                    print(event)
            if "manifest" in captured:
                break
            prompt = input("\nyou> ")
        return captured["manifest"]
