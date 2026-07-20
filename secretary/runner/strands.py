"""An AWS backend via the Strands Agents SDK, running against Amazon Bedrock.

Handy when Bedrock (IAM/AWS credentials) is easier to reach than an Anthropic or
Google key. Authenticates with your normal AWS credential chain (profile / env /
role) and a region; the model is a Bedrock model id, which can still be Claude
(a Bedrock Anthropic inference profile), just reached through Bedrock.

STATUS: scaffolded, NOT yet run here (needs `pip install strands-agents` and AWS
creds). Shape follows the documented Strands API; spots to confirm against the
installed version are marked TODO. Strands tools are plain Python functions and
it exposes tool calls, so our ToolSpec -> function conversion (with the
read/action confirm gate) maps over directly.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from strands import Agent
from strands.models import BedrockModel  # TODO: confirm class name in the installed version

from ..framework.core import AgentRunner, ConfirmFn, Task, ToolSpec

_PY_TYPE = {str: str, int: int, float: float, bool: bool, list: list, dict: dict}


def _op_detail(name: str, kwargs: dict) -> str:
    """The most relevant argument for an operation, for the one-line trace."""
    if name == "find":
        parts = [kwargs.get("root", "/")]
        if kwargs.get("name_glob"):
            parts.append(kwargs["name_glob"])
        if kwargs.get("kind"):
            parts.append(f"[{kwargs['kind']}]")
        detail = " ".join(parts)
    elif name == "record_artifact":
        detail = f"{kwargs.get('application', '')} {kwargs.get('binary', '')}".strip()
    else:
        detail = kwargs.get("path") or kwargs.get("dir_path") or ""
    detail = str(detail)
    return detail[:97] + "..." if len(detail) > 100 else detail


def _to_strands_tool(ts: ToolSpec, confirm_fn: ConfirmFn, on_call=None):
    """ToolSpec -> Strands tool. Strands reads the function signature + docstring
    for the schema, so we synthesize typed params. Our async handler is bridged
    to sync, the confirm gate is applied for action tools, and the
    {"content":[...]} return is adapted to a plain value. on_call, if given, is
    invoked with (name, kwargs) when the tool actually runs — that's where the
    real-time trace line comes from, since the arguments are complete by then."""
    from strands import tool

    def _invoke(**kwargs) -> Any:
        if on_call is not None:
            on_call(ts.name, kwargs)
        if ts.kind == "action" and ts.confirm and not confirm_fn(ts.name, kwargs):
            return "cancelled by user"
        result = asyncio.run(ts.handler(kwargs))
        try:
            text = result["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return result
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return text

    _invoke.__name__ = ts.name
    _invoke.__doc__ = ts.description
    _invoke.__signature__ = inspect.Signature([
        inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, annotation=_PY_TYPE.get(t, str))
        for n, t in ts.input_schema.items()
    ])
    _invoke.__annotations__ = {n: _PY_TYPE.get(t, str) for n, t in ts.input_schema.items()}
    return tool(_invoke)  # TODO: confirm strands.tool(func) call form


class StrandsRunner(AgentRunner):
    """AgentRunner backed by Strands + Bedrock. Same interface as SDKRunner."""

    def __init__(self, model: str | None = None, region: str = "us-east-1", verbose: bool = True):
        self.model = model          # None -> Strands' default (an Anthropic model on Bedrock)
        self.region = region
        self.verbose = verbose

    def _model(self):
        if self.model is None:
            return None
        return BedrockModel(model_id=self.model, region_name=self.region)  # TODO: confirm kwargs

    def _render(self, event) -> None:
        """Stream assistant text (dimmed). Tool lines print from the wrapper."""
        if self.verbose and isinstance(event, dict) and event.get("data"):
            from .. import console
            console.think(event["data"])
            self._mid_line = not event["data"].endswith("\n")

    def _trace(self, name: str, kwargs: dict) -> None:
        if not self.verbose:
            return
        from .. import console
        if getattr(self, "_mid_line", False):
            print()  # close a partial line of streamed text before the op line
            self._mid_line = False
        console.op(name, _op_detail(name, kwargs))

    async def run_agent(self, system_prompt: str, user_prompt: str,
                        tools: list[ToolSpec], confirm_fn: ConfirmFn) -> Any:
        self._mid_line = False
        kwargs = {"system_prompt": system_prompt,
                  "tools": [_to_strands_tool(t, confirm_fn, on_call=self._trace) for t in tools],
                  # silence Strands' own stdout printer so it doesn't double with ours
                  "callback_handler": None}
        model = self._model()
        if model is not None:
            kwargs["model"] = model
        agent = Agent(**kwargs)

        stream = getattr(agent, "stream_async", None)
        if stream is None:
            result = await asyncio.to_thread(agent, user_prompt)  # TODO: prefer invoke_async
            if self.verbose:
                print(result)
        else:
            async for event in stream(user_prompt):
                self._render(event)
            if self.verbose and self._mid_line:
                print()  # newline after trailing streamed text
        return None  # results collected via tool side effects (record_artifact -> sink)

    async def converse(self, task: Task) -> dict:
        from strands import tool
        captured: dict = {}

        def finalize_setup(manifest: dict) -> str:
            """Finalize the setup manifest (matching the task's schema) and end setup."""
            captured["manifest"] = manifest
            return "setup finalized"

        kwargs = {"system_prompt": task.setup_system_prompt(), "tools": [tool(finalize_setup)]}
        model = self._model()
        if model is not None:
            kwargs["model"] = model
        agent = Agent(**kwargs)

        prompt = "Let's set up this task. Ask me what you need."
        while "manifest" not in captured:
            out = await asyncio.to_thread(agent, prompt)
            if self.verbose:
                print(out)
            if "manifest" in captured:
                break
            prompt = input("\nyou> ")
        return captured["manifest"]
