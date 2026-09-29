"""
Contains LLM implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Any, List

if TYPE_CHECKING:
    from anthropic import Anthropic
    from openai import OpenAI


@dataclass
class Message:
    """A single chat message. Role is one of: system, user, assistant."""

    role: str
    content: str
    # tags: List[str] | None = None


@dataclass
class LLMResponse:
    """Normalized response across providers."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thinking: str | None = None
    raw: Any = None


class BaseLLM(ABC):
    """Abstract base class defining the unified LLM interface."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system = system
        self._client = self._build_client(api_key)
        self.kwargs = kwargs

    # --- Abstract Functions --- #
    @abstractmethod
    def _build_client(self, api_key: Optional[str]) -> Any:
        """Instantiate the provider's SDK client."""

    @abstractmethod
    def get_response(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Send messages; return the full response."""

    # --- Helper Functions --- #
    def update_system(self, system: str | None) -> None:
        """Replace the system prompt used for subsequent requests."""
        self.system = system

    def chat(self, prompt: str, history: Optional[list[Message]] = None) -> LLMResponse:
        """Send one user prompt (with optional prior history)."""
        messages = list(history or [])
        messages.append(Message(role="user", content=prompt))
        return self.get_response(messages)


class ClaudeLLM(BaseLLM):
    """Anthropic Claude implementation.

    Parameters
    ----------
    model: str
        Must be one of models listed in https://platform.claude.com/docs/en/about-claude/models/overview
        As of 2026, it can be one of ['claude-haiku-4-5', 'claude-sonnet-4-6', 'claude-opus-4-7']
    """

    def __init__(
        self,
        model: str,
        thinking: bool = False,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        cache: Literal["first"] | None = None,
        **kwargs,
    ):
        if thinking and temperature != 1:
            raise ValueError(
                "temperature must be 1 when thinking=True (Anthropic API requirement)."
            )
        if cache not in ("first", None):
            raise ValueError(f"cache must be 'first' or None, got {cache!r}")
        self.thinking = thinking
        self.cache = cache
        super().__init__(
            model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    def _build_client(self, api_key: Optional[str]) -> Anthropic:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
        return Anthropic(api_key=api_key) if api_key else Anthropic()

    def _prepare(self, messages: list[Message]) -> dict:
        msgs = []
        for i, m in enumerate(messages):
            if self.cache == "first" and i == 0:
                msgs.append(
                    {
                        "role": m.role,
                        "content": [
                            {
                                "type": "text",
                                "text": m.content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            else:
                msgs.append({"role": m.role, "content": m.content})

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": msgs,
        }
        if self.thinking:
            payload["thinking"] = {"type": "adaptive"}
        if self.system:
            payload["system"] = self.system
        payload.update(self.kwargs)
        return payload

    def get_response(self, messages: list[Message], **kwargs) -> LLMResponse:
        payload = self._prepare(messages)
        payload.update(kwargs)
        resp = self._client.messages.create(**payload)
        text = "".join(b.text for b in resp.content if b.type == "text")
        thinking = (
            "".join(b.thinking for b in resp.content if b.type == "thinking") or None
        )
        return LLMResponse(
            content=text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0),
            cache_write_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0),
            thinking=thinking,
            raw=resp,
        )


class OpenAILLM(BaseLLM):
    """OpenAI-compatible implementation (works with vLLM, OpenAI, and any OpenAI-compatible server).

    Parameters
    ----------
    model : str
        Model name, e.g. "gpt-4o" or "Qwen/Qwen2.5-1.5B-Instruct".
    base_url : str | None
        Override the API base URL, e.g. "http://localhost:8000/v1" for a local vLLM server.
        Defaults to the official OpenAI endpoint.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 8192,
        base_url: str | None = None,
        **kwargs,
    ):
        self.base_url = base_url
        super().__init__(
            model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        self._client: OpenAI

    def _build_client(self, api_key: Optional[str]) -> OpenAI:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(api_key=api_key or "none", **kwargs)

    def _prepare(self, messages: list[Message]) -> dict:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        for m in messages:
            msgs.append({"role": m.role, "content": m.content})
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": msgs,
        }
        if self.model in ("gpt-5.4-mini"):
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens
        payload.update(self.kwargs)
        return payload

    def get_response(self, messages: list[Message], **kwargs) -> LLMResponse:
        payload = self._prepare(messages)
        payload.update(kwargs)
        resp = self._client.chat.completions.create(**payload)
        choice = resp.choices[0]
        usage = (
            resp.usage or type("u", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        )
        return LLMResponse(
            content=choice.message.content or "",
            model=resp.model,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            raw=resp,
        )


# --- Example usage --------------------------------------------------------

if __name__ == "__main__":
    # Requires ANTHROPIC_API_KEY in the environment.

    claude = ClaudeLLM(model="claude-sonnet-4-6", thinking=True)
    messages = [
        Message(
            role="assistant",
            content="The user asked for a poem. I need to think of a good one.",
        ),
        Message(role="user", content="What did you just say?"),
    ]
    # print("Claude:", claude.chat("What is 2 + 2?").content)
    response = claude.get_response(messages)
    print(response)
    print(response.content)
