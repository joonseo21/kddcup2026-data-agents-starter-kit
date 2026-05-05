from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
import re
from openai import APIError, OpenAI


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def complete(self, messages: list[ModelMessage]) -> str:
        #/////////
        '''
        total_chars = sum(len(m.content) for m in messages)
        print(f"[DEBUG] 메시지 수: {len(messages)}, 총 글자수: {total_chars}")
        for m in messages:
            print(f"[DEBUG] role={m.role}, chars={len(m.content)}")
            if m.role == "assistant":
                print(f" assistant 내용 앞 200자: {m.content[:200]}")
            if m.role == "user" and m.content.startswith("Observation"):
                print(f" observation 내용: {m.content[:300]}")
                '''
        #//////////

        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": message.role, "content": message.content} for message in messages],
                temperature=self.temperature,
                #extra_body={"enable_thinking": False},
            )
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        #content = choices[0].message.content
        content = choices[0].message.content.encode('utf-8', errors='ignore').decode('utf-8')
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        print(f"[DEBUG] 응답 원문: {repr(content[:500])}")
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
