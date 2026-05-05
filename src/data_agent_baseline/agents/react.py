from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
    get_operator_hint,          #M1
    build_schema_context,       #M2
    validate_answer,            #M3
    M4_PLAN_SYSTEM_PROMPT,      # M4
    M4_VALIDATE_SYSTEM_PROMPT,  # M4
    M4_ANSWER_SYSTEM_PROMPT,    # M4
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    use_M0Plus: bool = False            # M0+: 남은 스텝 카운터
    use_M0PP: bool = False              # M0++: 이전 observation 압축
    use_operator_forcing: bool = False  # M1
    use_schema_linking: bool = False    # M2
    use_validator: bool = False         # M3
    use_DAG: bool = False               # M4 : M2와의 상성 확인 중요

def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState, step_index: int, schema_context: str | None = None,) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task, schema_context=schema_context)))

        completed_actions = [s.action for s in state.steps]
        remaining = self.config.max_steps - step_index + 1

        for i, step in enumerate(state.steps):
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            is_last = (i == len(state.steps) - 1)

            ##M0PP##########################################
            obs = step.observation
            if self.config.use_M0PP and not is_last:
                content = obs.get("content", {})
                if isinstance(content, dict):
                    rendered = json.dumps(content, ensure_ascii=False)
                    if len(rendered) > 800:
                        tool_name = obs.get("tool", "unknown")
                        obs = {
                            **obs,
                            "content": f"[previously viewed {tool_name} result, {len(rendered)} chars, omitted to save context]",
                        }
            #################################################

            ##M0Plus########################################
            obs_content = build_observation_prompt(
                obs,
                remaining_steps=remaining if (is_last and self.config.use_M0Plus) else None,
            )
            #################################################

            ##M1############################################
            if is_last and self.config.use_operator_forcing:
                hint = get_operator_hint(step_index, completed_actions)
                if hint:
                    obs_content += f"\n{hint}"
            #################################################

            messages.append(ModelMessage(role="user", content=obs_content))

        return messages
    
    ##M4############################
    def _run_m4(self, task: PublicTask, state: AgentRuntimeState, schema_context: str | None) -> None:
        step_index = 1

        # --- Phase 1: Plan 수립 ---
        remaining_info = ""
        if self.config.use_M0Plus:
            remaining_info = (
                f"You have a maximum budget of {self.config.max_steps} tool steps. You use a step whenever you return a result.\n\n"
            )

        plan_messages = [
            ModelMessage(role="system", content=M4_PLAN_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=(
                    f"Question: {task.question}\n\n"
                    + remaining_info
                    + (
                        f"Schema Context:\n{schema_context}"
                        if schema_context
                        else "No schema context provided. Include list_context as first step."
                    )
                ),
            ),
        ]
        plan_raw = self.model.complete(plan_messages)


        try:
            plan_text = _strip_json_fence(plan_raw)
            plan_text = plan_text.strip()
            plan: list[dict] = json.loads(plan_text)
            if not isinstance(plan, list):
                raise ValueError("Plan must be a JSON array.")
        except Exception as exc:
            plan = [{"action": "list_context", "action_input": {"max_depth": 4}}]
            state.steps.append(StepRecord(
                step_index=step_index,
                thought=f"Plan parsing failed: {exc}, falling back to list_context",
                action="__plan_error__",
                action_input={},
                raw_response=plan_raw,
                observation={"ok": False, "error": str(exc)},
                ok=False,
            ))
            step_index += 1

        state.steps.append(StepRecord(
            step_index=step_index,
            thought="M4: DAG plan created",
            action="__plan__",
            action_input={"plan": plan},
            raw_response=plan_raw,
            observation={"ok": True, "plan": plan},
            ok=True,
        ))
        step_index += 1

        # --- Phase 2: plan 실행 (컨텍스트 리셋 - 각 실행 결과만 누적) ---
        execution_results: list[dict] = []  # 압축된 결과 누적

        for node in plan:
            if step_index > self.config.max_steps:
                break

            action = node.get("action", "")
            action_input = node.get("action_input", {})

            if not isinstance(action_input, dict):
                action_input = {}

            try:
                tool_result = self.tools.execute(task, action, action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": action,
                    "content": tool_result.content,
                }
                # 실행 결과를 압축해서 누적 (Phase 3에서 사용)
                content_str = json.dumps(tool_result.content, ensure_ascii=False)
                execution_results.append({
                    "action": action,
                    "action_input": action_input,
                    "result_summary": content_str[:1500] + ("...[truncated]" if len(content_str) > 1500 else ""),
                })

                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought=f"M4: executing plan step {action}",
                    action=action,
                    action_input=action_input,
                    raw_response=json.dumps(node),
                    observation=observation,
                    ok=tool_result.ok,
                ))

                # terminal 툴이 plan에 포함된 경우 (비정상이지만 처리)
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    return

            except Exception as exc:
                observation = {"ok": False, "error": str(exc)}
                execution_results.append({
                    "action": action,
                    "action_input": action_input,
                    "result_summary": f"ERROR: {exc}",
                })
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought=f"M4: plan step {action} failed",
                    action="__error__",
                    action_input=action_input,
                    raw_response=json.dumps(node),
                    observation=observation,
                    ok=False,
                ))

            step_index += 1

        # --- Phase 3: 결과 요약 → answer (컨텍스트 리셋) ---
        if step_index > self.config.max_steps:
            state.failure_reason = "Agent did not submit an answer within max_steps."
            return

        results_text = json.dumps(execution_results, ensure_ascii=False, indent=2)
        remaining_info = ""

        if self.config.use_M0Plus:
            remaining = self.config.max_steps - step_index + 1
            remaining_info = (
                f"You have {remaining} steps remaining. You use a step whenever you return a result - return a final answer before they run out.\n"
            )

        answer_messages = [
            ModelMessage(role="system", content=M4_ANSWER_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=(
                    remaining_info
                    + f"Question: {task.question}\n\n"
                    f"Execution results:\n{results_text}\n\n"
                    "Now call the answer tool with the final result table."
                ),
            ),
        ]

        answer_raw = self.model.complete(answer_messages)

        try:
            model_step = parse_model_step(answer_raw)

            ##M3 validator############################
            if self.config.use_validator and model_step.action == "answer":
                columns = model_step.action_input.get("columns", [])
                rows = model_step.action_input.get("rows", [])
                valid, reason = validate_answer(columns, rows)
                if not valid:
                    # 한 번만 재시도
                    retry_messages = answer_messages + [
                        ModelMessage(role="assistant", content=answer_raw),
                        ModelMessage(role="user", content=f"[Validator] Answer rejected: {reason} Please fix and resubmit."),
                    ]
                    answer_raw = self.model.complete(retry_messages)
                    model_step = parse_model_step(answer_raw)
            ###########################################

            tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
            observation = {
                "ok": tool_result.ok,
                "tool": model_step.action,
                "content": tool_result.content,
            }
            state.steps.append(StepRecord(
                step_index=step_index,
                thought=model_step.thought,
                action=model_step.action,
                action_input=model_step.action_input,
                raw_response=answer_raw,
                observation=observation,
                ok=tool_result.ok,
            ))
            if tool_result.is_terminal:
                state.answer = tool_result.answer

        except Exception as exc:
            state.steps.append(StepRecord(
                step_index=step_index,
                thought="",
                action="__error__",
                action_input={},
                raw_response=answer_raw,
                observation={"ok": False, "error": str(exc)},
                ok=False,
            ))
    #################################

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()

        ##M2############################
        schema_context: str | None = None
        if self.config.use_schema_linking:
            schema_context = build_schema_context(task)
        #################################

        ##M4############################
        if self.config.use_DAG:
            self._run_m4(task, state, schema_context)
            if state.answer is None and state.failure_reason is None:
                state.failure_reason = "Agent did not submit an answer within max_steps."
            return AgentRunResult(
                task_id=task.task_id,
                answer=state.answer,
                steps=list(state.steps),
                failure_reason=state.failure_reason,
            )
        #################################

        for step_index in range(1, self.config.max_steps + 1):
            raw_response = self.model.complete(self._build_messages(task, state, step_index, schema_context=schema_context))  # M2
            try:
                model_step = parse_model_step(raw_response)
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)

                ##M3############################
                # answer 툴 호출 시 validator 통과 여부 확인
                if self.config.use_validator and model_step.action == "answer":
                    columns = model_step.action_input.get("columns", [])
                    rows = model_step.action_input.get("rows", [])
                    valid, reason = validate_answer(columns, rows)
                    if not valid:
                        # validator 실패 → error observation으로 다시 루프
                        observation = {
                            "ok": False,
                            "tool": "answer",
                            "error": f"[Validator] Answer rejected: {reason} Please fix and resubmit.",
                        }
                        state.steps.append(
                            StepRecord(
                                step_index=step_index,
                                thought=model_step.thought,
                                action=model_step.action,
                                action_input=model_step.action_input,
                                raw_response=raw_response,
                                observation=observation,
                                ok=False,
                            )
                        )
                        continue  # 루프 계속
                #################################

                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
                
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None:
            (f"[DEBUG] no answer was written")
            if state.failure_reason is None:
                 state.failure_reason = "Agent did not submit an answer within max_steps."
        

        results = AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
        return results