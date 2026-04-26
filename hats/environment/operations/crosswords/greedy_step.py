from copy import deepcopy
import asyncio
from typing import List, Any, Optional

from hats.llm import LLMRegistry, LLM
from hats.llm.format import LLM_Message
from hats.system import Operation, Message
from hats.environment.prompt import PromptSetRegistry, PromptSet
from hats.environment.tools.crosswords.parser import parse_response
from hats.environment.operations.operation_registry import OperationRegistry

@OperationRegistry.register("GreedyStep")
class GreedyStep(Operation):
    def __init__(
        self,
        domain: str,
        model_name: Optional[str],
        operation_description: str = "Perform greedy steps.",
    ):
        super().__init__(operation_description, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    @property
    def operation_name(self):
        return self.__class__.__name__

    async def _execute(
        self,
        raw_inputs: Message,
        neighbor_messages: List[Message],
        self_messages: List[Message],
        **kwargs
    ):

        input_envs = []
        for neighbor_message in neighbor_messages:
            textual_output = neighbor_message.textual_output
            if textual_output is None:
                continue
            if isinstance(textual_output, list):
                input_envs.extend(textual_output)
            else:
                raise ValueError("Unsupported type for neighbor_message textual_output: {}".format(type(textual_output)))

        for self_message in self_messages:
            textual_output = self_message.textual_output
            if textual_output is None:
                continue
            if isinstance(textual_output, list):
                input_envs.extend(textual_output)
            else:
                raise ValueError("Unsupported type for self_message textual_output: {}".format(type(textual_output)))

        input_envs = input_envs if input_envs else raw_inputs.task

        responses = []
        for env in input_envs:
            prompt = self.prompt_set.get_propose_prompt(env.render())
            response = await self.llm.agen([LLM_Message(role="user", content=prompt)])
            next_steps = parse_response(response)
            _env = deepcopy(env)
            for next_step, _ in next_steps:
                try:
                    _env.step(next_step, allow_change=False)
                    responses.append(_env)
                    break
                except:
                    continue

        return [Message(0, self.agent_id, None, self.id, raw_inputs.task, responses)]
