from copy import deepcopy
import asyncio
from typing import List, Any, Optional

from hats.llm import LLMRegistry, LLM
from hats.llm.format import LLM_Message
from hats.system import Operation, Message
from hats.environment.prompt import PromptSetRegistry, PromptSet
from hats.environment.tools.crosswords.parser import parse_response
from hats.environment.operations.operation_registry import OperationRegistry

@OperationRegistry.register("BranchingStep")
class BranchingStep(Operation):
    def __init__(
        self,
        domain: str,
        model_name: Optional[str],
        branch_factor: int = 3,
        prune: bool = True,
        operation_description: str = "Perform a step in tree search.",
    ):
        super().__init__(operation_description, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.branch_factor = branch_factor
        self.prune = prune

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

        next_envs = []
        for env in input_envs:
            prompt = self.prompt_set.get_propose_prompt(env.render())
            response = await self.llm.agen([LLM_Message(role="user", content=prompt)])
            next_steps = parse_response(response)[:self.branch_factor]
            for next_step, _ in next_steps:
                _env = deepcopy(env)
                _env.step(next_step)
                next_envs.append(_env)

        if len(next_envs) == 0:
            return [Message(1, self.agent_id, None, self.id, raw_inputs.task, [])]

        if self.prune:
            tasks = []
            for _env in next_envs:
                tasks.append(_env.check_termination(self.llm, self.prompt_set.get_value_prompt))
            await asyncio.gather(*tasks)

        responses = []
        for _env in next_envs:
            if not _env.extendable:
                continue
            responses.append(_env)

        if len(responses) == 0:
            return [Message(1, self.agent_id, None, self.id, raw_inputs.task, [])]

        return [Message(0, self.agent_id, None, self.id, raw_inputs.task, responses)]