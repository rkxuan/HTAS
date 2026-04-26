from copy import deepcopy
import asyncio
from typing import List, Any, Optional

from hats.llm import LLMRegistry, LLM
from hats.llm.format import LLM_Message
from hats.system import Operation, Message
from hats.environment.prompt import PromptSetRegistry, PromptSet
from hats.environment.operations.operation_registry import OperationRegistry

@OperationRegistry.register("Reflection")
class Reflection(Operation):
    def __init__(
        self,
        domain: str,
        model_name: Optional[str],
        branch_factor: int = 3,
        operation_description: str = "Learn from a solution.",
    ):
        super().__init__(operation_description, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.branch_factor = branch_factor
    
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

        tasks = []
        for input_env in input_envs:
            tasks.append(input_env.evaluate(self.llm, self.prompt_set.get_if_correct_prompt, self.prompt_set.get_value_prompt))
        await asyncio.gather(*tasks)

        responses = []
        for input_env in input_envs:
            impossible_words = input_env.impossible_words
            correct_words = input_env.correct_words
            incorrect_words = input_env.incorrect_words
            if len(impossible_words) + len(correct_words) + len(incorrect_words) == 0:
                input_env.reset()
                continue
           
            impossible_words_str = '\n'.join([f'{idx}{word} -- {meaning}' for idx, word, meaning in impossible_words])
            correct_words_str = '\n'.join([f'{idx}{word} -- {meaning}' for idx, word, meaning in correct_words])
            incorrect_words_str = '\n'.join([f'{idx}{word} -- {meaning}' for idx, word, meaning in incorrect_words])
            prompt = self.prompt_set.get_suggest_prompt(input_env.render_board(), 
                                                                impossible_words_str, 
                                                                correct_words_str, 
                                                                incorrect_words_str)
            
            response = await self.llm.agen([LLM_Message(role="user", content=prompt)])
            input_env.reset()
            input_env.hints.append(response)
            responses.append(input_env)
        return [Message(0, self.agent_id, None, self.id, raw_inputs.task, responses)]