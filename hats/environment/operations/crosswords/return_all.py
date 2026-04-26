
import asyncio
from typing import List, Any, Optional

from hats.llm import LLMRegistry, LLM
from hats.system import Operation, Message
from hats.environment.prompt import PromptSetRegistry, PromptSet
from hats.environment.operations.operation_registry import OperationRegistry

class ReturnAll(Operation):
    def __init__(
        self, 
        domain: str, 
        model_name: Optional[str] = None,
        operation_description: str = "Return all candidates for evaluation.",
    ):
        super().__init__(operation_description, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)

    async def _execute(
        self, 
        raw_inputs: Message,
        neighbor_messages: List[Message],
        self_messages: List[Message],
        **kwargs
    ):    
        input_envs = []
        for neighbor_message in neighbor_messages:
            if isinstance(neighbor_message.textual_output, list):
                input_envs.extend(neighbor_message.textual_output)

        return [Message(0, None, None, self.id, raw_inputs.task, input_envs)]