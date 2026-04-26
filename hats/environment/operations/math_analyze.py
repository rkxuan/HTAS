#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict, Counter
from typing import List, Any, Optional
import re

from hats.llm.format import LLM_Message
from hats.system import Operation, Message
from hats.environment.prompt import PromptSetRegistry, PromptSet
from hats.llm import LLMRegistry, LLM
from hats.environment.operations.operation_registry import OperationRegistry
from hats.environment.agents.agent_registry import AgentRegistry
from hats.environment.tools.coding.python_executor import execute_code_get_return


@OperationRegistry.register("MathAnalyze")
class MathAnalyze(Operation):
    def __init__(self, domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "Analyze the given math problem.",
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
        role_prompt = self.prompt_set.get_role_prompt(self.agent_role) 
        agent_role = self.prompt_set.get_role(self.agent_role)
        constraint = self.prompt_set.get_thought_constraint(self.agent_role)
        prompt = self.prompt_set.get_message_prompt(
            raw_inputs=raw_inputs, neighbor_messages=neighbor_messages, 
            self_messages=self_messages, role=self.agent_role
        )
        llm_message = [LLM_Message(role="system", content=f"{role_prompt}{constraint}"),
                   LLM_Message(role="user", content=prompt)]
        response = await self.llm.agen(llm_message)
        if self.agent_role == 'MathProgrammingExpert':
            try:
                answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
                response += f"\nthe answer is {answer}"
            except:
                print("can not execute:\n", response)
        return [Message(0, self.agent_id, agent_role, self.id, raw_inputs.task, response)]




