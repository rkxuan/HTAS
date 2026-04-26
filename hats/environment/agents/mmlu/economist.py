from typing import Optional

from hats.system import Agent
from hats.environment.operations import DirectAnswer
from hats.environment.agents.agent_registry import AgentRegistry
from hats.environment.prompt import PromptSetRegistry

@AgentRegistry.register('Economist')
class Economist(Agent):
    def __init__(
        self, 
        domain: str,
        model_name: Optional[str] = None
    ):
        super().__init__(domain, model_name)
        self.agent_description = "an economist good at economics, finance, and business."

    def build_agent(self):
        qa_operation = DirectAnswer(self.domain, self.model_name)
        self.add_operation(qa_operation)
        self.input_operations[qa_operation.id] = qa_operation
        self.output_operations[qa_operation.id] = qa_operation
