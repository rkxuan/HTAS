from typing import Optional

from hats.system import Agent
from hats.environment.operations import CodeWriting
from hats.environment.agents.agent_registry import AgentRegistry
from hats.environment.prompt import PromptSetRegistry

@AgentRegistry.register('TestAnalyst')
class TestAnalyst(Agent):
    def __init__(
        self, 
        domain: str,
        model_name: Optional[str] = None
    ):
        super().__init__(domain, model_name)
        self.agent_description = "a test analyst to provides problems in the current code or solution based on the test data."

    def build_agent(self):
        qa_operation = CodeWriting(self.domain, self.model_name)
        self.add_operation(qa_operation)
        self.input_operations[qa_operation.id] = qa_operation
        self.output_operations[qa_operation.id] = qa_operation