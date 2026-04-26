from typing import Optional

from hats.system import Agent
from hats.environment.operations import CodeWriting
from hats.environment.agents.agent_registry import AgentRegistry
from hats.environment.prompt import PromptSetRegistry

@AgentRegistry.register('BugFixer')
class BugFixer(Agent):
    def __init__(
        self, 
        domain: str,
        model_name: Optional[str] = None
    ):
        super().__init__(domain, model_name)
        self.agent_description = "a bug fixer to provide modified and improved python code based on the current overall code design."

    def build_agent(self):
        qa_operation = CodeWriting(self.domain, self.model_name)
        self.add_operation(qa_operation)
        self.input_operations[qa_operation.id] = qa_operation
        self.output_operations[qa_operation.id] = qa_operation