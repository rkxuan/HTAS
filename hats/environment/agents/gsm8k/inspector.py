from typing import Optional

from hats.system import Agent
from hats.environment.operations import MathAnalyze
from hats.environment.agents.agent_registry import AgentRegistry
from hats.environment.prompt import PromptSetRegistry

@AgentRegistry.register('Inspector')
class Inspector(Agent):
    def __init__(
        self, 
        domain: str,
        model_name: Optional[str] = None
    ):
        super().__init__(domain, model_name)
        self.agent_description = "an inspector to check whether the logic/calculation of the problem solving and analysis process is correct(if present)."

    def build_agent(self):
        qa_operation = MathAnalyze(self.domain, self.model_name)
        self.add_operation(qa_operation)
        self.input_operations[qa_operation.id] = qa_operation
        self.output_operations[qa_operation.id] = qa_operation