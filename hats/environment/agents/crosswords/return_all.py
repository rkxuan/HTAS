from typing import Optional

from hats.system import Agent
from hats.environment.operations import ReturnAll
from hats.environment.agents.agent_registry import AgentRegistry
from hats.environment.prompt import PromptSetRegistry


@AgentRegistry.register('CrosswordsReturnAll')
class CrosswordsReturnAll(Agent):             # as the output agent in the system
    def __init__(
        self, 
        domain: str,
        model_name: Optional[str] = None
    ):
        super().__init__(domain, model_name)
        self.agent_description = "final agent to return all potential solutions."

    def build_agent(self):
        final_decision_operation = ReturnAll(self.domain, self.model_name)
        self.add_operation(final_decision_operation)
        self.input_operations[final_decision_operation.id] = final_decision_operation
        self.output_operations[final_decision_operation.id] = final_decision_operation