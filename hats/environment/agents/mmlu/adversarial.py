from typing import Optional

from hats.system import Agent
from hats.environment.operations import AdversarialAnswer
from hats.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('Adversarial')
class Adversarial(Agent):        # could also be used as agent with backdoor, just replace KnowledgeableExpert to Adversarial in testing time.
    def __init__(
        self, 
        domain: str,
        model_name: Optional[str] = None
    ):
        super().__init__(domain, model_name)
        self.agent_description = "a redteam member to generate adversarial answers to mislead other agents."

    def build_agent(self):
        adversarial_operation = AdversarialAnswer(self.domain, self.model_name)
        self.add_operation(adversarial_operation)
        self.input_operations[adversarial_operation.id] = adversarial_operation
        self.output_operations[adversarial_operation.id] = adversarial_operation
