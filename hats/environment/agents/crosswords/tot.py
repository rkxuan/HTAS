from typing import Optional


from hats.system import Agent
from hats.environment.agents.agent_registry import AgentRegistry
from hats.environment.operations import BranchingStep


@AgentRegistry.register('CrosswordsToT')
class CrosswordsToT(Agent):
    def __init__(
        self, 
        domain: str, 
        model_name: Optional[str] = None, 
        depth=4, 
        branch_factor=2, 
        prune=True
    ):
        self.depth = depth
        self.branch_factor = branch_factor
        self.prune = prune
        super().__init__(domain, model_name)
        self.agent_description = "an agent to explore potential solutions with thought of tree."

    def build_agent(self):
        step = BranchingStep(self.domain, self.model_name, branch_factor=self.branch_factor, prune=self.prune)
        self.add_operation(step)
        self.input_operations[step.id] = step
        for _ in range(self.depth):
            next_step = BranchingStep(self.domain, self.model_name, branch_factor=self.branch_factor, prune=self.prune)
            self.add_operation(next_step)
            step.add_successor(next_step)
            step = next_step
        self.output_operations[step.id] = step
